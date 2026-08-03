#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через IONEX-файлы.
ИСПРАВЛЕНИЯ:
1. Умный выбор версии файла (codg для старых, c1pg для новых).
2. Синтетика ТОЛЬКО для Target, если нет данных (для демо). Фон всегда реальный.
3. Жесткая ошибка для будущих дат.
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IonosphereCollector:
    CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")
        
        self.gnss_available = self._check_gnssanalysis()

    def _check_gnssanalysis(self):
        try:
            from gnssanalysis.ionex import IonexMap
            logger.info("✅ gnssanalysis загружена")
            return True
        except ImportError:
            logger.warning("⚠️ gnssanalysis не установлена. Реальные данные недоступны.")
            return False

    @staticmethod
    def _get_ionex_filename(date: datetime) -> str | None:
        """
        Выбирает тип файла в зависимости от даты.
        - codg: Финальные (высокая точность, задержка ~14 дней).
        - c1pg: Прогнозные (низкая задержка, низкая точность).
        - None: Дата в будущем (данных не существует).
        """
        now = datetime.now(timezone.utc)
        days_diff = (now - date).days

        if days_diff > 15:
            # Прошлое: берем финальные данные
            return f"codg{date.year}{date.timetuple().tm_yday:03d}.01i"
        elif days_diff >= 0:
            # Недавнее (до 15 дней): берем прогнозные данные
            return f"c1pg{date.year}{date.timetuple().tm_yday:03d}.01i"
        else:
            # Будущее: данных нет физически
            return None

    def _generate_synthetic_tec(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Генерирует синтетические данные ТОЛЬКО для отладки/демо."""
        records = []
        current = start_time
        # Базовая модель: зависит от широты + суточный цикл + шум
        base_tec = max(5, 30 - abs(lat) * 0.5)
        
        while current <= end_time:
            hour = current.hour
            daily_cycle = 5 * np.sin((hour - 12) * np.pi / 12) # Пик около полудня
            noise = np.random.normal(0, 1.5) # Меньший шум для демо
            
            tec = max(0, base_tec + daily_cycle + noise)
            records.append({'time': current, 'tec_value': tec})
            current += timedelta(hours=2)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('time', inplace=True)
        return df

    def fetch_tec_for_point(self, lat, lon, start_time, end_time, use_synthetic_if_missing=False) -> pd.DataFrame:
        """
        Получает TEC для точки.
        use_synthetic_if_missing: 
          - True: Если нет реальных данных, вернет синтетику (для Target событий).
          - False: Если нет реальных данных, вернет пустой DataFrame (для Background).
        """
        if not self.gnss_available:
            if use_synthetic_if_missing:
                logger.warning("⚠️ gnssanalysis нет, используем синтетику.")
                return self._generate_synthetic_tec(lat, lon, start_time, end_time)
            else:
                logger.error("❌ gnssanalysis нет. Не можем получить реальный фон.")
                return pd.DataFrame()

        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current_date <= end_time:
            filename = self._get_ionex_filename(current_date)
            
            if filename is None:
                # Дата в будущем
                logger.error(f"❌ Дата {current_date.date()} в будущем. IONEX файлов не существует.")
                current_date += timedelta(days=1)
                continue

            local_path = self.cache_dir / filename
            url = f"{self.CDDIS_BASE}/{current_date.year}/{current_date.timetuple().tm_yday:03d}/{filename}"
            
            # Скачивание
            if not local_path.exists():
                try:
                    logger.debug(f"📥 Скачивание: {url}")
                    urllib.request.urlretrieve(url, local_path, timeout=30)
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось скачать {filename}: {e}")
                    current_date += timedelta(days=1)
                    continue
            
            # Парсинг
            if local_path.exists() and local_path.stat().st_size > 0:
                try:
                    from gnssanalysis.ionex import IonexMap
                    ionmap = IonexMap.from_file(str(local_path))
                    
                    records = []
                    for hour in range(0, 24, 2):
                        time_query = current_date.replace(hour=hour)
                        tec_val = ionmap.get_tec(lat=lat, lon=lon, time=time_query)
                        if tec_val is not None and not np.isnan(tec_val):
                            records.append({'time': time_query, 'tec_value': float(tec_val)})
                    
                    if records:
                        df = pd.DataFrame(records)
                        df.set_index('time', inplace=True)
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"❌ Ошибка парсинга {filename}: {e}", exc_info=True)
            
            current_date += timedelta(days=1)
        
        if not all_dfs:
            if use_synthetic_if_missing:
                logger.info(f"🔄 Нет реальных данных для ({lat}, {lon}), используем синтетику.")
                return self._generate_synthetic_tec(lat, lon, start_time, end_time)
            else:
                return pd.DataFrame()
        
        df = pd.concat(all_dfs)
        df.sort_index(inplace=True)
        # Интерполяция пропусков внутри дня
        if 'tec_value' in df.columns:
            df['tec_value'] = df['tec_value'].interpolate(method='time')
        return df

    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3) -> dict:
        event_time = event_time.replace(tzinfo=timezone.utc) if event_time.tzinfo is None else event_time
        
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        # 1. Target (Событие) - можно использовать синтетику, если данных нет (для теста)
        target_df = self.fetch_tec_for_point(lat, lon, start, end, use_synthetic_if_missing=True)
        
        if target_df.empty:
            return {
                'tec': pd.DataFrame(), 'background': pd.DataFrame(),
                'is_anomaly': False, 'z_score_max': 0.0, 'z_score_min': 0.0,
                'mean_bg': 0.0, 'std_bg': 0.0, 'details': 'Нет данных Target'
            }
        
        # 2. Background (Фон) - СТРОГО реальные данные. Синтетику НЕ используем!
        bg_start = start.replace(year=start.year - 1)
        bg_end = end.replace(year=end.year - 1)
        
        bg_df = self.fetch_tec_for_point(lat, lon, bg_start, bg_end, use_synthetic_if_missing=False)
        
        if bg_df.empty:
            # Если фона нет (например, нет файлов за прошлый год), мы не можем рассчитать аномалию честно.
            # Возвращаем ошибку анализа.
            logger.error(f"❌ Не удалось получить данные фона (прошлый год) для расчета аномалии.")
            return {
                'tec': target_df, 'background': pd.DataFrame(),
                'is_anomaly': False, 'z_score_max': 0.0, 'z_score_min': 0.0,
                'mean_bg': 0.0, 'std_bg': 0.0,
                'details': 'Недостаточно данных фона (нужен прошлый год)'
            }

        # Расчет статистики
        mean_bg = bg_df['tec_value'].mean()
        std_bg = bg_df['tec_value'].std()
        
        if std_bg <= 0.1: # Защита от деления на ноль или слишком малого разброса
            details = "Слишком малый разброс фона, расчет Z-score невозможен"
            logger.warning(details)
            return {
                'tec': target_df, 'background': bg_df,
                'is_anomaly': False, 'z_score_max': 0.0, 'z_score_min': 0.0,
                'mean_bg': mean_bg, 'std_bg': std_bg, 'details': details
            }

        target_df['z_score'] = (target_df['tec_value'] - mean_bg) / std_bg
        z_max = target_df['z_score'].max()
        z_min = target_df['z_score'].min()
        
        anomaly_detected = z_max > 2.0 or z_min < -2.0
        
        if anomaly_detected:
            details = f"Аномалия: max={z_max:.2f}σ, min={z_min:.2f}σ"
            logger.critical(f"   🔴🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ: {details}")
        else:
            details = f"В норме (max={z_max:.2f}σ)"
            logger.info(f"   ✅ Ионосфера: {details}")
        
        return {
            'tec': target_df,
            'background': bg_df,
            'is_anomaly': anomaly_detected,
            'z_score_max': z_max,
            'z_score_min': z_min,
            'mean_bg': mean_bg,
            'std_bg': std_bg,
            'details': details
        }
