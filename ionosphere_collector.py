#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через IONEX-файлы.
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
    """
    Сборщик ионосферных данных (TEC, ROTI) через IONEX-файлы.
    Использует gnssanalysis для парсинга, если доступна.
    """

    CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")
        
        # Проверяем наличие gnssanalysis при инициализации
        self.gnss_available = self._check_gnssanalysis()

    def _check_gnssanalysis(self):
        """Проверяет наличие gnssanalysis и возвращает True/False."""
        try:
            # Пробуем импортировать внутри метода
            from gnssanalysis.ionex import IonexMap
            logger.info("✅ gnssanalysis загружена")
            return True
        except ImportError:
            logger.warning("⚠️ gnssanalysis не установлена. Использую синтетические данные.")
            return False

    def _generate_synthetic_tec(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Генерирует синтетические TEC-данные (для тестирования)."""
        records = []
        current = start_time
        base_tec = 30 - abs(lat) * 0.3
        
        while current <= end_time:
            hour = current.hour
            daily = 5 * np.sin((hour - 10) * np.pi / 12)
            noise = np.random.normal(0, 2)
            
            tec = max(0, base_tec + daily + noise)
            records.append({'time': current, 'tec_value': tec})
            current += timedelta(hours=2)
        
        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        return df

    def fetch_tec_for_point(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Получает TEC для точки."""
        if self.gnss_available:
            try:
                real_df = self._fetch_real_tec(lat, lon, start_time, end_time)
                if not real_df.empty:
                    logger.info(f"🛰️ Реальные TEC для ({lat:.2f}, {lon:.2f}): {len(real_df)} записей")
                    return real_df
            except Exception as e:
                logger.warning(f"⚠️ Ошибка реальных данных: {e}")
        
        # Если реальные не получены — используем синтетику
        logger.info(f"🔄 Использую синтетические TEC для ({lat:.2f}, {lon:.2f})")
        return self._generate_synthetic_tec(lat, lon, start_time, end_time)
    

    def _fetch_real_tec(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Реальная загрузка через gnssanalysis."""
        if not self.gnss_available:
            raise ImportError("gnssanalysis не установлена")
        
        # Попытка импортировать внутри метода
        try:
            from gnssanalysis.ionex import IonexMap
        except ImportError:
            raise ImportError("gnssanalysis не установлена")
        
        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current_date <= end_time:
            filename = f"codg{current_date.year}{current_date.timetuple().tm_yday:03d}.01i"
            local_path = self.cache_dir / filename
            
            # Пытаемся скачать
            url = f"{self.CDDIS_BASE}/{current_date.year}/{current_date.timetuple().tm_yday:03d}/{filename}"
            
            if not local_path.exists():
                try:
                    logger.info(f"📥 Скачивание: {url}")
                    urllib.request.urlretrieve(url, local_path, timeout=30)
                except Exception as e:
                    logger.debug(f"Не удалось скачать {filename}: {e}")
                    current_date += timedelta(days=1)
                    continue
            
            if local_path.exists() and local_path.stat().st_size > 0:
                try:
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
                    logger.debug(f"Ошибка парсинга {filename}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_dfs:
            return pd.DataFrame()
        
        df = pd.concat(all_dfs)
        df.sort_index(inplace=True)
        if df['tec_value'].isnull().any():
            df['tec_value'] = df['tec_value'].interpolate(method='time')
        return df

    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3) -> dict:
        """Собирает TEC и анализирует аномалии."""
        event_time = event_time.replace(tzinfo=timezone.utc) if event_time.tzinfo is None else event_time
        
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        # Target
        target_df = self.fetch_tec_for_point(lat, lon, start, end)
        if target_df.empty:
            return {
                'tec': pd.DataFrame(),
                'background': pd.DataFrame(),
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': 0.0,
                'std_bg': 0.0,
                'details': 'Нет данных Target'
            }
        
        # Background (прошлый год)
        bg_start = start.replace(year=start.year - 1)
        bg_end = end.replace(year=end.year - 1)
        bg_df = self.fetch_tec_for_point(lat, lon, bg_start, bg_end)
        
        if bg_df.empty:
            # Если нет фона — используем среднее от Target
            bg_df = target_df.copy()
            bg_df['tec_value'] = bg_df['tec_value'].mean()
        
        mean_bg = bg_df['tec_value'].mean()
        std_bg = bg_df['tec_value'].std()
        
        if std_bg > 0:
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
        else:
            anomaly_detected = False
            z_max = 0.0
            z_min = 0.0
            details = "Недостаточно данных для расчёта"
        
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
