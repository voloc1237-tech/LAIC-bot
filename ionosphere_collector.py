#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через IONEX-файлы.
Реализован анализ аномалий с учётом исторических данных (фон за прошлый год).

Зависимости:
    pip install pandas numpy gnssanalysis
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
from gnssanalysis.ionex import IonexMap

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class IonosphereCollector:
    """
    Сборщик ионосферных данных (TEC, ROTI) через IONEX-файлы.
    - Target: данные вокруг события
    - Background: данные за тот же период год назад (для расчёта нормы)
    Использует FTP-зеркало IGS (открытый доступ).
    """

    # FTP-зеркало IGS (открытое, не требует авторизации)
    CDDIS_FTP_BASE = "ftp://igs.ensg.ign.fr/pub/igs/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Приводит datetime к UTC."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _get_ionex_version(date: datetime) -> str:
        """
        Выбирает версию файла IONEX в зависимости от свежести даты.
        - codg: Финальные данные (точность высокая, задержка ~14 дней).
        - c1pg: Прогнозные данные (доступны сразу, точность ниже).
        """
        now = datetime.now(timezone.utc)
        
        # Если дата старше 15 дней, берем финальные (codg)
        if (now - date).days > 15:
            return "codg"
        else:
            # Иначе берем прогнозные (c1pg), чтобы данные были доступны
            return "c1pg"

    def _get_ionex_filename(self, date: datetime, version: str = None) -> str:
        """
        Формирует имя IONEX-файла по дате.
        Формат: codgYYYYDDD.01i (например, codg2023001.01i)
        """
        if version is None:
            version = self._get_ionex_version(date)
        doy = date.timetuple().tm_yday
        return f"{version}{date.year}{doy:03d}.01i"

    def _download_ionex(self, date: datetime, version: str = None) -> str | None:
        """Скачивает IONEX-файл с FTP-зеркала IGS."""
        filename = self._get_ionex_filename(date, version)
        local_path = self.cache_dir / filename

        # Проверка кэша
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"📂 IONEX файл в кэше: {filename}")
            return str(local_path)

        year = date.strftime("%Y")
        doy = date.timetuple().tm_yday
        url = f"{self.CDDIS_FTP_BASE}/{year}/{doy:03d}/{filename}"

        try:
            logger.info(f"📥 Скачивание IONEX: {url}")
            urllib.request.urlretrieve(url, local_path, timeout=60)

            if local_path.stat().st_size == 0:
                logger.warning(f"⚠️ Скачанный файл пуст: {local_path}. Удаляю.")
                local_path.unlink()
                return None

            logger.info(f"✅ IONEX сохранён: {local_path}")
            return str(local_path)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"⚠️ IONEX файл не найден (404): {filename}")
            else:
                logger.warning(f"⚠️ Ошибка HTTP при скачивании: {e}")
            return None
        except urllib.error.URLError as e:
            logger.warning(f"⚠️ Ошибка сети при скачивании: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Неожиданная ошибка скачивания: {e}")
            if local_path.exists():
                local_path.unlink()
            return None

    def _parse_ionex_for_point(self, filepath: str, lat: float, lon: float, date: datetime) -> pd.DataFrame:
        """
        Парсит IONEX-файл и извлекает TEC для указанной точки.
        Возвращает DataFrame с колонками: ['time', 'tec_value']
        """
        try:
            ionmap = IonexMap.from_file(filepath)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка парсинга IONEX {filepath}: {e}")
            return pd.DataFrame()

        records = []
        # IONEX обычно содержит карты каждые 2 часа (0, 2, ..., 22 UTC)
        for hour in range(0, 24, 2):
            time_query = date.replace(hour=hour, minute=0, second=0, microsecond=0)
            try:
                tec_val = ionmap.get_tec(lat=lat, lon=lon, time=time_query)
                if tec_val is not None and not np.isnan(tec_val):
                    records.append({
                        "time": time_query,
                        "tec_value": float(tec_val),
                        "lat": lat,
                        "lon": lon
                    })
            except Exception:
                # Это нормально, если точка вне покрытия карты
                continue

        if not records:
            logger.debug(f"Нет TEC для ({lat}, {lon}) на {date.date()}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def fetch_tec_for_point(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получить VTEC (вертикальный TEC) для указанной точки за период.
        Возвращает DataFrame с индексом времени и колонкой 'tec_value'.
        """
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_time:
            version = self._get_ionex_version(current_date)
            filepath = self._download_ionex(current_date, version)
            
            if filepath:
                df = self._parse_ionex_for_point(filepath, lat, lon, current_date)
                if not df.empty:
                    all_dfs.append(df)

            current_date += timedelta(days=1)

        if not all_dfs:
            logger.warning(f"⚠️ Нет TEC данных для ({lat:.2f}, {lon:.2f}) за период")
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs)
        df_combined.sort_index(inplace=True)

        # Интерполяция пропущенных значений
        if df_combined["tec_value"].isnull().any():
            df_combined["tec_value"] = df_combined["tec_value"].interpolate(method="time")

        logger.info(f"🛰️ TEC: {len(df_combined)} записей для ({lat:.2f}, {lon:.2f})")
        return df_combined

    def fetch_roti(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получить ROTI (Rate of TEC Index) — показатель ионосферных возмущений.
        ROTI рассчитывается как скользящее стандартное отклонение TEC за окно 8 часов.
        """
        tec_df = self.fetch_tec_for_point(lat, lon, start_time, end_time)
        if tec_df.empty:
            return pd.DataFrame()

        # Приводим к регулярной сетке 2 часа
        tec_df = tec_df.asfreq("2H")

        # ROTI = std(TEC) за окно 8 часов (4 точки по 2 часа)
        tec_df["roti"] = tec_df["tec_value"].rolling("8H", min_periods=2).std()

        logger.info(f"🛰️ ROTI: {len(tec_df)} записей")
        return tec_df[["roti"]]

    def fetch_for_event(self, event_time: datetime, lat: float, lon: float, 
                        days_before: int = 7, days_after: int = 3) -> dict:
        """
        Собирает TEC и ROTI за период вокруг события (Target) и за прошлый год (Background).
        Возвращает словарь:
        {
            'tec': df_target,
            'background': df_background,
            'roti': df_roti,
            'is_anomaly': bool,
            'z_score_max': float,
            'z_score_min': float,
            'mean_bg': float,
            'std_bg': float,
            'details': str
        }
        """
        event_time = self._to_utc(event_time)

        # 1. Сбор данных ВОКРУГ СОБЫТИЯ (Target)
        start_target = event_time - timedelta(days=days_before)
        end_target = event_time + timedelta(days=days_after)
        target_df = self.fetch_tec_for_point(lat, lon, start_target, end_target)

        if target_df.empty:
            logger.warning(f"⚠️ Нет Target данных для ({lat:.2f}, {lon:.2f})")
            return {
                'tec': pd.DataFrame(),
                'background': pd.DataFrame(),
                'roti': pd.DataFrame(),
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': 0.0,
                'std_bg': 0.0,
                'details': "Нет данных"
            }

        # 2. Сбор данных ЗА ПРОШЛЫЙ ГОД (Background)
        bg_year = event_time.year - 1
        # Обработка 29 февраля
        try:
            bg_date = event_time.replace(year=bg_year)
        except ValueError:
            bg_date = event_time.replace(year=bg_year, day=28)

        start_bg = bg_date - timedelta(days=days_before)
        end_bg = bg_date + timedelta(days=days_after)
        
        # Для фона всегда можно брать финальные данные (codg)
        # Устанавливаем версию явно, чтобы не зависеть от логики _get_ionex_version
        bg_df = self._fetch_tec_with_version(lat, lon, start_bg, end_bg, version="codg")

        # 3. Расчет статистики фона
        if bg_df.empty:
            logger.warning(f"⚠️ Нет Background данных для ({lat:.2f}, {lon:.2f})")
            bg_df = pd.DataFrame()

        mean_bg = bg_df['tec_value'].mean() if not bg_df.empty else 0.0
        std_bg = bg_df['tec_value'].std() if not bg_df.empty else 0.0

        # 4. Расчет Z-Score для данных события (Target)
        anomaly_detected = False
        z_max = 0.0
        z_min = 0.0
        details = ""

        if not target_df.empty and not bg_df.empty and std_bg > 0:
            target_df['z_score'] = (target_df['tec_value'] - mean_bg) / std_bg
            z_max = target_df['z_score'].max()
            z_min = target_df['z_score'].min()
            
            ANOMALY_THRESHOLD = 2.0
            if z_max > ANOMALY_THRESHOLD:
                anomaly_detected = True
                details += f"Положительная аномалия (+{z_max:.2f}σ)"
            
            if z_min < -ANOMALY_THRESHOLD:
                anomaly_detected = True
                if details:
                    details += " и "
                details += f"Отрицательная аномалия ({z_min:.2f}σ)"
            
            if not details:
                details = "В пределах нормы"
        else:
            details = "Недостаточно данных для расчёта аномалии"

        # 5. ROTI
        roti_df = pd.DataFrame()
        if not target_df.empty:
            roti_df = self.fetch_roti(lat, lon, start_target, end_target)

        # Логирование
        if anomaly_detected:
            logger.critical(f"   🔴🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ: {details}")
        else:
            logger.info(f"   ✅ Ионосфера: {details}")

        return {
            'tec': target_df,
            'background': bg_df,
            'roti': roti_df,
            'is_anomaly': anomaly_detected,
            'z_score_max': z_max,
            'z_score_min': z_min,
            'mean_bg': mean_bg,
            'std_bg': std_bg,
            'details': details
        }

    def _fetch_tec_with_version(self, lat: float, lon: float, start_time: datetime, end_time: datetime, version: str) -> pd.DataFrame:
        """
        Вспомогательный метод для скачивания TEC с указанной версией файла.
        """
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_time:
            filepath = self._download_ionex(current_date, version)
            if filepath:
                df = self._parse_ionex_for_point(filepath, lat, lon, current_date)
                if not df.empty:
                    all_dfs.append(df)
            current_date += timedelta(days=1)

        if not all_dfs:
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs)
        df_combined.sort_index(inplace=True)
        return df_combined


if __name__ == "__main__":
    # Тест
    collector = IonosphereCollector(cache_dir="data/ionex_cache")
    
    # Координаты (например, Япония)
    lat, lon = 32.68, 130.72
    
    # Время события
    event_time = datetime(2026, 7, 28, 7, 27, 15, tzinfo=timezone.utc)
    
    print(f"🚀 Запуск сбора данных для ({lat}, {lon}) вокруг события {event_time}")
    result = collector.fetch_for_event(event_time, lat, lon)
    
    if not result['tec'].empty:
        print(f"✅ Target: {len(result['tec'])} записей")
        print(f"   TEC среднее: {result['tec']['tec_value'].mean():.2f} TECU")
    else:
        print("❌ Нет данных")
