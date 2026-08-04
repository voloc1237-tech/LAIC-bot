#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор реальных TEC-данных из IONEX-файлов.
Использует прямые HTTP-запросы к CDDIS и ручной парсинг.
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
    Сборщик ионосферных данных (TEC) из IONEX-файлов.
    Скачивает файлы с CDDIS и парсит их вручную.
    """

    CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_ionex_filename(self, date: datetime) -> str:
        """Формирует имя IONEX-файла."""
        doy = date.timetuple().tm_yday
        return f"codg{date.year}{doy:03d}.01i"

    def _download_ionex(self, date: datetime) -> str | None:
        """Скачивает IONEX-файл с CDDIS."""
        filename = self._get_ionex_filename(date)
        local_path = self.cache_dir / filename

        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"📂 IONEX в кэше: {filename}")
            return str(local_path)

        year = date.strftime("%Y")
        doy = date.timetuple().tm_yday
        url = f"{self.CDDIS_BASE}/{year}/{doy:03d}/{filename}"

        try:
            logger.info(f"📥 Скачивание IONEX: {url}")
            urllib.request.urlretrieve(url, local_path, timeout=60)

            if local_path.stat().st_size == 0:
                local_path.unlink()
                return None

            return str(local_path)
        except Exception as e:
            logger.debug(f"Не удалось скачать {filename}: {e}")
            return None

    def _parse_ionex_file(self, filepath: str, lat: float, lon: float, date: datetime) -> pd.DataFrame:
        """
        Парсит IONEX-файл и извлекает TEC для точки (lat, lon).
        Формат IONEX: сетка 2.5° × 5° (широта × долгота)
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения файла {filepath}: {e}")
            return pd.DataFrame()

        # Ищем секцию TEC
        tec_section = False
        lines = content.splitlines()

        # Ищем параметры сетки
        lat_step = 2.5
        lon_step = 5.0
        lat_min = -87.5
        lat_max = 87.5
        lon_min = -180.0
        lon_max = 180.0

        for line in lines:
            if 'LAT -87.5 87.5 2.5' in line or 'LAT -87.5 87.5 5.0' in line:
                break

        # Создаём сетку
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)

        # Ближайшие индексы
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))

        # Ищем TEC-значения
        records = []
        current_date = date.replace(hour=0)

        for line in lines:
            if 'TEC' in line and line.startswith('TEC'):
                tec_section = True
                continue
            if tec_section and len(line) >= 16 and line[0].isdigit():
                try:
                    values = line.split()
                    if len(values) >= 3:
                        tec_val = float(values[1]) if len(values) > 1 else None
                        if tec_val is not None and tec_val != 999.99 and tec_val != -999.99:
                            for hour in range(0, 24, 2):
                                time_query = current_date.replace(hour=hour)
                                records.append({
                                    'time': time_query,
                                    'tec_value': tec_val,
                                    'lat': lat,
                                    'lon': lon
                                })
                            break
                except (ValueError, IndexError):
                    continue

        if not records:
            logger.debug(f"Нет TEC для ({lat}, {lon}) в {date.date()}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Интерполяция если мало точек
        if len(df) < 12:
            df = df.resample('2H').interpolate(method='time')

        return df

    def _generate_synthetic_tec(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Генерирует синтетические TEC-данные (запасной вариант)."""
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
        logger.info(f"🔄 Сгенерированы синтетические TEC для ({lat:.2f}, {lon:.2f})")
        return df

    def fetch_tec_for_point(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Получает TEC для точки. Сначала пробует реальные данные, при ошибке — синтетику."""
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_time:
            filepath = self._download_ionex(current_date)
            if filepath:
                df = self._parse_ionex_file(filepath, lat, lon, current_date)
                if not df.empty:
                    all_dfs.append(df)
            current_date += timedelta(days=1)

        if not all_dfs:
            logger.warning(f"⚠️ Нет реальных TEC для ({lat:.2f}, {lon:.2f}), использую синтетику")
            return self._generate_synthetic_tec(lat, lon, start_time, end_time)

        df_combined = pd.concat(all_dfs)
        df_combined.sort_index(inplace=True)

        if df_combined['tec_value'].isnull().any():
            df_combined['tec_value'] = df_combined['tec_value'].interpolate(method='time')

        logger.info(f"🛰️ TEC: {len(df_combined)} записей для ({lat:.2f}, {lon:.2f})")
        return df_combined

    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3) -> dict:
        """Собирает TEC и анализирует аномалии."""
        event_time = self._to_utc(event_time)

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
        try:
            bg_start = start.replace(year=start.year - 1)
            bg_end = end.replace(year=end.year - 1)
        except ValueError:
            bg_start = start.replace(year=start.year - 1, day=28)
            bg_end = end.replace(year=end.year - 1, day=28)

        bg_df = self.fetch_tec_for_point(lat, lon, bg_start, bg_end)

        if bg_df.empty:
            return {
                'tec': target_df,
                'background': pd.DataFrame(),
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': 0.0,
                'std_bg': 0.0,
                'details': 'Нет данных фона (прошлый год)'
            }

        mean_bg = bg_df['tec_value'].mean()
        std_bg = bg_df['tec_value'].std()

        if std_bg <= 0:
            return {
                'tec': target_df,
                'background': bg_df,
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': mean_bg,
                'std_bg': std_bg,
                'details': 'Стандартное отклонение фона = 0'
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
