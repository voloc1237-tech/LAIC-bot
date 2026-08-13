#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_cache_data.py — быстрый сбор данных за период (2024).
"""

import os
import sys
import json
import pickle
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import requests

from data_collector import USGSCollector
from weather_collector import WeatherCollector
from space_weather import SpaceWeatherCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('CacheCollector')

# ===== НАСТРОЙКА ПЕРИОДА (2024) =====
START_DATE = datetime(2024, 8, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, tzinfo=timezone.utc)
MIN_MAGNITUDE = 5.0
# =====================================


# ----- Климатологические модели -----

def get_climatology_lst(lat, lon, event_time):
    """Возвращает климатологическое значение LST в °C."""
    lat_abs = abs(lat)
    base_temp = 30 - lat_abs * 0.3
    day_of_year = event_time.timetuple().tm_yday
    seasonal = 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    return base_temp + seasonal + 2


def get_climatology_tec(lat, lon, event_time):
    """Возвращает климатологическое значение TEC в TECU."""
    lat_abs = abs(lat)
    base_tec = 25 - lat_abs * 0.2
    hour = event_time.hour
    daily = 10 * np.sin((hour - 8) * np.pi / 12)
    day_of_year = event_time.timetuple().tm_yday
    seasonal = 5 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
    tec = base_tec + daily + seasonal
    return max(tec, 3)


# ----- Загрузка глобальных индексов (исправлено) -----

def load_kp_for_period(start_date, end_date):
    """Загружает Kp из GFZ и фильтрует по датам."""
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        records = []
        for line in resp.text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                y = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                dt_date = datetime(y, month, day, tzinfo=timezone.utc)
                if dt_date < start_date or dt_date > end_date:
                    continue
                for i in range(8):
                    kp_val = float(parts[5+i]) if parts[5+i] else None
                    if kp_val is not None:
                        hour = i*3
                        dt = datetime(y, month, day, hour, 0, tzinfo=timezone.utc)
                        if start_date <= dt <= end_date:
                            records.append({'time': dt, 'kp': kp_val})
            except:
                continue
        if records:
            df = pd.DataFrame(records)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"✅ Kp: загружено {len(df)} записей")
            return df
    except Exception as e:
        logger.error(f"Ошибка загрузки Kp: {e}")
    return pd.DataFrame()


def load_dst_for_period(start_date, end_date):
    """Загружает Dst из Kyoto WDC (если доступен)."""
    all_records = []
    current = start_date.replace(day=1)
    while current <= end_date:
        year = current.year
        month = current.month
        month_str = f"{month:02d}"
        for url_template in [
            f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month_str}/{year}{month_str}.dst",
            f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}/{year}{month_str}.dst"
        ]:
            try:
                resp = requests.get(url_template, timeout=30)
                if resp.status_code != 200:
                    continue
                lines = resp.text.splitlines()
                for line in lines:
                    if not line.strip() or line.startswith('#'):
                        continue
                    try:
                        year_str = line[2:6].strip()
                        day_of_year = int(line[7:10].strip())
                        values = []
                        for i in range(24):
                            pos = 20 + i*4
                            val_str = line[pos:pos+4].strip()
                            if val_str and val_str != '9999':
                                values.append(float(val_str))
                        if values:
                            dst_mean = np.mean(values)
                            dt = datetime(int(year_str), 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year-1)
                            if start_date <= dt <= end_date:
                                all_records.append({'time': dt, 'dst': dst_mean})
                    except:
                        continue
                break
            except:
                continue
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"✅ Dst: загружено {len(df)} записей")
        return df

    logger.warning("⚠️ Dst не найден. Использую среднее -20 нТл.")
    return None


def load_f107_for_period(start_date, end_date):
    """Загружает F10.7 из GFZ и фильтрует по датам."""
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        records = []
        for line in resp.text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                y = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                dt_date = datetime(y, month, day, tzinfo=timezone.utc)
                if dt_date < start_date or dt_date > end_date:
                    continue
                f107_val = float(parts[12]) if parts[12] else None
                if f107_val is not None and f107_val > 0:
                    records.append({'time': dt_date, 'f107': f107_val})
            except:
                continue
        if records:
            df = pd.DataFrame(records)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"✅ F10.7: загружено {len(df)} записей")
            return df
    except Exception as e:
        logger.error(f"Ошибка загрузки F10.7: {e}")
    return pd.DataFrame()


# ----- Основная функция сбора -----

def get_event_features(event_time, lat, lon, kp_df, dst_df, f107_df):
    if kp_df.empty or f107_df.empty:
        return None

    event_ts = pd.Timestamp(event_time).tz_convert('UTC')

    idx = kp_df.index.get_indexer([event_ts], method='nearest')[0]
    if idx == -1:
        return None
    kp_mean = float(kp_df.iloc[idx]['kp'])

    if dst_df is not None and not dst_df.empty:
        idx = dst_df.index.get_indexer([event_ts], method='nearest')[0]
        dst_mean = float(dst_df.iloc[idx]['dst']) if idx != -1 else -20.0
    else:
        dst_mean = -20.0

    date_mask = f107_df.index.date == event_ts.date()
    if not date_mask.any():
        return None
    f107_mean = float(f107_df[date_mask]['f107'].mean())

    weather = WeatherCollector()
    try:
        wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
        if wdf.empty:
            return None
        temp_mean = float(wdf['temp_2m_mean'].mean())
        humidity_mean = float(wdf['humidity_2m_mean'].mean())
    except:
        return None

    lst_celsius = get_climatology_lst(lat, lon, event_time)
    tec_mean = get_climatology_tec(lat, lon, event_time)

    return {
        'lst_celsius': lst_celsius,
        'temp_mean': temp_mean,
        'humidity_mean': humidity_mean,
        'tec_mean': tec_mean,
        'kp_mean': kp_mean,
        'dst_mean': dst_mean,
        'f107_mean': f107_mean,
    }


def collect_and_save():
    start_date = START_DATE
    end_date = END_DATE

    logger.info(f"📥 Загрузка глобальных индексов за период {start_date.date()} — {end_date.date()}...")
    kp_df = load_kp_for_period(start_date, end_date)
    dst_df = load_dst_for_period(start_date, end_date)
    f107_df = load_f107_for_period(start_date, end_date)

    if kp_df.empty or f107_df.empty:
        logger.error("❌ Не удалось загрузить Kp или F10.7. Прерывание.")
        return

    logger.info(f"📥 Загрузка USGS за период...")
    collector = USGSCollector(cache_enabled=True)
    df_events = collector.fetch_global(start_date, end_date, min_magnitude=MIN_MAGNITUDE)

    if df_events.empty:
        logger.error("Нет данных USGS за указанный период")
        return

    logger.info(f"✅ Загружено {len(df_events)} событий")

    all_data = []
    total = len(df_events)
    start_time = time.time()
    skipped = 0

    for i, (_, row) in enumerate(df_events.iterrows()):
        if i % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(f"Обработка {i+1}/{total} (прошло {elapsed:.1f} мин), пропущено {skipped}")

        lat, lon = row['latitude'], row['longitude']
        event_time = row['time']

        features = get_event_features(event_time, lat, lon, kp_df, dst_df, f107_df)
        if features is None:
            skipped += 1
            continue

        record = {
            'id': row['id'],
            'time': event_time.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'magnitude': row['magnitude'],
            'depth_km': row['depth_km'],
            **features
        }
        all_data.append(record)

    logger.info(f"✅ Собрано {len(all_data)} событий (пропущено {skipped})")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"cache_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Сохранено в {json_path}")

    pkl_path = output_dir / f"cache_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_data, f)
    logger.info(f"✅ Сохранено в {pkl_path}")

    size_mb = json_path.stat().st_size / (1024 * 1024)
    logger.info(f"Размер JSON: {size_mb:.2f} МБ")


if __name__ == "__main__":
    collect_and_save()
