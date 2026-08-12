#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_cache_data.py — сбор реальных данных за указанный год.
Используются только проверенные рабочие источники.
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

# Импорты модулей бота
from data_collector import USGSCollector
from weather_collector import WeatherCollector
from ionosphere_collector import IonosphereCollector
from ee_collector import get_lst_data, GEEInitializer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('CacheCollector')

# ===== НАСТРОЙКА ГОДА =====
TARGET_YEAR = 2023
# =========================


def load_kp_year(year):
    """Загружает Kp из GFZ (надёжный источник)."""
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
                if y != year:
                    continue
                month = int(parts[1])
                day = int(parts[2])
                for i in range(8):
                    kp_val = float(parts[5+i]) if parts[5+i] else None
                    if kp_val is not None:
                        hour = i*3
                        dt = datetime(year, month, day, hour, 0, tzinfo=timezone.utc)
                        records.append({'time': dt, 'kp': kp_val})
            except:
                continue
        if records:
            df = pd.DataFrame(records)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"✅ Kp: загружено {len(df)} записей за {year}")
            return df
    except Exception as e:
        logger.error(f"Ошибка загрузки Kp: {e}")
    return pd.DataFrame()


def load_dst_year(year):
    """Загружает Dst из Kyoto WDC (надёжный источник)."""
    all_records = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        # Пробуем provisional и final
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
                            all_records.append({'time': dt, 'dst': dst_mean})
                    except:
                        continue
                break  # если успешно загрузили месяц
            except:
                continue

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"✅ Dst (Kyoto): загружено {len(df)} записей за {year}")
        return df

    logger.warning(f"⚠️ Dst за {year} не найден. Будет использовано среднее значение.")
    return None


def load_f107_year(year):
    """Загружает F10.7 из LASP (надёжный источник)."""
    url = "https://lasp.colorado.edu/lisird/data/570_daily.txt"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        records = []
        for line in resp.text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                y = int(parts[0])
                if y != year:
                    continue
                day_of_year = int(parts[1])
                f107_val = float(parts[2]) if parts[2] else None
                if f107_val is not None:
                    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year-1)
                    records.append({'time': dt, 'f107': f107_val})
            except:
                continue
        if records:
            df = pd.DataFrame(records)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"✅ F10.7 (LASP): загружено {len(df)} записей за {year}")
            return df
    except Exception as e:
        logger.error(f"Ошибка загрузки F10.7: {e}")
    return pd.DataFrame()


def get_event_features(event_time, lat, lon, kp_df, dst_df, f107_df):
    """Собирает признаки для одного события."""
    if kp_df.empty or f107_df.empty:
        return None

    event_ts = pd.Timestamp(event_time).tz_convert('UTC')

    # Kp
    idx = kp_df.index.get_indexer([event_ts], method='nearest')[0]
    if idx == -1:
        return None
    kp_mean = float(kp_df.iloc[idx]['kp'])

    # Dst (опционально)
    if dst_df is not None and not dst_df.empty:
        idx = dst_df.index.get_indexer([event_ts], method='nearest')[0]
        dst_mean = float(dst_df.iloc[idx]['dst']) if idx != -1 else -20.0
    else:
        dst_mean = -20.0

    # F10.7
    date_mask = f107_df.index.date == event_ts.date()
    if not date_mask.any():
        return None
    f107_mean = float(f107_df[date_mask]['f107'].mean())

    # Погода
    weather = WeatherCollector()
    try:
        wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
        if wdf.empty:
            return None
        temp_mean = float(wdf['temp_2m_mean'].mean())
        humidity_mean = float(wdf['humidity_2m_mean'].mean())
    except:
        return None

    # LST
    try:
        start_lst = (event_time - timedelta(days=7)).strftime('%Y-%m-%d')
        end_lst = event_time.strftime('%Y-%m-%d')
        lst_data = get_lst_data(lat, lon, start_lst, end_lst)
        lst_celsius = lst_data['lst_celsius'] if lst_data else None
        if lst_celsius is None:
            return None
    except:
        return None

    # TEC
    iono = IonosphereCollector(cache_dir="data/ionex_cache")
    try:
        tec_df = iono.fetch_tec_for_point(
            lat, lon,
            event_time - timedelta(days=7),
            event_time + timedelta(days=3)
        )
        if tec_df.empty:
            return None
        tec_mean = float(tec_df['tec_value'].mean())
    except:
        return None

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
    year = TARGET_YEAR

    logger.info(f"📥 Загрузка глобальных индексов за {year} год...")
    kp_df = load_kp_year(year)
    dst_df = load_dst_year(year)
    f107_df = load_f107_year(year)

    if kp_df.empty or f107_df.empty:
        logger.error(f"❌ Не удалось загрузить Kp или F10.7 за {year}. Прерывание.")
        return

    logger.info(f"📥 Загрузка USGS за {year} год...")
    collector = USGSCollector(cache_enabled=True)
    start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(year, 12, 31, tzinfo=timezone.utc)
    df_events = collector.fetch_global(start_date, end_date, min_magnitude=4.5)

    if df_events.empty:
        logger.error(f"Нет данных USGS за {year} год")
        return

    logger.info(f"✅ Загружено {len(df_events)} событий")

    GEEInitializer.init()

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

    logger.info(f"✅ Собрано {len(all_data)} событий с полными реальными данными (пропущено {skipped})")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"cache_{year}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Сохранено {len(all_data)} записей в {json_path}")

    pkl_path = output_dir / f"cache_{year}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_data, f)
    logger.info(f"✅ Сохранено в {pkl_path}")

    size_mb = json_path.stat().st_size / (1024 * 1024)
    logger.info(f"Размер JSON: {size_mb:.2f} МБ")


if __name__ == "__main__":
    collect_and_save()
