#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_cache_data.py — сбор всех данных за 2024 год в один JSON-файл
Запускать через GitHub Actions (один раз).
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
import yaml

# Импорты модулей бота (предполагается, что они лежат в той же папке)
from data_collector import USGSCollector
from weather_collector import WeatherCollector
from ionosphere_collector import IonosphereCollector
from space_weather import SpaceWeatherCollector
from ee_collector import get_lst_data, GEEInitializer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('CacheCollector')

def collect_and_save():
    # 1. Загружаем настройки (если есть)
    settings_path = "config/settings.yaml"
    settings = {}
    if os.path.exists(settings_path):
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

    # 2. Инициализируем сборщики
    weather = WeatherCollector()
    iono = IonosphereCollector(cache_dir="data/ionex_cache")
    space = SpaceWeatherCollector()

    # 3. Загружаем USGS за 2024 год (M≥4.5)
    logger.info("📥 Загрузка USGS за 2024 год...")
    collector = USGSCollector(cache_enabled=True)
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df_events = collector.fetch_global(start_date, end_date, min_magnitude=4.5)

    if df_events.empty:
        logger.error("Нет данных USGS за 2024 год")
        return

    logger.info(f"✅ Загружено {len(df_events)} событий")

    # 4. Инициализируем GEE для LST (ключ должен быть в окружении)
    GEEInitializer.init()

    # 5. Для каждого события собираем признаки
    all_data = []
    total = len(df_events)
    start_time = time.time()
    
    for i, (_, row) in enumerate(df_events.iterrows()):
        if i % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(f"Обработка {i+1}/{total} (прошло {elapsed:.1f} мин)")

        event_id = row['id']
        lat, lon = row['latitude'], row['longitude']
        event_time = row['time']
        mag = row['magnitude']
        depth = row['depth_km']

        # Погода (7 дней до, 3 после)
        try:
            wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
            if not wdf.empty:
                temp_mean = float(wdf['temp_2m_mean'].mean())
                humidity_mean = float(wdf['humidity_2m_mean'].mean())
            else:
                temp_mean = humidity_mean = None
        except Exception as e:
            logger.debug(f"Ошибка погоды для {event_id}: {e}")
            temp_mean = humidity_mean = None

        # Космическая погода (7 дней до, 3 после)
        try:
            start_sp = event_time - timedelta(days=7)
            end_sp = event_time + timedelta(days=3)
            space_data = space.fetch_all_for_period(start_sp, end_sp)
            kp_df = space_data.get('kp', pd.DataFrame())
            dst_df = space_data.get('dst', pd.DataFrame())
            f107_df = space_data.get('f107', pd.DataFrame())
            kp_mean = float(kp_df['kp'].mean()) if not kp_df.empty else None
            dst_mean = float(dst_df['dst'].mean()) if not dst_df.empty else None
            f107_mean = float(f107_df['f107'].mean()) if not f107_df.empty else None
        except Exception as e:
            logger.debug(f"Ошибка космоса для {event_id}: {e}")
            kp_mean = dst_mean = f107_mean = None

        # LST (за 7 дней до события)
        try:
            start_lst = (event_time - timedelta(days=7)).strftime('%Y-%m-%d')
            end_lst = event_time.strftime('%Y-%m-%d')
            lst_data = get_lst_data(lat, lon, start_lst, end_lst)
            lst_celsius = lst_data['lst_celsius'] if lst_data else None
        except Exception as e:
            logger.debug(f"Ошибка LST для {event_id}: {e}")
            lst_celsius = None

        # TEC (ионосфера) за 7 дней до и 3 после
        try:
            tec_df = iono.fetch_tec_for_point(lat, lon, 
                                              event_time - timedelta(days=7), 
                                              event_time + timedelta(days=3))
            tec_mean = float(tec_df['tec_value'].mean()) if not tec_df.empty else None
        except Exception as e:
            logger.debug(f"Ошибка TEC для {event_id}: {e}")
            tec_mean = None

        # Формируем запись
        record = {
            'id': event_id,
            'time': event_time.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'magnitude': mag,
            'depth_km': depth,
            'lst_celsius': lst_celsius,
            'temp_mean': temp_mean,
            'humidity_mean': humidity_mean,
            'tec_mean': tec_mean,
            'kp_mean': kp_mean,
            'dst_mean': dst_mean,
            'f107_mean': f107_mean,
        }
        all_data.append(record)

    # 6. Сохраняем в JSON и pickle
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # JSON
    json_path = output_dir / "cache_2024.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Сохранено {len(all_data)} записей в {json_path}")

    # Pickle
    pkl_path = output_dir / "cache_2024.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_data, f)
    logger.info(f"✅ Сохранено в {pkl_path}")

    # Размер
    size_mb = json_path.stat().st_size / (1024 * 1024)
    logger.info(f"Размер JSON: {size_mb:.2f} МБ")

if __name__ == "__main__":
    collect_and_save()
