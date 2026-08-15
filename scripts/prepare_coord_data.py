#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_zone_data.py — подготовка данных для классификации по регионам
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fault_zones import FaultZoneAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_library(years=[2022, 2023, 2024]):
    all_data = []
    data_dir = Path(__file__).parent.parent / "data"
    for year in years:
        path = data_dir / f"cache_{year}08_{year}12.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
            all_data.append(df)
            logger.info(f"✅ {year}: {len(df)} событий")
    if not all_data:
        return None
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.drop_duplicates(subset=['id']).sort_values('time')
    logger.info(f"📊 Итого: {len(merged)} событий")
    return merged


def main():
    logger.info("=" * 60)
    logger.info("📊 ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ ПО РЕГИОНАМ")
    logger.info("=" * 60)

    df_all = load_library()
    if df_all is None:
        return

    # Загружаем карту разломов
    zone_analyzer = FaultZoneAnalyzer()

    # Для каждого события определяем регион
    regions = []
    for _, row in df_all.iterrows():
        zone_id, _ = zone_analyzer.get_nearest_zone(row['latitude'], row['longitude'], max_distance_km=150)
        if zone_id:
            # Пытаемся получить регион из GeoJSON
            region = zone_analyzer.get_zone_region(zone_id)
            if not region:
                region = zone_id  # если региона нет, используем ID зоны
        else:
            region = "unknown"
        regions.append(region)

    df_all['region'] = regions

    # Убираем редкие регионы (менее 10 событий)
    region_counts = df_all['region'].value_counts()
    active_regions = region_counts[region_counts >= 10].index.tolist()
    df_filtered = df_all[df_all['region'].isin(active_regions)]

    # Создаём словарь регион → ID
    region_to_id = {r: i for i, r in enumerate(active_regions)}
    df_filtered['region_id'] = df_filtered['region'].map(region_to_id)

    logger.info(f"✅ Активных регионов: {len(active_regions)}")
    logger.info(f"   Событий после фильтрации: {len(df_filtered)}")

    # Сохраняем данные для обучения
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "region_mapping.pkl", 'wb') as f:
        pickle.dump(region_to_id, f)

    df_filtered.to_pickle(data_dir / "events_with_regions.pkl")
    logger.info(f"✅ Сохранено в {data_dir}")


if __name__ == "__main__":
    main()
