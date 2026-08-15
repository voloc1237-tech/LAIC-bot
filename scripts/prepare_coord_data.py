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

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from fault_zones import FaultZoneAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== ПРОГРЕСС-БАР =====
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("⚠️ tqdm не установлен. Установите: pip install tqdm")
# =========================


def load_library(years=[2022, 2023, 2024]):
    """Загружает объединённую библиотеку данных."""
    all_data = []
    data_dir = Path(__file__).parent.parent / "data"
    
    for year in years:
        # Пробуем разные варианты имён файлов
        possible_paths = [
            data_dir / f"cache_{year}08_{year}12.json",
            data_dir / f"cache_{year}.json",
            data_dir / f"library_{year}.json",
        ]
        loaded = False
        for path in possible_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
                all_data.append(df)
                logger.info(f"✅ {year}: {len(df)} событий из {path.name}")
                loaded = True
                break
        if not loaded:
            logger.warning(f"⚠️ Данные за {year} не найдены")
    
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

    # 1. Загружаем библиотеку
    df_all = load_library(years=[2022, 2023, 2024])
    if df_all is None:
        logger.error("❌ Нет данных")
        return

    # 2. Загружаем карту разломов
    logger.info("🔄 Загрузка карты разломов...")
    zone_analyzer = FaultZoneAnalyzer()
    
    if zone_analyzer.gdf is None:
        logger.error("❌ Карта разломов не загружена")
        return

    # 3. Для каждого события определяем зону
    logger.info("🔄 Привязка событий к зонам...")
    
    regions = []
    total = len(df_all)
    
    # Используем прогресс-бар, если установлен tqdm
    iterator = tqdm(df_all.iterrows(), total=total, desc="Привязка") if HAS_TQDM else df_all.iterrows()
    
    for _, row in iterator:
        lat, lon = row['latitude'], row['longitude']
        
        # Пытаемся найти зону (увеличиваем радиус для ускорения)
        zone_id, dist = zone_analyzer.get_nearest_zone(lat, lon, max_distance_km=200)
        
        if zone_id:
            # Пытаемся получить название региона из GeoJSON
            # Если есть поле 'region' — используем его, иначе используем название разлома
            if zone_analyzer.gdf is not None:
                zone_row = zone_analyzer.gdf[zone_analyzer.gdf['id'] == zone_id]
                if not zone_row.empty:
                    # Пробуем разные поля для названия
                    region = zone_row.iloc[0].get('region')
                    if not region:
                        region = zone_row.iloc[0].get('name')
                    if not region:
                        region = zone_id
                else:
                    region = zone_id
            else:
                region = zone_id
        else:
            # Если зона не найдена — используем сетку
            lat_grid = int(lat / 5) * 5
            lon_grid = int(lon / 5) * 5
            region = f"grid_{lat_grid}_{lon_grid}"
        
        regions.append(region)

    df_all['region'] = regions

    # 4. Подсчитываем статистику
    region_counts = df_all['region'].value_counts()
    total_regions = len(region_counts)
    active_regions = region_counts[region_counts >= 5].index.tolist()
    
    logger.info(f"✅ Всего регионов: {total_regions}")
    logger.info(f"✅ Активных регионов (≥5 событий): {len(active_regions)}")
    
    # Показываем топ-10 регионов
    logger.info("📊 Топ-10 регионов:")
    for region, count in region_counts.head(10).items():
        logger.info(f"   {region}: {count} событий")

    # 5. Фильтруем и создаём словарь
    df_filtered = df_all[df_all['region'].isin(active_regions)]
    region_to_id = {r: i for i, r in enumerate(active_regions)}
    df_filtered['region_id'] = df_filtered['region'].map(region_to_id)

    logger.info(f"✅ Событий после фильтрации: {len(df_filtered)}")

    # 6. Сохраняем
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "region_mapping.pkl", 'wb') as f:
        pickle.dump(region_to_id, f)
    
    df_filtered.to_pickle(data_dir / "events_with_regions.pkl")
    
    logger.info(f"✅ Сохранено в {data_dir}")
    logger.info("   - region_mapping.pkl")
    logger.info("   - events_with_regions.pkl")


if __name__ == "__main__":
    main()
