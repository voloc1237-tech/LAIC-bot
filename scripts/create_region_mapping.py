#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_region_mapping.py — создаёт словарь регионов из GeoJSON
"""

import pickle
import json
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    region_mapping = {}

    # 1. Пробуем загрузить из events_with_regions.pkl (если есть)
    try:
        df = pd.read_pickle(data_dir / "events_with_regions.pkl")
        region_mapping = df[['region_id', 'region']].drop_duplicates().set_index('region_id').to_dict()['region']
        print(f"✅ Загружено {len(region_mapping)} регионов из events_with_regions.pkl")
    except:
        print("⚠️ events_with_regions.pkl не найден, создаю из GeoJSON...")

    # 2. Если не получилось — читаем из GeoJSON
    if not region_mapping:
        geojson_path = data_dir / "gem_active_faults.geojson"
        if geojson_path.exists():
            with open(geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Собираем все уникальные названия зон
            zone_names = set()
            for feature in data.get('features', []):
                props = feature.get('properties', {})
                name = props.get('name') or props.get('region') or props.get('catalog_id')
                if name:
                    zone_names.add(name)
            
            # Создаём словарь {id: name}
            for i, name in enumerate(sorted(zone_names)):
                region_mapping[i] = name
            
            print(f"✅ Создано {len(region_mapping)} регионов из GeoJSON")
        else:
            print("❌ GeoJSON не найден")

    # 3. Если всё ещё нет — используем fallback
    if not region_mapping:
        region_mapping = {
            0: 'Lianga Fault',
            1: 'zone_15339',
            2: 'zone_740',
            3: 'zone_13489',
            4: 'grid_-20_-175',
            5: 'zone_12917',
            6: 'Middle America Trench megathrust',
            7: 'zone_12288',
            8: 'Longitudinal Valley fault',
            9: 'zone_13946',
            10: 'zone_15347',
            11: 'Miyakojima',
            12: 'zone_13958',
            13: 'zone_13345',
            14: 'zone_11445',
            15: 'Mapastapec Fault',
            16: 'zone_15345',
            17: 'GARRAPATAS_FAULT',
            18: 'zone_15321',
        }
        print(f"⚠️ Использую fallback словарь ({len(region_mapping)} регионов)")

    # Сохраняем
    with open(data_dir / "region_mapping.pkl", 'wb') as f:
        pickle.dump(region_mapping, f)

    print(f"✅ region_mapping.pkl сохранён ({len(region_mapping)} регионов)")

if __name__ == "__main__":
    main()
