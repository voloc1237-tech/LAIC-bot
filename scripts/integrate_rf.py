#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_rf.py — проверка модели RandomForest и пример использования
"""

import os
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    models_dir = Path(__file__).parent.parent / "models"
    
    # 1. Загружаем модель и скейлер
    with open(models_dir / "zone_rf_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(models_dir / "scaler_rf.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(models_dir / "feature_importance.pkl", 'rb') as f:
        feature_importance = pickle.load(f)
    
    # 2. Загружаем словарь регионов
    import pandas as pd
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_pickle(data_dir / "events_with_regions.pkl")
    region_mapping = df[['region_id', 'region']].drop_duplicates().set_index('region_id').to_dict()['region']
    
    logger.info("✅ Модель и данные загружены")
    logger.info(f"   Всего регионов: {len(region_mapping)}")
    logger.info(f"   Важность признаков: {feature_importance}")
    
    # 3. Пример прогноза
    # Предположим, у нас есть новые данные (магнитуда, глубина, LST, температура, влажность, TEC, Kp, Dst, F10.7)
    sample_features = np.array([[5.2, 10.0, 25.0, 22.0, 60.0, 20.0, 2.0, -10.0, 150.0]])
    sample_scaled = scaler.transform(sample_features)
    pred_region_id = model.predict(sample_scaled)[0]
    pred_region = region_mapping.get(pred_region_id, "Неизвестный регион")
    
    logger.info("🔮 Пример прогноза:")
    logger.info(f"   Признаки: {sample_features[0]}")
    logger.info(f"   Предсказанный регион: {pred_region} (ID: {pred_region_id})")

if __name__ == "__main__":
    main()
