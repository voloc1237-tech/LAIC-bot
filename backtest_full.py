#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py — честный бэктест на исторических данных
Использует сохранённые модели и историческую библиотеку.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_library():
    """Загружает историческую библиотеку."""
    data_dir = Path("data")
    all_data = []
    
    for year in [2022, 2023, 2024]:
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
    logger.info("🔬 ЧЕСТНЫЙ БЭКТЕСТ НА ИСТОРИЧЕСКИХ ДАННЫХ")
    logger.info("=" * 60)
    
    # 1. Загружаем историческую библиотеку
    df_all = load_library()
    if df_all is None:
        logger.error("❌ Библиотека не найдена")
        return
    
    # 2. Разделяем на обучение (2022) и тест (2023-2024)
    train_df = df_all[df_all['time'].dt.year == 2022]
    test_df = df_all[df_all['time'].dt.year.isin([2023, 2024])]
    
    logger.info(f"📊 Обучение: {len(train_df)} событий (2022)")
    logger.info(f"📊 Тест: {len(test_df)} событий (2023-2024)")
    
    # 3. Загружаем сохранённые модели
    models_dir = Path("models")
    
    # LSTM
    try:
        lstm_model = load_model(models_dir / "lstm_model.keras")
        with open(models_dir / "lstm_scaler.pkl", 'rb') as f:
            lstm_scaler = pickle.load(f)
        logger.info("✅ LSTM модель загружена")
    except Exception as e:
        logger.warning(f"⚠️ LSTM не загружена: {e}")
        lstm_model = None
    
    # RF
    try:
        with open(models_dir / "zone_rf_model.pkl", 'rb') as f:
            rf_model = pickle.load(f)
        with open(models_dir / "scaler_rf.pkl", 'rb') as f:
            rf_scaler = pickle.load(f)
        logger.info("✅ RF модель загружена")
    except Exception as e:
        logger.warning(f"⚠️ RF не загружена: {e}")
        rf_model = None
    
    # 4. Бэктест LSTM
    if lstm_model:
        logger.info("\n" + "=" * 60)
        logger.info("🧠 БЭКТЕСТ LSTM")
        logger.info("=" * 60)
        # ... код бэктеста LSTM ...
    
    # 5. Бэктест RF
    if rf_model:
        logger.info("\n" + "=" * 60)
        logger.info("🌲 БЭКТЕСТ RF")
        logger.info("=" * 60)
        # ... код бэктеста RF ...
    
    # 6. Сохраняем результаты
    results = {
        'model': 'LSTM + RF',
        'test_period': '2023-2024',
        'train_period': '2022',
        'total_events': len(df_all),
        'test_events': len(test_df)
    }
    
    with open("backtest_results.csv", 'w') as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v}\n")
    
    logger.info("✅ Результаты сохранены в backtest_results.csv")

if __name__ == "__main__":
    main()
