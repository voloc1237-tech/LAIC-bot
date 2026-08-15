#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_coord_data.py — подготовка данных для LSTM (регрессия координат)
Запуск: python scripts/prepare_coord_data.py
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

# Добавляем корневую папку в путь, чтобы импортировать модули бота
sys.path.insert(0, str(Path(__file__).parent.parent))

from fault_zones import FaultZoneAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def prepare_coord_sequences(df, sequence_length=7, prediction_window=7):
    """
    Подготавливает последовательности для LSTM (регрессия координат).
    """
    if len(df) < sequence_length + prediction_window:
        return None, None
    
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']
    
    X, y_coords = [], []
    df_sorted = df.sort_values('time').reset_index(drop=True)
    
    for i in range(sequence_length, len(df_sorted) - prediction_window):
        # Проверяем, что все признаки есть
        row = df_sorted.iloc[i]
        if row[feature_names].isnull().any():
            continue
        
        X.append(df_sorted.iloc[i-sequence_length:i][feature_names].values)
        future_event = df_sorted.iloc[i + prediction_window]
        y_coords.append([future_event['latitude'], future_event['longitude']])
    
    if not X:
        return None, None
    
    return np.array(X), np.array(y_coords)


def main():
    logger.info("=" * 60)
    logger.info("📊 ПОДГОТОВКА ДАННЫХ ДЛЯ LSTM (РЕГРЕССИЯ КООРДИНАТ)")
    logger.info("=" * 60)
    
    # 1. Загружаем библиотеку
    df_all = load_library(years=[2022, 2023, 2024])
    if df_all is None:
        logger.error("❌ Нет данных для обучения")
        return
    
    # 2. Подготавливаем последовательности
    logger.info("🔄 Подготовка последовательностей...")
    X, y_coords = prepare_coord_sequences(df_all, sequence_length=7, prediction_window=7)
    
    if X is None:
        logger.error("❌ Недостаточно данных для последовательностей")
        return
    
    logger.info(f"✅ Подготовлено {len(X)} последовательностей")
    logger.info(f"   X.shape: {X.shape}")
    logger.info(f"   y.shape: {y_coords.shape}")
    logger.info(f"   Пример координат: lat={y_coords[0][0]:.2f}, lon={y_coords[0][1]:.2f}")
    
    # 3. Сохраняем
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    np.save(data_dir / "X_coords.npy", X)
    np.save(data_dir / "y_coords.npy", y_coords)
    
    logger.info(f"✅ Сохранено в {data_dir}")
    logger.info("   - X_coords.npy")
    logger.info("   - y_coords.npy")


if __name__ == "__main__":
    main()
