#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_coord_lstm.py — обучение LSTM для прогноза координат
Запуск: python scripts/train_coord_lstm.py
"""

import os
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние между двумя точками в км."""
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_model(input_shape):
    """Строит LSTM-модель для регрессии координат."""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2)  # широта, долгота
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    logger.info("=" * 60)
    logger.info("🧠 ОБУЧЕНИЕ LSTM ДЛЯ ПРОГНОЗА КООРДИНАТ")
    logger.info("=" * 60)
    
    # 1. Загружаем данные
    data_dir = Path(__file__).parent.parent / "data"
    
    X_path = data_dir / "X_coords.npy"
    y_path = data_dir / "y_coords.npy"
    
    if not X_path.exists() or not y_path.exists():
        logger.error("❌ Данные не найдены. Сначала запустите prepare_coord_data.py")
        return
    
    X = np.load(X_path)
    y_coords = np.load(y_path)
    
    logger.info(f"📊 Данные: X={X.shape}, y={y_coords.shape}")
    
    # 2. Разделяем
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_coords, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    logger.info(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 3. Строим и обучаем модель
    model = build_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    logger.info("⚡ Обучение модели...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 4. Оцениваем
    y_pred = model.predict(X_test)
    
    distances = []
    for i in range(len(y_test)):
        dist = haversine_distance(y_test[i][0], y_test[i][1], y_pred[i][0], y_pred[i][1])
        distances.append(dist)
    
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    p75_distance = np.percentile(distances, 75)
    
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  MAE (градусы):  {mean_absolute_error(y_test, y_pred):.3f}")
    logger.info(f"  RMSE (градусы): {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    logger.info(f"  Среднее расстояние: {avg_distance:.1f} км")
    logger.info(f"  Медианное расстояние: {median_distance:.1f} км")
    logger.info(f"  75-й перцентиль: {p75_distance:.1f} км")
    logger.info("=" * 60)
    
    # 5. Сохраняем модель
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "coord_lstm_model.keras"
    model.save(model_path)
    
    # Сохраняем метрики
    metrics = {
        'avg_distance_km': float(avg_distance),
        'median_distance_km': float(median_distance),
        'p75_distance_km': float(p75_distance),
        'mae_degrees': float(mean_absolute_error(y_test, y_pred)),
        'rmse_degrees': float(np.sqrt(mean_squared_error(y_test, y_pred)))
    }
    with open(models_dir / "coord_metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    
    logger.info(f"✅ Модель сохранена в {model_path}")
    logger.info(f"✅ Метрики сохранены в {models_dir / 'coord_metrics.pkl'}")


if __name__ == "__main__":
    main()
