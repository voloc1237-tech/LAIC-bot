#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_coord_lstm.py — обучение LSTM для прогноза координат
"""

import os
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    logger.info("=" * 60)
    logger.info("🧠 ОБУЧЕНИЕ LSTM ДЛЯ ПРОГНОЗА КООРДИНАТ")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    X = np.load(data_dir / "X_coords.npy")
    y_coords = np.load(data_dir / "y_coords.npy")

    logger.info(f"📊 Данные: X={X.shape}, y={y_coords.shape}")

    # ===== МАСШТАБИРОВАНИЕ =====
    n_samples, n_steps, n_features = X.shape
    
    scaler_X = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_coords)

    # Сохраняем скейлеры
    with open(models_dir / "scaler_X.pkl", 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(models_dir / "scaler_y.pkl", 'wb') as f:
        pickle.dump(scaler_y, f)
    logger.info("✅ Скейлеры сохранены")

    # Разделяем
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Обучаем
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

    # Оценка
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)

    distances = []
    for i in range(len(y_test_orig)):
        dist = haversine_distance(
            y_test_orig[i][0], y_test_orig[i][1],
            y_pred[i][0], y_pred[i][1]
        )
        distances.append(dist)

    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    p75_distance = np.percentile(distances, 75)

    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Среднее расстояние: {avg_distance:.1f} км")
    logger.info(f"  Медианное расстояние: {median_distance:.1f} км")
    logger.info(f"  75-й перцентиль: {p75_distance:.1f} км")
    logger.info("=" * 60)

    # Сохраняем модель и метрики
    model.save(models_dir / "coord_lstm_model.keras")
    
    metrics = {
        'avg_distance_km': float(avg_distance),
        'median_distance_km': float(median_distance),
        'p75_distance_km': float(p75_distance)
    }
    with open(models_dir / "coord_metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)

    logger.info(f"✅ Модель сохранена в {models_dir / 'coord_lstm_model.keras'}")
    logger.info(f"✅ Метрики сохранены в {models_dir / 'coord_metrics.pkl'}")


if __name__ == "__main__":
    main()
