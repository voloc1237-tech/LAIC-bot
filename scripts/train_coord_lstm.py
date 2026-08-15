#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_zone_lstm.py — обучение LSTM для классификации регионов
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_sequences(df, sequence_length=7, prediction_window=7):
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']

    X, y = [], []
    df_sorted = df.sort_values('time').reset_index(drop=True)

    for i in range(sequence_length, len(df_sorted) - prediction_window):
        X.append(df_sorted.iloc[i-sequence_length:i][feature_names].values)
        y.append(df_sorted.iloc[i + prediction_window]['region_id'])

    return np.array(X), np.array(y)


def main():
    logger.info("=" * 60)
    logger.info("🧠 ОБУЧЕНИЕ LSTM ДЛЯ КЛАССИФИКАЦИИ РЕГИОНОВ")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Загружаем данные
    df = pd.read_pickle(data_dir / "events_with_regions.pkl")
    n_classes = df['region_id'].nunique()

    logger.info(f"📊 Данные: {len(df)} событий, {n_classes} регионов")

    # Подготавливаем последовательности
    X, y = prepare_sequences(df, sequence_length=7, prediction_window=7)
    logger.info(f"   Последовательностей: {len(X)}")

    # Масштабируем признаки
    n_samples, n_steps, n_features = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    # Сохраняем скейлер
    with open(models_dir / "scaler_zone.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Разделяем
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Строим модель
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Обучаем
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    logger.info("⚡ Обучение модели...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

    # Оценка
    y_pred = np.argmax(model.predict(X_test), axis=1)

    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    logger.info(f"  F1-score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    logger.info("=" * 60)

    # Сохраняем модель
    model.save(models_dir / "zone_lstm_model.keras")
    logger.info(f"✅ Модель сохранена в {models_dir / 'zone_lstm_model.keras'}")


if __name__ == "__main__":
    main()
