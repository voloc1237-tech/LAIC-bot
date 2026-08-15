#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_zone_lstm_lite.py — LSTM Lite для классификации регионов
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
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_sequences(df, sequence_length=5, prediction_window=5):
    """Подготавливает короткие последовательности для LSTM Lite."""
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
    logger.info("🧠 LSTM LITE ДЛЯ КЛАССИФИКАЦИИ РЕГИОНОВ")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Загружаем данные
    df = pd.read_pickle(data_dir / "events_with_regions.pkl")
    n_classes = df['region_id'].nunique()

    logger.info(f"📊 Данные: {len(df)} событий, {n_classes} регионов")

    # Короткие последовательности (5 дней вместо 7)
    X, y = prepare_sequences(df, sequence_length=5, prediction_window=5)
    logger.info(f"   Последовательностей: {len(X)}")

    # Масштабируем
    n_samples, n_steps, n_features = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    with open(models_dir / "scaler_lstm_lite.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Разделяем
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # ===== LSTM LITE =====
    model = Sequential([
        # Один слой LSTM с сильной регуляризацией
        LSTM(8, activation='relu', input_shape=(n_steps, n_features),
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(n_classes, activation='softmax', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Ранняя остановка
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    logger.info("⚡ Обучение LSTM Lite...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Оценка
    y_pred = np.argmax(model.predict(X_test), axis=1)

    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ (LSTM Lite):")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"  F1-score (weighted): {f1_score(y_test, y_pred, average='weighted'):.3f}")
    logger.info("=" * 60)

    # Сохраняем модель
    model.save(models_dir / "zone_lstm_lite_model.keras")
    logger.info(f"✅ Модель сохранена в {models_dir / 'zone_lstm_lite_model.keras'}")


if __name__ == "__main__":
    main()
