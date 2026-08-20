#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_lstm.py — обучение LSTM на полных годовых данных
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_full_year(year):
    """Загружает данные за полный год."""
    data_dir = Path(__file__).parent.parent / "data"
    path = data_dir / f"cache_{year}01_{year}12.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
        logger.info(f"✅ {year}: {len(df)} событий")
        return df
    logger.warning(f"⚠️ Файл за {year} не найден")
    return None


def prepare_sequences(df, sequence_length=7, prediction_window=7):
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean', 'swarm_anomaly']
    X, y = [], []
    df_sorted = df.sort_values('time').reset_index(drop=True)
    for i in range(sequence_length, len(df_sorted) - prediction_window):
        X.append(df_sorted.iloc[i-sequence_length:i][feature_names].values)
        event_time = df_sorted.iloc[i]['time']
        future = df_sorted[
            (df_sorted['time'] > event_time) &
            (df_sorted['time'] <= event_time + timedelta(days=prediction_window)) &
            (df_sorted['magnitude'] >= 6.0)
        ]
        y.append(1 if not future.empty else 0)
    return np.array(X), np.array(y)


def main():
    logger.info("=" * 60)
    logger.info("🧠 ОБУЧЕНИЕ LSTM НА ПОЛНЫХ ГОДОВЫХ ДАННЫХ")
    logger.info("=" * 60)
    
    # 1. Загружаем данные
    years = [2022, 2023, 2024]
    all_dfs = []
    for year in years:
        df = load_full_year(year)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        logger.error("❌ Нет данных")
        return
    
    # 2. Объединяем все годы
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['id']).sort_values('time')
    logger.info(f"📊 Итого: {len(df_all)} событий за {years}")
    
    # 3. Подготавливаем последовательности
    X, y = prepare_sequences(df_all, sequence_length=7, prediction_window=7)
    logger.info(f"📊 Последовательностей: {len(X)}")
    logger.info(f"   Положительных: {sum(y)}, Отрицательных: {len(y)-sum(y)}")
    
    # 4. Разделяем на train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # 5. Масштабируем
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    # 6. Строим модель
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 7. Обучаем
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    logger.info("⚡ Обучение LSTM...")
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
              validation_data=(X_val_scaled, y_val), callbacks=[early_stop], verbose=1)
    
    # 8. Оцениваем
    y_pred_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_prob):.3f}")
    logger.info("=" * 60)
    
    # 9. Сохраняем модель и скейлер
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save(models_dir / "lstm_full_model.keras")
    with open(models_dir / "lstm_full_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"✅ Модель сохранена в {models_dir / 'lstm_full_model.keras'}")


if __name__ == "__main__":
    main()
