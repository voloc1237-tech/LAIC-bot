#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py — ЧЕСТНЫЙ бэктест (без утечки данных)
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
    return merged

def prepare_sequences(df, sequence_length=7, prediction_window=7):
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']
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
    logger.info("🔬 ЧЕСТНЫЙ БЭКТЕСТ (БЕЗ УТЕЧКИ ДАННЫХ)")
    logger.info("=" * 60)
    
    df_all = load_library()
    if df_all is None:
        return
    
    # Разделяем по годам
    train_df = df_all[df_all['time'].dt.year == 2022]
    val_df = df_all[df_all['time'].dt.year == 2023]
    test_df = df_all[df_all['time'].dt.year == 2024]
    
    logger.info(f"\n📊 Обучение: {len(train_df)} событий (2022)")
    logger.info(f"📊 Валидация: {len(val_df)} событий (2023)")
    logger.info(f"📊 Тест: {len(test_df)} событий (2024)")
    
    # ===== LSTM =====
    logger.info("\n" + "=" * 60)
    logger.info("🧠 LSTM (обучение на 2022, тест на 2024)")
    logger.info("=" * 60)
    
    X_train, y_train = prepare_sequences(train_df)
    X_test, y_test = prepare_sequences(test_df)
    
    if len(X_train) > 0 and len(X_test) > 0:
        # Масштабируем
        n_samples, n_steps, n_features = X_train.shape
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        # Строим модель (не загружаем сохранённую, чтобы избежать утечки)
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(n_steps, n_features)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        logger.info("⚡ Обучение LSTM на 2022 году...")
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)
        
        y_pred_prob = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        logger.info("📊 РЕЗУЛЬТАТЫ LSTM НА 2024 ГОДУ:")
        logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
        logger.info(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
        logger.info(f"  F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
            logger.info(f"  ROC-AUC:   {auc:.3f}")
        except:
            pass
    else:
        logger.warning("⚠️ Недостаточно данных")
    
    # ===== RF =====
    logger.info("\n" + "=" * 60)
    logger.info("🌲 RF (обучение на 2022, тест на 2024)")
    logger.info("=" * 60)
    
    X_train_rf, y_train_rf = prepare_rf_data(train_df)
    X_test_rf, y_test_rf = prepare_rf_data(test_df)
    
    if len(X_train_rf) > 0 and len(X_test_rf) > 0:
        scaler_rf = StandardScaler()
        X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
        X_test_rf_scaled = scaler_rf.transform(X_test_rf)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        logger.info("⚡ Обучение RF на 2022 году...")
        rf.fit(X_train_rf_scaled, y_train_rf)
        
        y_pred_rf = rf.predict(X_test_rf_scaled)
        y_pred_prob_rf = rf.predict_proba(X_test_rf_scaled)[:, 1]
        
        logger.info("📊 РЕЗУЛЬТАТЫ RF НА 2024 ГОДУ:")
        logger.info(f"  Accuracy:  {accuracy_score(y_test_rf, y_pred_rf):.3f}")
        logger.info(f"  Precision: {precision_score(y_test_rf, y_pred_rf, zero_division=0):.3f}")
        logger.info(f"  Recall:    {recall_score(y_test_rf, y_pred_rf, zero_division=0):.3f}")
        logger.info(f"  F1-score:  {f1_score(y_test_rf, y_pred_rf, zero_division=0):.3f}")
        try:
            auc = roc_auc_score(y_test_rf, y_pred_prob_rf)
            logger.info(f"  ROC-AUC:   {auc:.3f}")
        except:
            pass

if __name__ == "__main__":
    main()
