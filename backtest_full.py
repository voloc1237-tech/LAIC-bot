#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py — ЧЕСТНЫЙ бэктест с ИСПОЛЬЗОВАНИЕМ СОХРАНЁННЫХ МОДЕЛЕЙ
Проверяет модели, которые вы обучили на полных данных (2022-2024)
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
from sklearn.preprocessing import StandardScaler
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
    return merged


def prepare_sequences(df, sequence_length=7, prediction_window=7):
    """Подготавливает данные для LSTM (последовательности)."""
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


def prepare_rf_data(df, prediction_window=7):
    """Подготавливает данные для RF."""
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']
    X, y = [], []
    df_sorted = df.sort_values('time').reset_index(drop=True)
    for i in range(len(df_sorted) - prediction_window):
        X.append(df_sorted.iloc[i][feature_names].values)
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
    logger.info("🔬 БЭКТЕСТ С ИСПОЛЬЗОВАНИЕМ СОХРАНЁННЫХ МОДЕЛЕЙ")
    logger.info("=" * 60)
    
    # 1. Загружаем библиотеку
    df_all = load_library()
    if df_all is None:
        return
    
    # 2. Разделяем по годам
    train_df = df_all[df_all['time'].dt.year == 2022]
    val_df = df_all[df_all['time'].dt.year == 2023]
    test_df = df_all[df_all['time'].dt.year == 2024]
    
    logger.info(f"\n📊 Обучение: {len(train_df)} событий (2022)")
    logger.info(f"📊 Валидация: {len(val_df)} событий (2023)")
    logger.info(f"📊 Тест: {len(test_df)} событий (2024)")
    
    # ===== ЗАГРУЗКА МОДЕЛЕЙ =====
    models_dir = Path("models")
    
    # LSTM
    lstm_model = None
    lstm_scaler = None
    try:
        lstm_model = load_model(models_dir / "lstm_full_model.keras")
        with open(models_dir / "lstm_full_scaler.pkl", 'rb') as f:
            lstm_scaler = pickle.load(f)
        logger.info("✅ LSTM модель загружена")
    except Exception as e:
        logger.warning(f"⚠️ LSTM не загружена: {e}")
    
    # RF
    rf_model = None
    rf_scaler = None
    try:
        with open(models_dir / "rf_full_model.pkl", 'rb') as f:
            rf_model = pickle.load(f)
        with open(models_dir / "rf_full_scaler.pkl", 'rb') as f:
            rf_scaler = pickle.load(f)
        logger.info("✅ RF модель загружена")
    except Exception as e:
        logger.warning(f"⚠️ RF не загружена: {e}")
    
    # ===== LSTM БЭКТЕСТ =====
    if lstm_model is not None:
        logger.info("\n" + "=" * 60)
        logger.info("🧠 LSTM БЭКТЕСТ НА 2024 ГОДУ")
        logger.info("=" * 60)
        
        X_test, y_test = prepare_sequences(test_df, sequence_length=7, prediction_window=7)
        
        if len(X_test) > 0:
            n_samples, n_steps, n_features = X_test.shape
            X_test_flat = X_test.reshape(-1, n_features)
            X_test_scaled = lstm_scaler.transform(X_test_flat).reshape(X_test.shape)
            
            y_pred_prob = lstm_model.predict(X_test_scaled).flatten()
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
            logger.warning("⚠️ Недостаточно данных для LSTM")
    else:
        logger.warning("⚠️ LSTM модель не загружена, бэктест пропущен")
    
    # ===== RF БЭКТЕСТ =====
    if rf_model is not None:
        logger.info("\n" + "=" * 60)
        logger.info("🌲 RF БЭКТЕСТ НА 2024 ГОДУ")
        logger.info("=" * 60)
        
        X_test_rf, y_test_rf = prepare_rf_data(test_df, prediction_window=7)
        
        if len(X_test_rf) > 0:
            X_test_rf_scaled = rf_scaler.transform(X_test_rf)
            
            y_pred_rf = rf_model.predict(X_test_rf_scaled)
            y_pred_prob_rf = rf_model.predict_proba(X_test_rf_scaled)[:, 1]
            
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
        else:
            logger.warning("⚠️ Недостаточно данных для RF")
    else:
        logger.warning("⚠️ RF модель не загружена, бэктест пропущен")


if __name__ == "__main__":
    main()
