#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py — полный бэктест LSTM и RF на исторических данных
Запуск: python backtest_full.py
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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_library():
    """Загружает историческую библиотеку из data/"""
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
        else:
            logger.warning(f"⚠️ Файл за {year} не найден")
    
    if not all_data:
        return None
    
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.drop_duplicates(subset=['id']).sort_values('time')
    logger.info(f"📊 Итого: {len(merged)} событий")
    return merged


def prepare_sequences(df, sequence_length=7, prediction_window=7):
    """
    Подготавливает последовательности для LSTM.
    Возвращает X (признаки) и y (целевая переменная: было ли событие M≥6.0 в следующие 7 дней)
    """
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
    logger.info("🔬 ПОЛНЫЙ БЭКТЕСТ LSTM И RF НА ИСТОРИЧЕСКИХ ДАННЫХ")
    logger.info("=" * 60)
    
    # 1. Загружаем библиотеку
    df_all = load_library()
    if df_all is None:
        logger.error("❌ Библиотека не найдена")
        return
    
    # 2. Разделяем на обучение (2022) и тест (2023-2024)
    train_df = df_all[df_all['time'].dt.year == 2022]
    test_df = df_all[df_all['time'].dt.year.isin([2023, 2024])]
    
    logger.info(f"\n📊 Обучение: {len(train_df)} событий (2022)")
    logger.info(f"📊 Тест: {len(test_df)} событий (2023-2024)")
    
    results = {}
    
    # =========================================================
    # 1. БЭКТЕСТ LSTM
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("🧠 БЭКТЕСТ LSTM")
    logger.info("=" * 60)
    
    # Проверяем, есть ли сохранённая модель
    models_dir = Path("models")
    lstm_path = models_dir / "lstm_model.keras"
    scaler_path = models_dir / "lstm_scaler.pkl"
    
    if lstm_path.exists() and scaler_path.exists():
        try:
            # Загружаем сохранённую модель
            lstm_model = load_model(lstm_path)
            with open(scaler_path, 'rb') as f:
                lstm_scaler = pickle.load(f)
            logger.info("✅ LSTM модель загружена из файла")
            
            # Подготавливаем тестовые данные
            X_test_lstm, y_test_lstm = prepare_sequences(test_df, sequence_length=7, prediction_window=7)
            
            if len(X_test_lstm) > 0:
                # Масштабируем
                n_samples, n_steps, n_features = X_test_lstm.shape
                X_test_flat = X_test_lstm.reshape(-1, n_features)
                X_test_scaled = lstm_scaler.transform(X_test_flat).reshape(X_test_lstm.shape)
                
                # Прогноз
                y_pred_prob = lstm_model.predict(X_test_scaled).flatten()
                y_pred = (y_pred_prob >= 0.5).astype(int)
                
                # Метрики
                results['lstm'] = {
                    'accuracy': accuracy_score(y_test_lstm, y_pred),
                    'precision': precision_score(y_test_lstm, y_pred, zero_division=0),
                    'recall': recall_score(y_test_lstm, y_pred, zero_division=0),
                    'f1': f1_score(y_test_lstm, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test_lstm, y_pred_prob) if len(np.unique(y_test_lstm)) > 1 else 0.5
                }
                
                logger.info(f"📊 Результаты LSTM на тесте (2023-2024):")
                logger.info(f"  Accuracy:  {results['lstm']['accuracy']:.3f}")
                logger.info(f"  Precision: {results['lstm']['precision']:.3f}")
                logger.info(f"  Recall:    {results['lstm']['recall']:.3f}")
                logger.info(f"  F1-score:  {results['lstm']['f1']:.3f}")
                logger.info(f"  ROC-AUC:   {results['lstm']['auc']:.3f}")
            else:
                logger.warning("⚠️ Недостаточно тестовых данных для LSTM")
                results['lstm'] = None
                
        except Exception as e:
            logger.error(f"❌ Ошибка LSTM: {e}")
            results['lstm'] = None
    else:
        logger.warning("⚠️ LSTM модель не найдена, пропускаем")
        results['lstm'] = None
    
    # =========================================================
    # 2. БЭКТЕСТ RF
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("🌲 БЭКТЕСТ RANDOMFOREST")
    logger.info("=" * 60)
    
    rf_path = models_dir / "zone_rf_model.pkl"
    rf_scaler_path = models_dir / "scaler_rf.pkl"
    
    if rf_path.exists() and rf_scaler_path.exists():
        try:
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            with open(rf_scaler_path, 'rb') as f:
                rf_scaler = pickle.load(f)
            logger.info("✅ RF модель загружена из файла")
            
            # Подготавливаем тестовые данные
            X_test_rf, y_test_rf = prepare_rf_data(test_df, prediction_window=7)
            
            if len(X_test_rf) > 0:
                X_test_rf_scaled = rf_scaler.transform(X_test_rf)
                
                y_pred_rf = rf_model.predict(X_test_rf_scaled)
                y_pred_prob_rf = rf_model.predict_proba(X_test_rf_scaled)[:, 1]
                
                results['rf'] = {
                    'accuracy': accuracy_score(y_test_rf, y_pred_rf),
                    'precision': precision_score(y_test_rf, y_pred_rf, zero_division=0),
                    'recall': recall_score(y_test_rf, y_pred_rf, zero_division=0),
                    'f1': f1_score(y_test_rf, y_pred_rf, zero_division=0),
                    'auc': roc_auc_score(y_test_rf, y_pred_prob_rf) if len(np.unique(y_test_rf)) > 1 else 0.5
                }
                
                logger.info(f"📊 Результаты RF на тесте (2023-2024):")
                logger.info(f"  Accuracy:  {results['rf']['accuracy']:.3f}")
                logger.info(f"  Precision: {results['rf']['precision']:.3f}")
                logger.info(f"  Recall:    {results['rf']['recall']:.3f}")
                logger.info(f"  F1-score:  {results['rf']['f1']:.3f}")
                logger.info(f"  ROC-AUC:   {results['rf']['auc']:.3f}")
            else:
                logger.warning("⚠️ Недостаточно тестовых данных для RF")
                results['rf'] = None
                
        except Exception as e:
            logger.error(f"❌ Ошибка RF: {e}")
            results['rf'] = None
    else:
        logger.warning("⚠️ RF модель не найдена, пропускаем")
        results['rf'] = None
    
    # =========================================================
    # 3. СОХРАНЯЕМ РЕЗУЛЬТАТЫ
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)
    
    # Сохраняем в CSV
    df_results = pd.DataFrame({
        'model': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    })
    
    for model_name in ['lstm', 'rf']:
        if results.get(model_name):
            df_results = pd.concat([df_results, pd.DataFrame({
                'model': [model_name.upper()],
                'accuracy': [results[model_name]['accuracy']],
                'precision': [results[model_name]['precision']],
                'recall': [results[model_name]['recall']],
                'f1': [results[model_name]['f1']],
                'auc': [results[model_name]['auc']]
            })], ignore_index=True)
    
    # Сохраняем CSV
    df_results.to_csv('backtest_full_results.csv', index=False)
    logger.info("✅ Результаты сохранены в backtest_full_results.csv")
    
    # Сохраняем текстовый отчёт
    with open('backtest_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("🔬 ПОЛНЫЙ БЭКТЕСТ LSTM И RF\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Обучение: 2022 год ({len(train_df)} событий)\n")
        f.write(f"Тест: 2023-2024 годы ({len(test_df)} событий)\n\n")
        
        for model_name in ['lstm', 'rf']:
            if results.get(model_name):
                r = results[model_name]
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Accuracy:  {r['accuracy']:.3f}\n")
                f.write(f"  Precision: {r['precision']:.3f}\n")
                f.write(f"  Recall:    {r['recall']:.3f}\n")
                f.write(f"  F1-score:  {r['f1']:.3f}\n")
                f.write(f"  ROC-AUC:   {r['auc']:.3f}\n")
            else:
                f.write(f"\n{model_name.upper()}: не удалось выполнить бэктест\n")
    
    logger.info("✅ Текстовый отчёт сохранён в backtest_report.txt")
    
    # Выводим итоговую таблицу
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
