#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm_model.py — обучение LSTM на объединённой библиотеке 2022-2024.
Сохраняет модель и скейлер для использования в боте.
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMTrainer:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 7
        self.feature_names = [
            'magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
            'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean'
        ]
        
    def load_library(self, years=[2022, 2023, 2024]):
        """Загружает объединённую библиотеку данных."""
        all_data = []
        for year in years:
            path = Path(f"data/cache_{year}08_{year}12.json")
            if not path.exists():
                logger.warning(f"⚠️ Файл за {year} не найден, пропускаем")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], utc=True)
            all_data.append(df)
            logger.info(f"✅ {year}: {len(df)} событий")
        
        if not all_data:
            logger.error("❌ Нет данных для обучения")
            return None
        
        merged = pd.concat(all_data, ignore_index=True)
        merged = merged.drop_duplicates(subset=['id']).sort_values('time')
        logger.info(f"📊 Итого: {len(merged)} событий")
        return merged
    
    def prepare_sequences(self, df):
        """
        Подготавливает последовательности для LSTM.
        Создаёт целевую переменную: событие M≥6.0 в следующие 7 дней.
        """
        X, y = [], []
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        for i in range(self.sequence_length, len(df_sorted)):
            # Признаки за последние sequence_length дней
            X.append(df_sorted.iloc[i-self.sequence_length:i][self.feature_names].values)
            
            # Целевая переменная: было ли событие M≥6.0 в следующие 7 дней
            event_time = df_sorted.iloc[i]['time']
            future = df_sorted[
                (df_sorted['time'] > event_time) &
                (df_sorted['time'] <= event_time + timedelta(days=7)) &
                (df_sorted['magnitude'] >= 6.0)
            ]
            y.append(1 if not future.empty else 0)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Строит LSTM-модель."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # бинарная классификация
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Обучает модель с ранней остановкой."""
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        """Оценивает модель на тестовой выборке."""
        y_pred_prob = self.model.predict(X_test).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_prob)
        }
        return metrics
    
    def save_model(self, model_path="models/lstm_model.keras", scaler_path="models/lstm_scaler.pkl"):
        """Сохраняет модель и скейлер."""
        Path("models").mkdir(exist_ok=True)
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"💾 Модель сохранена в {model_path}")
        logger.info(f"💾 Скейлер сохранён в {scaler_path}")
    
    def load_model(self, model_path="models/lstm_model.keras", scaler_path="models/lstm_scaler.pkl"):
        """Загружает модель и скейлер."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"📂 Модель загружена из {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False


def main():
    # 1. Загружаем данные
    logger.info("📥 Загрузка библиотеки данных...")
    trainer = LSTMTrainer()
    df = trainer.load_library(years=[2022, 2023, 2024])
    if df is None:
        return

    # 2. Подготавливаем последовательности
    logger.info("🔄 Подготовка последовательностей...")
    X, y = trainer.prepare_sequences(df)
    logger.info(f"📊 Последовательностей: {len(X)}, положительных: {sum(y)}, отрицательных: {len(y)-sum(y)}")

    # 3. Разделяем на train/val/test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # 4. Масштабируем признаки (для каждого временного шага отдельно)
    # Приводим к 2D для масштабирования
    n_samples, n_steps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    trainer.scaler.fit(X_train_flat)
    
    # Масштабируем все наборы
    X_train_scaled = trainer.scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = trainer.scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = trainer.scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # 5. Строим модель
    logger.info("🧠 Построение LSTM модели...")
    trainer.model = trainer.build_model((n_steps, n_features))

    # 6. Обучаем
    logger.info("⚡ Обучение модели...")
    trainer.train(X_train_scaled, y_train, X_val_scaled, y_val, epochs=50)

    # 7. Оцениваем
    logger.info("📊 Оценка модели на тестовой выборке...")
    metrics = trainer.evaluate(X_test_scaled, y_test)
    logger.info("="*50)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1-score:  {metrics['f1']:.3f}")
    logger.info(f"  ROC-AUC:   {metrics['auc']:.3f}")
    logger.info("="*50)

    # 8. Сохраняем модель
    trainer.save_model()

if __name__ == "__main__":
    main()
