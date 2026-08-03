#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py — прогнозирование землетрясений с использованием LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class EarthquakePredictor:
    """
    LSTM-модель для прогнозирования вероятности сильного землетрясения.
    """
    
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 7  # дней
        self.feature_names = ['magnitude', 'energy', 'kp_mean', 'temp_mean']
    
    def _prepare_sequences(self, df, target_col='magnitude'):
        """
        Подготавливает последовательности для LSTM.
        """
        # Извлекаем признаки
        data = df[self.feature_names].values
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        
        # Создаём последовательности
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # предсказываем магнитуду
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Строит LSTM-модель.
        """
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, df, epochs=50, validation_split=0.2):
        """
        Обучает модель на исторических данных.
        """
        if len(df) < 20:
            logger.warning("⚠️ Слишком мало данных для обучения LSTM (нужно ≥20)")
            return False
        
        X, y = self._prepare_sequences(df)
        if len(X) < 10:
            logger.warning("⚠️ Недостаточно последовательностей")
            return False
        
        # Разделяем на train/test
        split = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Строим модель
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        # Обучаем
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Сохраняем модель и скалер
        self.save_model()
        
        logger.info(f"🧠 LSTM обучена на {len(X)} последовательностях")
        logger.info(f"   Loss: {history.history['loss'][-1]:.4f}, Val Loss: {history.history['val_loss'][-1]:.4f}")
        return True
    
    def predict(self, df):
        """
        Прогнозирует вероятность сильного землетрясения.
        Возвращает вероятность (0-1) и прогнозируемую магнитуду.
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.warning("⚠️ Модель не загружена")
                return None, None
        
        if len(df) < self.sequence_length:
            logger.warning(f"⚠️ Недостаточно данных для прогноза (нужно {self.sequence_length} дней)")
            return None, None
        
        # Подготавливаем данные
        recent_data = df.tail(self.sequence_length)
        scaled = self.scaler.transform(recent_data[self.feature_names].values)
        X_input = np.array([scaled])
        
        # Прогнозируем
        pred_scaled = self.model.predict(X_input, verbose=0)[0][0]
        pred_magnitude = self.scaler.inverse_transform(
            np.array([[pred_scaled, 0, 0, 0]])
        )[0][0]
        
        # Вероятность сильного землетрясения (M≥6.0)
        # Используем сигмоидоподобную функцию
        probability = 1 / (1 + np.exp(-(pred_magnitude - 5.5) * 2))
        probability = min(max(probability, 0), 0.95)
        
        return probability, pred_magnitude
    
    def save_model(self):
        """Сохраняет модель и скалер."""
        if self.model is not None:
            self.model.save(os.path.join(self.model_dir, 'lstm_model.keras'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            logger.info("💾 Модель сохранена")
    
    def load_model(self):
        """Загружает модель и скалер."""
        try:
            from tensorflow.keras.models import load_model
            model_path = os.path.join(self.model_dir, 'lstm_model.keras')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("📂 Модель загружена")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки модели: {e}")
        
        return False
