#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anomaly_detector.py — ИИ-анализ аномалий в данных LAIC
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Обнаружение аномалий в данных LAIC.
    Использует Isolation Forest для выявления необычных паттернов.
    """
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = []
    
    def prepare_features(self, event_data):
        """
        Преобразует данные события в вектор признаков.
        event_data: словарь с данными (магнитуда, погода, Kp, LST)
        """
        features = []
        
        # 1. Магнитуда и глубина
        mag = event_data.get('magnitude', 0)
        depth = event_data.get('depth_km', 0)
        features.extend([mag, depth])
        
        # 2. Погода (средняя температура)
        weather = event_data.get('weather', {})
        temp_mean = weather.get('temp_mean', 0)
        humidity_mean = weather.get('humidity_mean', 0)
        features.extend([temp_mean, humidity_mean])
        
        # 3. Kp-индекс (средний, максимум, вариация)
        kp_data = event_data.get('kp', {})
        kp_mean = kp_data.get('mean', 0)
        kp_max = kp_data.get('max', 0)
        kp_std = kp_data.get('std', 0)
        features.extend([kp_mean, kp_max, kp_std])
        
        # 4. LST (температура поверхности)
        lst = event_data.get('lst_celsius', 0)
        features.append(lst)
        
        self.feature_names = [
            'magnitude', 'depth_km',
            'temp_mean', 'humidity_mean',
            'kp_mean', 'kp_max', 'kp_std',
            'lst_celsius'
        ]
        
        return features
    
    def fit(self, X):
        """Обучает модель на исторических данных."""
        if len(X) < 10:
            logger.warning("⚠️ Слишком мало данных для обучения модели (нужно минимум 10)")
            self.fitted = False
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        logger.info("🧠 Модель аномалий обучена на %d образцах", len(X))
        return True
    
    def predict(self, X):
        """Предсказывает аномалии (1 – норма, -1 – аномалия)."""
        if not self.fitted:
            raise ValueError("Модель не обучена")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_single(self, event_data):
        """Анализирует одно событие и возвращает статус аномалии."""
        if not self.fitted:
            logger.warning("⚠️ Модель не обучена, пропускаем")
            return None
        X = np.array([self.prepare_features(event_data)])
        return self.predict(X)[0]
    
    def analyze_events(self, events_data):
        """
        Анализирует список событий и возвращает DataFrame с результатами.
        events_data: список словарей с данными событий
        """
        if not events_data:
            return pd.DataFrame()
        
        # Подготовка признаков
        X = []
        valid_indices = []
        for i, evt in enumerate(events_data):
            try:
                features = self.prepare_features(evt)
                X.append(features)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Ошибка подготовки признаков для события {i}: {e}")
                continue
        
        if len(X) < 10:
            logger.warning("⚠️ Недостаточно данных для обучения (нужно >= 10)")
            return pd.DataFrame()
        
        # Обучение модели
        X = np.array(X)
        self.fit(X)
        
        if not self.fitted:
            return pd.DataFrame()
        
        # Предсказание
        predictions = self.predict(X)
        
        # Формируем результаты
        results = []
        for i, idx in enumerate(valid_indices):
            evt = events_data[idx]
            results.append({
                'event_id': evt.get('id', f'event_{idx}'),
                'magnitude': evt.get('magnitude', 0),
                'is_anomaly': predictions[i] == -1,
                'anomaly_score': predictions[i],
                'region': evt.get('region', 'global'),
                'time': evt.get('time', None)
            })
        
        df = pd.DataFrame(results)
        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"🧠 Обнаружено аномалий: {anomaly_count} из {len(df)} событий")
        
        return df
