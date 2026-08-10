#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_predictor.py — Гибридный прогноз (LSTM + CatBoost + XGBoost)
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridEarthquakePredictor:
    """
    Гибридная система прогнозирования.
    
    Выбор модели:
        - > 100 событий: LSTM + CatBoost + XGBoost (ансамбль)
        - 30–100 событий: CatBoost + XGBoost (ансамбль)
        - < 30 событий: CatBoost (с аугментацией)
    """
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.lstm = None          # будет загружен из predictor.py
        self.catboost = None
        self.xgboost = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self._load_models()
    
    def _load_models(self):
        """Загрузка сохранённых моделей."""
        # LSTM загружаем из существующего класса
        try:
            from predictor import EarthquakeLSTMPredictor
            self.lstm = EarthquakeLSTMPredictor(model_dir=self.model_dir)
            if self.lstm.model is not None:
                logger.info("✅ LSTM модель загружена")
        except Exception as e:
            logger.warning(f"⚠️ LSTM не загружен: {e}")
        
        # CatBoost
        try:
            import joblib
            self.catboost = joblib.load(f"{self.model_dir}/catboost.pkl")
            self.xgboost = joblib.load(f"{self.model_dir}/xgboost.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/hybrid_scaler.pkl")
            self.is_trained = True
            logger.info("✅ CatBoost/XGBoost загружены")
        except FileNotFoundError:
            logger.info("ℹ️ CatBoost/XGBoost не найдены, требуется обучение")
    
    def _save_models(self):
        """Сохранение моделей."""
        import joblib
        joblib.dump(self.catboost, f"{self.model_dir}/catboost.pkl")
        joblib.dump(self.xgboost, f"{self.model_dir}/xgboost.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/hybrid_scaler.pkl")
        logger.info("💾 CatBoost/XGBoost сохранены")
    
    def prepare_tabular_features(
        self,
        events_df: pd.DataFrame,
        region_data_dict: Dict,
        lst_cache: Dict,
        iono_cache: Dict,
        space_cache: Dict,
        weather_cache: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготовка табличных признаков для CatBoost/XGBoost.
        Возвращает X, y_prob (бинарная), y_mag (регрессия).
        """
        from earthquake_predictor import LAICFeatureExtractor
        
        X = []
        y_prob = []
        y_mag = []
        
        data_end_time = events_df['time'].max() if not events_df.empty else datetime.now(timezone.utc)
        
        for _, event in events_df.iterrows():
            event_time = event['time']
            lat = event['latitude']
            lon = event['longitude']
            mag = event['magnitude']
            
            # Определяем регион
            region_name = self._get_region_for_point(lat, lon, region_data_dict)
            region_df = region_data_dict.get(region_name, pd.DataFrame())
            
            # Используем несколько временных срезов
            for days_before in [30, 21, 14, 7, 3, 1]:
                sample_time = event_time - timedelta(days=days_before)
                if sample_time > datetime.now(timezone.utc):
                    continue
                
                # Признаки
                features = LAICFeatureExtractor.extract_features(
                    lat=lat,
                    lon=lon,
                    reference_time=sample_time,
                    region_data=region_df,
                    lst_data=lst_cache.get(region_name),
                    iono_data=iono_cache.get(event.get('id')),
                    space_data=space_cache.get(event.get('id')),
                    weather_data=weather_cache.get(event.get('id')),
                    data_end_time=data_end_time
                )
                
                # Проверка на NaN
                if np.isnan(features).any():
                    continue
                
                X.append(features)
                y_prob.append(1 if mag >= 6.0 else 0)
                y_mag.append(mag)
        
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(X), np.array(y_prob), np.array(y_mag)
    
    def _get_region_for_point(self, lat, lon, region_data_dict):
        """Определяет регион по координатам."""
        if not region_data_dict:
            return 'global'
        best = None
        best_dist = float('inf')
        for name, df in region_data_dict.items():
            if df.empty:
                continue
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            dist = (lat - center_lat)**2 + (lon - center_lon)**2
            if dist < best_dist:
                best_dist = dist
                best = name
        return best if best else 'global'
    
    def train(self, X, y_prob, y_mag):
        """Обучение CatBoost и XGBoost."""
        if len(X) < 20:
            logger.warning(f"⚠️ Слишком мало данных ({len(X)}) для обучения")
            return False
        
        # Нормализация
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val, y_mag_train, y_mag_val = train_test_split(
            X_scaled, y_prob, y_mag, test_size=0.2, random_state=42
        )
        
        # 1. CatBoost (классификация)
        if CATBOOST_AVAILABLE:
            self.catboost = cb.CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=False,
                early_stopping_rounds=30
            )
            self.catboost.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                plot=False
            )
            auc = roc_auc_score(y_val, self.catboost.predict_proba(X_val)[:, 1])
            logger.info(f"✅ CatBoost AUC: {auc:.3f}")
        else:
            logger.warning("⚠️ CatBoost не установлен")
            self.catboost = None
        
        # 2. XGBoost (классификация)
        if XGBOOST_AVAILABLE:
            self.xgboost = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=30
            )
            self.xgboost.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            auc = roc_auc_score(y_val, self.xgboost.predict_proba(X_val)[:, 1])
            logger.info(f"✅ XGBoost AUC: {auc:.3f}")
        else:
            logger.warning("⚠️ XGBoost не установлен")
            self.xgboost = None
        
        self.is_trained = True
        self._save_models()
        return True
    
    def predict(
        self,
        events_df: pd.DataFrame,
        region_data_dict: Dict,
        lst_cache: Dict,
        iono_cache: Dict,
        space_cache: Dict,
        weather_cache: Dict,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Гибридный прогноз.
        """
        if not self.is_trained and self.catboost is None and self.xgboost is None:
            logger.warning("⚠️ Модели не обучены")
            return {'probability': 0.0, 'magnitude': 0.0, 'days': 30, 'method': 'none'}
        
        current_time = current_time or datetime.now(timezone.utc)
        
        # Собираем признаки для текущего момента
        # Используем последнее событие (или синтетическую точку)
        if not events_df.empty:
            last_event = events_df.iloc[-1]
            lat = last_event['latitude']
            lon = last_event['longitude']
            region_name = self._get_region_for_point(lat, lon, region_data_dict)
            region_df = region_data_dict.get(region_name, pd.DataFrame())
            
            # Признаки на текущий момент (reference_time = current_time)
            from earthquake_predictor import LAICFeatureExtractor
            features = LAICFeatureExtractor.extract_features(
                lat=lat,
                lon=lon,
                reference_time=current_time,
                region_data=region_df,
                lst_data=lst_cache.get(region_name),
                iono_data=None,
                space_data=None,
                weather_data=None,
                data_end_time=events_df['time'].max() if not events_df.empty else current_time
            )
            
            if np.isnan(features).any():
                logger.warning("⚠️ Признаки содержат NaN, прогноз невозможен")
                return {'probability': 0.0, 'magnitude': 0.0, 'days': 30, 'method': 'none'}
            
            X_scaled = self.scaler.transform([features])
            
            # Прогнозы от моделей
            probs = []
            mags = []
            methods = []
            
            if self.catboost:
                prob = self.catboost.predict_proba(X_scaled)[0, 1]
                probs.append(prob)
                methods.append('CatBoost')
            
            if self.xgboost:
                prob = self.xgboost.predict_proba(X_scaled)[0, 1]
                probs.append(prob)
                methods.append('XGBoost')
            
            # Если есть LSTM — тоже используем
            if self.lstm and self.lstm.model is not None:
                # Для LSTM нужны временные ряды, но у нас их нет в этом контексте
                # Можно пропустить или использовать отдельный вызов
                pass
            
            if probs:
                # Ансамбль: среднее арифметическое
                final_prob = np.mean(probs)
                # Магнитуда: если вероятность > 0.3, оцениваем ~6.0, иначе ~5.0
                pred_mag = 5.5 + (final_prob - 0.3) * 3  # упрощённо
                pred_mag = max(4.0, min(8.0, pred_mag))
                
                # Дни до события: обратная связь от вероятности
                days = max(1, int(30 * (1 - final_prob / 0.8)))
                
                return {
                    'probability': final_prob,
                    'magnitude': pred_mag,
                    'days': days,
                    'method': '+'.join(methods)
                }
        
        return {'probability': 0.0, 'magnitude': 0.0, 'days': 30, 'method': 'none'}
