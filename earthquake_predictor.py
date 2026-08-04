#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
earthquake_predictor.py — ИИ-прогнозирование землетрясений
На основе LAIC-предвестников (ионосфера, спутниковые данные, космическая погода)
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PredictionConfig:
    """Конфигурация прогнозирования."""
    MAGNITUDE_THRESHOLD: float = 6.0      # Порог для "крупного" события
    GRID_RESOLUTION: float = 2.0           # Шаг сетки в градусах
    PROBABILITY_THRESHOLD: float = 0.3     # Минимальная вероятность для отчёта
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6 # Порог для алерта
    DAYS_BEFORE_FEATURES: List[int] = None # Дни до события для обучения
    
    def __post_init__(self):
        if self.DAYS_BEFORE_FEATURES is None:
            self.DAYS_BEFORE_FEATURES = [30, 21, 14, 7, 3, 0]


CONFIG = PredictionConfig()


# ═══════════════════════════════════════════════════════════════
# КЛАСС ПРИЗНАКОВ (FEATURE EXTRACTOR)
# ═══════════════════════════════════════════════════════════════

class LAICFeatureExtractor:
    """Извлечение признаков из LAIC-данных."""
    
    @staticmethod
    def extract_features(
        lat: float,
        lon: float,
        event_time: datetime,
        region_data: pd.DataFrame,
        lst_data: Optional[Dict] = None,
        iono_data: Optional[Dict] = None,
        space_data: Optional[Dict] = None,
        weather_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Извлечение вектора признаков для точки и времени.
        """
        features = {}
        
        # 1. Сейсмические признаки (история региона)
        seismic_features = LAICFeatureExtractor._get_seismic_features(
            lat, lon, event_time, region_data
        )
        features.update(seismic_features)
        
        # 2. Ионосферные признаки
        if iono_data:
            features.update({
                'tec_anomaly_max': iono_data.get('z_score_max', 0),
                'tec_anomaly_min': iono_data.get('z_score_min', 0),
                'iono_anomaly_level': LAICFeatureExtractor._encode_level(
                    iono_data.get('level', 'normal')
                ),
                'iono_historical_years': iono_data.get('n_historical_years', 0),
                'iono_yoy_change': iono_data.get('year_over_year_change', 0),
            })
        else:
            features.update({
                'tec_anomaly_max': 0,
                'tec_anomaly_min': 0,
                'iono_anomaly_level': 0,
                'iono_historical_years': 0,
                'iono_yoy_change': 0,
            })
        
        # 3. Спутниковые признаки
        if lst_data and isinstance(lst_data, dict):
            lst_val = lst_data.get('lst_celsius', None)
            features.update({
                'lst_available': 1 if lst_val is not None else 0,
                'lst_value': lst_val if lst_val is not None else 20,
                'lst_source_real': 1 if 'sentinel' in str(lst_data.get('source', '')) else 0,
            })
        else:
            features.update({
                'lst_available': 0,
                'lst_value': 20,
                'lst_source_real': 0,
            })
        
        # 4. Космическая погода
        if space_data:
            dst_df = space_data.get('dst', pd.DataFrame())
            kp_df = space_data.get('kp', pd.DataFrame())
            f107_df = space_data.get('f107', pd.DataFrame())
            
            features.update({
                'dst_mean': dst_df['dst'].mean() if not dst_df.empty else -10,
                'dst_std': dst_df['dst'].std() if not dst_df.empty else 5,
                'dst_min': dst_df['dst'].min() if not dst_df.empty else -50,
                'kp_max': kp_df['kp'].max() if not kp_df.empty else 3,
                'kp_mean': kp_df['kp'].mean() if not kp_df.empty else 1.5,
                'f107_mean': f107_df['f107'].mean() if not f107_df.empty else 140,
                'f107_trend': LAICFeatureExtractor._compute_trend(f107_df) if not f107_df.empty else 0,
            })
        else:
            features.update({
                'dst_mean': -10, 'dst_std': 5, 'dst_min': -50,
                'kp_max': 3, 'kp_mean': 1.5,
                'f107_mean': 140, 'f107_trend': 0,
            })
        
        # 5. Временные признаки
        features.update({
            'day_of_year': event_time.timetuple().tm_yday / 365.0,
            'hour': event_time.hour / 24.0,
            'month': event_time.month / 12.0,
        })
        
        # 6. Географические признаки
        features.update({
            'lat': lat / 90.0,
            'lon': lon / 180.0,
            'abs_lat': abs(lat) / 90.0,
            'is_northern': 1 if lat > 0 else 0,
        })
        
        # 7. Погодные признаки
        if weather_data is not None and not weather_data.empty:
            features.update({
                'temp_mean': weather_data.get('temp_mean', weather_data.get('temperature', pd.Series([20])).mean()),
                'temp_range': weather_data.get('temp_max', pd.Series([25])).max() - weather_data.get('temp_min', pd.Series([15])).min() if not weather_data.empty else 10,
                'humidity_mean': weather_data.get('humidity', pd.Series([60])).mean() if not weather_data.empty else 60,
            })
        else:
            features.update({
                'temp_mean': 20,
                'temp_range': 10,
                'humidity_mean': 60,
            })
        
        # Упорядоченный вектор признаков
        feature_names = [
            'seismic_rate_30d', 'seismic_rate_7d', 'b_value', 'max_mag_30d',
            'mean_mag_30d', 'depth_mean', 'events_count_near',
            'tec_anomaly_max', 'tec_anomaly_min', 'iono_anomaly_level',
            'iono_historical_years', 'iono_yoy_change',
            'lst_available', 'lst_value', 'lst_source_real',
            'dst_mean', 'dst_std', 'dst_min', 'kp_max', 'kp_mean',
            'f107_mean', 'f107_trend',
            'day_of_year', 'hour', 'month',
            'lat', 'lon', 'abs_lat', 'is_northern',
            'temp_mean', 'temp_range', 'humidity_mean'
        ]
        
        return np.array([features.get(name, 0) for name in feature_names])
    
    @staticmethod
    def _get_seismic_features(
        lat: float,
        lon: float,
        event_time: datetime,
        region_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Сейсмические признаки из истории."""
        if region_data.empty:
            return {
                'seismic_rate_30d': 0,
                'seismic_rate_7d': 0,
                'b_value': 1.0,
                'max_mag_30d': 0,
                'mean_mag_30d': 0,
                'depth_mean': 10,
                'events_count_near': 0,
            }
        
        # Фильтруем события до текущего времени
        past_events = region_data[region_data['time'] < event_time]
        
        # События за 30 и 7 дней
        days_30 = event_time - timedelta(days=30)
        days_7 = event_time - timedelta(days=7)
        
        events_30d = past_events[past_events['time'] >= days_30]
        events_7d = past_events[past_events['time'] >= days_7]
        
        # События в радиусе 200 км
        near_events = past_events[
            ((past_events['latitude'] - lat)**2 + 
             (past_events['longitude'] - lon)**2)**0.5 < 2.0  # ~200 км
        ]
        
        # b-value (упрощённо)
        mags = events_30d['magnitude'].values if not events_30d.empty else np.array([3.0])
        if len(mags) > 5:
            # log10(N) = a - b*M => b = slope
            hist, bins = np.histogram(mags, bins=10)
            log_n = np.log10(hist[hist > 0])
            bin_centers = (bins[:-1] + bins[1:])[hist > 0] / 2
            if len(log_n) > 1:
                b_value = -np.polyfit(bin_centers, log_n, 1)[0]
                b_value = max(0.3, min(2.0, b_value))
            else:
                b_value = 1.0
        else:
            b_value = 1.0
        
        return {
            'seismic_rate_30d': len(events_30d),
            'seismic_rate_7d': len(events_7d),
            'b_value': b_value,
            'max_mag_30d': events_30d['magnitude'].max() if not events_30d.empty else 0,
            'mean_mag_30d': events_30d['magnitude'].mean() if not events_30d.empty else 0,
            'depth_mean': events_30d['depth_km'].mean() if not events_30d.empty and 'depth_km' in events_30d.columns else 10,
            'events_count_near': len(near_events),
        }
    
    @staticmethod
    def _encode_level(level: str) -> int:
        """Кодирование уровня аномалии."""
        levels = {'normal': 0, 'low': 1, 'moderate': 2, 'high': 3, 'critical': 4}
        return levels.get(level, 0)
    
    @staticmethod
    def _compute_trend(df: pd.DataFrame) -> float:
        """Вычисление тренда временного ряда."""
        if len(df) < 3:
            return 0.0
        x = np.arange(len(df))
        y = df.iloc[:, 0].values  # Первая колонка значений
        return np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════
# МОДЕЛЬ ПРОГНОЗИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════

class EarthquakePredictor:
    """
    Мультимодельная система прогнозирования землетрясений.
    """
    
    FEATURE_NAMES = [
        'seismic_rate_30d', 'seismic_rate_7d', 'b_value', 'max_mag_30d',
        'mean_mag_30d', 'depth_mean', 'events_count_near',
        'tec_anomaly_max', 'tec_anomaly_min', 'iono_anomaly_level',
        'iono_historical_years', 'iono_yoy_change',
        'lst_available', 'lst_value', 'lst_source_real',
        'dst_mean', 'dst_std', 'dst_min', 'kp_max', 'kp_mean',
        'f107_mean', 'f107_trend',
        'day_of_year', 'hour', 'month',
        'lat', 'lon', 'abs_lat', 'is_northern',
        'temp_mean', 'temp_range', 'humidity_mean'
    ]
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.probability_model = None
        self.magnitude_model = None
        self.time_model = None
        self.lat_model = None
        self.lon_model = None
        self.scaler = StandardScaler()
        
        self.is_trained = False
        
        # Пытаемся загрузить существующие модели
        self._load_models()
    
    def _load_models(self):
        """Загрузка сохранённых моделей."""
        try:
            self.probability_model = joblib.load(f"{self.model_dir}/probability_model.pkl")
            self.magnitude_model = joblib.load(f"{self.model_dir}/magnitude_model.pkl")
            self.time_model = joblib.load(f"{self.model_dir}/time_model.pkl")
            self.lat_model = joblib.load(f"{self.model_dir}/lat_model.pkl")
            self.lon_model = joblib.load(f"{self.model_dir}/lon_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.is_trained = True
            logger.info("✅ Модели загружены из кэша")
        except FileNotFoundError:
            logger.info("ℹ️ Сохранённых моделей не найдено, требуется обучение")
    
    def _save_models(self):
        """Сохранение моделей."""
        joblib.dump(self.probability_model, f"{self.model_dir}/probability_model.pkl")
        joblib.dump(self.magnitude_model, f"{self.model_dir}/magnitude_model.pkl")
        joblib.dump(self.time_model, f"{self.model_dir}/time_model.pkl")
        joblib.dump(self.lat_model, f"{self.model_dir}/lat_model.pkl")
        joblib.dump(self.lon_model, f"{self.model_dir}/lon_model.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
        logger.info("💾 Модели сохранены")
    
    def prepare_training_data(
        self,
        events_df: pd.DataFrame,
        region_data_dict: Dict[str, pd.DataFrame],
        lst_cache: Dict,
        iono_cache: Dict,
        space_cache: Dict,
        weather_cache: Dict
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Подготовка обучающих данных.
        
        Для каждого события M≥5.0 создаём записи за разные дни до события.
        """
        X = []
        y_prob = []
        y_mag = []
        y_time = []
        y_lat = []
        y_lon = []
        
        # Фильтруем значимые события
        significant_events = events_df[events_df['magnitude'] >= 5.0].copy()
        
        logger.info(f"📊 Подготовка данных: {len(significant_events)} событий")
        
        for _, event in significant_events.iterrows():
            event_time = event['time']
            lat = event['latitude']
            lon = event['longitude']
            
            # Определяем регион
            region_name = self._get_region_for_point(lat, lon, region_data_dict)
            region_df = region_data_dict.get(region_name, pd.DataFrame())
            
            # Для разных моментов до события
            for days_before in CONFIG.DAYS_BEFORE_FEATURES:
                sample_time = event_time - timedelta(days=days_before)
                
                # Пропускаем если время в будущем
                if sample_time > datetime.now(timezone.utc):
                    continue
                
                # Извлекаем признаки
                features = LAICFeatureExtractor.extract_features(
                    lat=lat,
                    lon=lon,
                    event_time=sample_time,
                    region_data=region_df,
                    lst_data=lst_cache.get(region_name),
                    iono_data=iono_cache.get(event.get('id')),
                    space_data=space_cache.get(event.get('id')),
                    weather_data=weather_cache.get(event.get('id'))
                )
                
                X.append(features)
                
                # Целевые переменные
                y_prob.append(1 if event['magnitude'] >= CONFIG.MAGNITUDE_THRESHOLD else 0)
                y_mag.append(event['magnitude'])
                y_time.append(days_before)
                y_lat.append(lat)
                y_lon.append(lon)
        
        X = np.array(X)
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, {
            'probability': np.array(y_prob),
            'magnitude': np.array(y_mag),
            'time': np.array(y_time),
            'lat': np.array(y_lat),
            'lon': np.array(y_lon)
        }
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """
        Обучение моделей.
        """
        if len(X) < 20:
            logger.warning("⚠️ Слишком мало данных для обучения (< 20 образцов)")
            return False
        
        logger.info(f"🧠 Обучение на {len(X)} образцах...")
        
        # Разделение на train/test
        X_train, X_test, y_prob_train, y_prob_test = train_test_split(
            X, y['probability'], test_size=0.2, random_state=42, stratify=y['probability']
        )
        
        # Модель 1: Вероятность M≥threshold
        logger.info("  📊 Обучение модели вероятности...")
        self.probability_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.probability_model.fit(X, y['probability'])
        
        prob_score = self.probability_model.score(X_test, y_prob_test)
        logger.info(f"  ✅ Вероятность: accuracy={prob_score:.3f}")
        
        # Модель 2: Магнитуда
        logger.info("  📊 Обучение модели магнитуды...")
        self.magnitude_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.magnitude_model.fit(X, y['magnitude'])
        
        mag_pred = self.magnitude_model.predict(X_test)
        mag_mae = mean_absolute_error(y['magnitude'][:len(mag_pred)], mag_pred)
        logger.info(f"  ✅ Магнитуда: MAE={mag_mae:.2f}")
        
        # Модель 3: Время до события
        logger.info("  📊 Обучение модели времени...")
        self.time_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        )
        self.time_model.fit(X, y['time'])
        
        time_pred = self.time_model.predict(X_test)
        time_mae = mean_absolute_error(y['time'][:len(time_pred)], time_pred)
        logger.info(f"  ✅ Время: MAE={time_mae:.1f} дней")
        
        # Модель 4: Широта
        logger.info("  📊 Обучение модели широты...")
        self.lat_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            random_state=42
        )
        self.lat_model.fit(X, y['lat'])
        
        # Модель 5: Долгота
        logger.info("  📊 Обучение модели долготы...")
        self.lon_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            random_state=42
        )
        self.lon_model.fit(X, y['lon'])
        
        self.is_trained = True
        self._save_models()
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.FEATURE_NAMES,
            'importance': self.probability_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("  📈 Top-10 важных признаков:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"     {row['feature']}: {row['importance']:.3f}")
        
        return True
    
    def predict(
        self,
        lat: float,
        lon: float,
        current_time: datetime,
        region_df: pd.DataFrame = None,
        lst_data: Dict = None,
        iono_data: Dict = None,
        space_data: Dict = None
    ) -> Dict:
        """
        Прогноз для точки.
        """
        if not self.is_trained:
            logger.warning("⚠️ Модели не обучены!")
            return self._empty_prediction()
        
        # Извлекаем признаки
        features = LAICFeatureExtractor.extract_features(
            lat=lat,
            lon=lon,
            event_time=current_time,
            region_data=region_df or pd.DataFrame(),
            lst_data=lst_data,
            iono_data=iono_data,
            space_data=space_data
        )
        
        X = self.scaler.transform(features.reshape(1, -1))
        
        # Предсказания
        prob = self.probability_model.predict_proba(X)[0][1]
        mag = self.magnitude_model.predict(X)[0]
        days = self.time_model.predict(X)[0]
        pred_lat = self.lat_model.predict(X)[0]
        pred_lon = self.lon_model.predict(X)[0]
        
        # Уверенность
        confidence = self._get_confidence(prob, days)
        
        # Радиус ошибки
        radius = self._estimate_radius(prob)
        
        return {
            'input_lat': lat,
            'input_lon': lon,
            'predicted_lat': round(float(pred_lat), 2),
            'predicted_lon': round(float(pred_lon), 2),
            'probability_m6': round(float(prob), 3),
            'predicted_magnitude': round(float(mag), 1),
            'days_to_event': round(float(days), 1),
            'confidence': confidence,
            'radius_km': radius,
            'is_alert': prob >= CONF
