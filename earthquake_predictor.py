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
from sklearn.metrics import mean_absolute_error
import joblib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PredictionConfig:
    """Конфигурация прогнозирования."""
    MAGNITUDE_THRESHOLD: float = 6.0
    GRID_RESOLUTION: float = 2.0
    PROBABILITY_THRESHOLD: float = 0.3
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6
    DAYS_BEFORE_FEATURES: List[int] = None
    
    def __post_init__(self):
        if self.DAYS_BEFORE_FEATURES is None:
            self.DAYS_BEFORE_FEATURES = [30, 21, 14, 7, 3, 0]


CONFIG = PredictionConfig()


# ═══════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
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
        """Извлечение вектора признаков."""
        features = {}
        
        # 1. Сейсмические признаки
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
                'tec_anomaly_max': 0, 'tec_anomaly_min': 0,
                'iono_anomaly_level': 0,
                'iono_historical_years': 0, 'iono_yoy_change': 0,
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
                'lst_available': 0, 'lst_value': 20, 'lst_source_real': 0,
            })
        
        # 4. Космическая погода
        if space_data:
            dst_df = space_data.get('dst', pd.DataFrame())
            kp_df = space_data.get('kp', pd.DataFrame())
            f107_df = space_data.get('f107', pd.DataFrame())
            
            features.update({
                'dst_mean': dst_df['dst'].mean() if not dst_df.empty and 'dst' in dst_df.columns else -10,
                'dst_std': dst_df['dst'].std() if not dst_df.empty and 'dst' in dst_df.columns else 5,
                'dst_min': dst_df['dst'].min() if not dst_df.empty and 'dst' in dst_df.columns else -50,
                'kp_max': kp_df['kp'].max() if not kp_df.empty and 'kp' in kp_df.columns else 3,
                'kp_mean': kp_df['kp'].mean() if not kp_df.empty and 'kp' in kp_df.columns else 1.5,
                'f107_mean': f107_df['f107'].mean() if not f107_df.empty and 'f107' in f107_df.columns else 140,
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
            temp_col = weather_data.get('temp_mean', weather_data.get('temperature', pd.Series([20])))
            temp_max = weather_data.get('temp_max', pd.Series([25])).max() if 'temp_max' in weather_data.columns else 25
            temp_min = weather_data.get('temp_min', pd.Series([15])).min() if 'temp_min' in weather_data.columns else 15
            humidity = weather_data.get('humidity', pd.Series([60])).mean() if 'humidity' in weather_data.columns else 60
            
            features.update({
                'temp_mean': temp_col.mean() if hasattr(temp_col, 'mean') else 20,
                'temp_range': temp_max - temp_min,
                'humidity_mean': humidity,
            })
        else:
            features.update({
                'temp_mean': 20, 'temp_range': 10, 'humidity_mean': 60,
            })
        
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
                'seismic_rate_30d': 0, 'seismic_rate_7d': 0,
                'b_value': 1.0, 'max_mag_30d': 0,
                'mean_mag_30d': 0, 'depth_mean': 10,
                'events_count_near': 0,
            }
        
        past_events = region_data[region_data['time'] < event_time]
        
        days_30 = event_time - timedelta(days=30)
        days_7 = event_time - timedelta(days=7)
        
        events_30d = past_events[past_events['time'] >= days_30]
        events_7d = past_events[past_events['time'] >= days_7]
        
        near_events = past_events[
            ((past_events['latitude'] - lat)**2 + 
             (past_events['longitude'] - lon)**2)**0.5 < 2.0
        ]
        
        mags = events_30d['magnitude'].values if not events_30d.empty else np.array([3.0])
        if len(mags) > 5:
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
        """
        Вычисление тренда временного ряда.
        ✅ ИСПРАВЛЕНО: берём числовую колонку, не даты!
        """
        if len(df) < 3:
            return 0.0
        
        # Ищем числовые колонки (исключаем time/datetime)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Убираем 'time' если он случайно числовой
        numeric_cols = [c for c in numeric_cols if c not in ['time', 'timestamp']]
        
        if len(numeric_cols) == 0:
            return 0.0
        
        # Берём первую числовую колонку (f107, dst, kp)
        y = df[numeric_cols[0]].values
        
        if len(set(y)) <= 1:  # Все значения одинаковые
            return 0.0
        
        x = np.arange(len(y))
        try:
            return float(np.polyfit(x, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0


# ═══════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС ПРОГНОЗИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════

class EarthquakePredictor:
    """Мультимодельная система прогнозирования землетрясений."""
    
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
        """Подготовка обучающих данных."""
        X = []
        y_prob = []
        y_mag = []
        y_time = []
        y_lat = []
        y_lon = []
        
        significant_events = events_df[events_df['magnitude'] >= 5.0].copy()
        
        logger.info(f"📊 Подготовка данных: {len(significant_events)} событий")
        
        for _, event in significant_events.iterrows():
            event_time = event['time']
            lat = event['latitude']
            lon = event['longitude']
            
            region_name = self._get_region_for_point(lat, lon, region_data_dict)
            region_df = region_data_dict.get(region_name, pd.DataFrame())
            
            for days_before in CONFIG.DAYS_BEFORE_FEATURES:
                sample_time = event_time - timedelta(days=days_before)
                
                if sample_time > datetime.now(timezone.utc):
                    continue
                
                try:
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
                    y_prob.append(1 if event['magnitude'] >= CONFIG.MAGNITUDE_THRESHOLD else 0)
                    y_mag.append(event['magnitude'])
                    y_time.append(days_before)
                    y_lat.append(lat)
                    y_lon.append(lon)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка извлечения признаков для {event.get('id')}: {e}")
                    continue
        
        if len(X) == 0:
            logger.warning("⚠️ Нет данных для обучения")
            return np.array([]), {}
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, {
            'probability': np.array(y_prob),
            'magnitude': np.array(y_mag),
            'time': np.array(y_time),
            'lat': np.array(y_lat),
            'lon': np.array(y_lon)
        }
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> bool:
        """Обучение моделей."""
        if len(X) < 20:
            logger.warning("⚠️ Слишком мало данных для обучения (< 20 образцов)")
            return False
        
        logger.info(f"🧠 Обучение на {len(X)} образцах...")
        
        # Модель 1: Вероятность
        logger.info("  📊 Обучение модели вероятности...")
        self.probability_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.probability_model.fit(X, y['probability'])
        
        # Модель 2: Магнитуда
        logger.info("  📊 Обучение модели магнитуды...")
        self.magnitude_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.magnitude_model.fit(X, y['magnitude'])
        
        # Модель 3: Время
        logger.info("  📊 Обучение модели времени...")
        self.time_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        )
        self.time_model.fit(X, y['time'])
        
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
        
        # Важность признаков
        importance = pd.DataFrame({
            'feature': self.FEATURE_NAMES,
            'importance': self.probability_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("  📈 Топ-10 важных признаков:")
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
        """Прогноз для точки."""
        if not self.is_trained:
            logger.warning("⚠️ Модели не обучены!")
            return self._empty_prediction()
        
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
        
        prob = self.probability_model.predict_proba(X)[0][1]
        mag = self.magnitude_model.predict(X)[0]
        days = self.time_model.predict(X)[0]
        pred_lat = self.lat_model.predict(X)[0]
        pred_lon = self.lon_model.predict(X)[0]
        
        confidence = self._get_confidence(prob, days)
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
            'is_alert': prob >= CONFIG.HIGH_CONFIDENCE_THRESHOLD and confidence == 'HIGH'
        }
    
    def predict_grid(
        self,
        region_bounds: Dict[str, float],
        current_time: datetime,
        region_data_dict: Dict[str, pd.DataFrame],
        lst_cache: Dict = None,
        iono_cache: Dict = None,
        space_cache: Dict = None,
        resolution: float = None
    ) -> pd.DataFrame:
        """Прогноз по сетке для региона."""
        if not self.is_trained:
            logger.warning("⚠️ Модели не обучены!")
            return pd.DataFrame()
        
        resolution = resolution or CONFIG.GRID_RESOLUTION
        
        lats = np.arange(region_bounds['lat_min'], region_bounds['lat_max'], resolution)
        lons = np.arange(region_bounds['lon_min'], region_bounds['lon_max'], resolution)
        
        predictions = []
        total = len(lats) * len(lons)
        logger.info(f"🔮 Прогноз по сетке: {len(lats)}×{len(lons)} = {total} точек")
        
        for i, lat in enumerate(lats):
            for lon in lons:
                region_name = self._get_region_for_point(lat, lon, region_data_dict)
                region_df = region_data_dict.get(region_name, pd.DataFrame())
                
                pred = self.predict(
                    lat=lat,
                    lon=lon,
                    current_time=current_time,
                    region_df=region_df,
                    lst_data=lst_cache.get(region_name) if lst_cache else None,
                    iono_data=None,
                    space_data=None
                )
                
                if pred['probability_m6'] >= CONFIG.PROBABILITY_THRESHOLD:
                    predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Обработано: {(i+1) * len(lons)}/{total}")
        
        df = pd.DataFrame(predictions)
        if not df.empty:
            df = df.sort_values('probability_m6', ascending=False)
        
        logger.info(f"✅ Прогноз готов: {len(df)} точек с вероятностью ≥{CONFIG.PROBABILITY_THRESHOLD}")
        return df
    
    def _get_region_for_point(self, lat: float, lon: float, region_data_dict: Dict) -> str:
        """Определение ближайшего региона."""
        if not region_data_dict:
            return 'global'
        
        best_region = 'global'
        min_dist = float('inf')
        
        for name, df in region_data_dict.items():
            if not df.empty and 'latitude' in df.columns:
                mean_lat = df['latitude'].mean()
                mean_lon = df['longitude'].mean()
                dist = ((lat - mean_lat)**2 + (lon - mean_lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_region = name
        
        return best_region
    
    def _get_confidence(self, prob: float, days: float) -> str:
        """Оценка уверенности."""
        if prob > 0.7 and days < 14:
            return 'HIGH'
        elif prob > 0.4 and days < 21:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_radius(self, prob: float) -> int:
        """Радиус ошибки в км."""
        if prob > 0.7:
            return 100
        elif prob > 0.4:
            return 200
        else:
            return 500
    
    def _empty_prediction(self) -> Dict:
        """Пустой прогноз."""
        return {
            'input_lat': 0.0,
            'input_lon': 0.0,
            'predicted_lat': 0.0,
            'predicted_lon': 0.0,
            'probability_m6': 0.0,
            'predicted_magnitude': 0.0,
            'days_to_event': 0.0,
            'confidence': 'NONE',
            'radius_km': 0,
            'is_alert': False
        }


# ═══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def create_risk_report(predictions_df: pd.DataFrame, top_n: int = 5) -> str:
    """Создание текстового отчёта о рисках."""
    if predictions_df.empty:
        return "🔍 Значимых рисков не обнаружено."
    
    report = ["🌍 <b>ПРОГНОЗ ЗЕМЛЕТРЯСЕНИЙ</b>\n"]
    report.append(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    
    for i, (_, row) in enumerate(predictions_df.head(top_n).iterrows(), 1):
        emoji = "🔴" if row['is_alert'] else "🟠" if row['confidence'] == 'MEDIUM' else "🟡"
        
        report.append(f"\n{emoji} <b>Зона риска #{i}</b>")
        report.append(f"📍 Координаты: {row['predicted_lat']}°, {row['predicted_lon']}°")
        report.append(f"📏 Радиус: ±{row['radius_km']} км")
        report.append(f"⏰ Прогноз: через {row['days_to_event']:.0f} дней")
        report.append(f"📊 Вероятность M≥6.0: <b>{row['probability_m6']*100:.1f}%</b>")
        report.append(f"🎯 Магнитуда: M{row['predicted_magnitude']}")
        report.append(f"🔒 Уверенность: {row['confidence']}")
    
    report.append(f"\n⚠️ Прогноз основан на LAIC-анализе и ИИ-модели")
    report.append(f"📊 Всего зон с риском: {len(predictions_df)}")
    
    return "\n".join(report)


# ═══════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_events = pd.DataFrame({
        'id': ['evt1', 'evt2', 'evt3'],
        'time': [
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 15, tzinfo=timezone.utc),
            datetime(2026, 7, 28, tzinfo=timezone.utc),
        ],
        'latitude': [35.0, -20.0, 40.0],
        'longitude': [140.0, -70.0, 35.0],
        'magnitude': [6.5, 5.2, 7.0],
        'depth_km': [10, 20, 15]
    })
    
    predictor = EarthquakePredictor(model_dir="data/models_test")
    
    X, y = predictor.prepare_training_data(
        events_df=test_events,
        region_data_dict={'global': test_events},
        lst_cache={},
        iono_cache={},
        space_cache={},
        weather_cache={}
    )
    
    if len(X) > 0:
        predictor.train(X, y)
        
        pred = predictor.predict(35.0, 140.0, datetime.now(timezone.utc))
        print(f"\nПрогноз: {pred}")
