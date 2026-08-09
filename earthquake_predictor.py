#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
earthquake_predictor.py — ИИ-прогнозирование землетрясений
Исправленная версия с правильной семантикой времени
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

logging.getLogger("sklearn").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PredictionConfig:
    MAGNITUDE_THRESHOLD: float = 6.0
    GRID_RESOLUTION: float = 2.0
    PROBABILITY_THRESHOLD: float = 0.3
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6
    DAYS_BEFORE_FEATURES: List[int] = None
    FORECAST_HORIZON_DAYS: int = 30
    MATCH_SCORE_THRESHOLD: float = 0.6
    
    def __post_init__(self):
        if self.DAYS_BEFORE_FEATURES is None:
            self.DAYS_BEFORE_FEATURES = [30, 21, 14, 7, 3, 1]


CONFIG = PredictionConfig()


# ═══════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ═══════════════════════════════════════════════════════════════

class LAICFeatureExtractor:
    """Извлечение признаков из LAIC-данных."""
    
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
        'temp_mean', 'temp_range', 'humidity_mean',
        'data_freshness_days'  # ← НОВОЕ: насколько свежие данные
    ]
    
    @staticmethod
    def extract_features(
        lat: float,
        lon: float,
        reference_time: datetime,  # ← переименовано: время, на котором смотрим признаки
        region_data: pd.DataFrame,
        lst_data: Optional[Dict] = None,
        iono_data: Optional[Dict] = None,
        space_data: Optional[Dict] = None,
        weather_data: Optional[pd.DataFrame] = None,
        data_end_time: Optional[datetime] = None  # ← когда заканчиваются реальные данные
    ) -> np.ndarray:
        """Извлечение вектора признаков на момент reference_time."""
        features = {}
        
        # Важно: если reference_time в будущем от данных — помечаем
        data_freshness = 0
        if data_end_time and reference_time > data_end_time:
            data_freshness = (reference_time - data_end_time).days
        
        # 1. Сейсмические признаки — только ДО reference_time
        seismic_features = LAICFeatureExtractor._get_seismic_features(
            lat, lon, reference_time, region_data
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
            'day_of_year': reference_time.timetuple().tm_yday / 365.0,
            'hour': reference_time.hour / 24.0,
            'month': reference_time.month / 12.0,
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
        
        # 8. Свежесть данных
        features['data_freshness_days'] = data_freshness
        
        return np.array([features.get(name, np.nan) for name in LAICFeatureExtractor.FEATURE_NAMES])
    
    @staticmethod
    def _get_seismic_features(
        lat: float,
        lon: float,
        reference_time: datetime,  # ← ИСПРАВЛЕНО: не event_time, а время среза
        region_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Сейсмические признаки — только события ДО reference_time."""
        if region_data.empty:
            return {
                'seismic_rate_30d': 0, 'seismic_rate_7d': 0,
                'b_value': 1.0, 'max_mag_30d': 0,
                'mean_mag_30d': 0, 'depth_mean': 10,
                'events_count_near': 0,
            }
        
        # ← ИСПРАВЛЕНО: строго < reference_time, не <=
        past_events = region_data[region_data['time'] < reference_time]
        
        # Если нет данных — возвращаем "тишину"
        if past_events.empty:
            return {
                'seismic_rate_30d': 0, 'seismic_rate_7d': 0,
                'b_value': 1.0, 'max_mag_30d': 0,
                'mean_mag_30d': 0, 'depth_mean': 10,
                'events_count_near': 0,
            }
        
        days_30 = reference_time - timedelta(days=30)
        days_7 = reference_time - timedelta(days=7)
        
        events_30d = past_events[past_events['time'] >= days_30]
        events_7d = past_events[past_events['time'] >= days_7]
        
        # Ближайшие события в радиусе 2 градуса
        near_events = past_events[
            ((past_events['latitude'] - lat)**2 + 
             (past_events['longitude'] - lon)**2)**0.5 < 2.0
        ]
        
        # b-value по закону Гутенберга-Рихтера
        mags = events_30d['magnitude'].values if not events_30d.empty else np.array([])
        
        b_value = 1.0  # дефолт
        if len(mags) > 5:
            # Логарифмическая гистограмма
            mag_min, mag_max = mags.min(), mags.max()
            if mag_max > mag_min:
                bins = np.linspace(mag_min, mag_max, min(10, len(mags)))
                hist, bin_edges = np.histogram(mags, bins=bins)
                # Только bins с событиями
                mask = hist > 0
                if mask.sum() > 1:
                    log_n = np.log10(hist[mask])
                    bin_centers = (bin_edges[:-1][mask] + bin_edges[1:][mask]) / 2
                    # Линейная регрессия log10(N) vs M
                    coeffs = np.polyfit(bin_centers, log_n, 1)
                    b_value = max(0.3, min(2.0, -coeffs[0]))
        
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
        levels = {'normal': 0, 'low': 1, 'moderate': 2, 'high': 3, 'critical': 4}
        return levels.get(level, 0)
    
    @staticmethod
    def _compute_trend(df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['time', 'timestamp']]
        
        if len(numeric_cols) == 0:
            return 0.0
        
        y = df[numeric_cols[0]].values
        if len(set(y)) <= 1:
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
    """
    Мультимодельная система прогнозирования.
    
    Семантика:
    - Обучение: "при таких признаках за days_before дней до события, 
                 было ли событие магнитудой ≥ threshold?"
    - Прогноз: "если сейчас такие признаки, через сколько дней ожидать событие?"
    """
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        
        self.probability_model = None   # P(M ≥ threshold | features)
        self.magnitude_model = None     # E[magnitude | features, M ≥ threshold]
        self.days_before_model = None   # E[days_before | features] — для matching
        
        self.is_trained = False
        
        self._load_models()
    
    def _load_models(self):
        try:
            self.probability_model = joblib.load(f"{self.model_dir}/prob_model.pkl")
            self.magnitude_model = joblib.load(f"{self.model_dir}/mag_model.pkl")
            self.days_before_model = joblib.load(f"{self.model_dir}/days_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.is_trained = True
            logger.info("✅ Модели загружены из кэша")
        except FileNotFoundError:
            logger.info("ℹ️ Сохранённых моделей не найдено, требуется обучение")
    
    def _save_models(self):
        joblib.dump(self.probability_model, f"{self.model_dir}/prob_model.pkl")
        joblib.dump(self.magnitude_model, f"{self.model_dir}/mag_model.pkl")
        joblib.dump(self.days_before_model, f"{self.model_dir}/days_model.pkl")
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
        y_days = []  # ← days_before, не days_to_event
    
        significant_events = events_df[events_df['magnitude'] >= 4.5].copy()
        logger.info(f"📊 Подготовка данных: {len(significant_events)} событий M≥4.5")
    
        # Определяем конец данных для freshness
        data_end_time = events_df['time'].max() if not events_df.empty else datetime.now(timezone.utc)
    
        try:
            from space_weather import get_collector
            from ionosphere_collector import IonosphereCollector
            space_collector = get_collector()
            iono_collector = IonosphereCollector()
            collectors_available = True
        except ImportError:
            collectors_available = False
            logger.warning("⚠️ Коллекторы недоступны, используем кэш")
    
        for _, event in significant_events.iterrows():
            event_time = event['time']
            lat = event['latitude']
            lon = event['longitude']
        
            region_name = self._get_region_for_point(lat, lon, region_data_dict)
            region_df = region_data_dict.get(region_name, pd.DataFrame())
        
            for days_before in CONFIG.DAYS_BEFORE_FEATURES:
                sample_time = event_time - timedelta(days=days_before)
            
                # Не используем будущее относительно сейчас (для чистоты)
                if sample_time > datetime.now(timezone.utc) + timedelta(days=1):
                    continue
            
                # Данные на момент sample_time
                space_data_for_sample = space_cache.get(event.get('id')) if space_cache else None
                iono_data_for_sample = iono_cache.get(event.get('id')) if iono_cache else None
            
                # Если нет в кэше — собираем (только для недавних)
                if collectors_available and not space_data_for_sample:
                    try:
                        start_space = sample_time - timedelta(days=10)
                        space_data_for_sample = space_collector.fetch_all_for_period(start_space, sample_time)
                    except Exception:
                        pass
                
                if collectors_available and not iono_data_for_sample:
                    try:
                        iono_data_for_sample = iono_collector.fetch_for_event(
                            event_time=sample_time,
                            lat=lat, lon=lon,
                            days_before=7, days_after=0
                        )
                    except Exception:
                        pass
            
                try:
                    features = LAICFeatureExtractor.extract_features(
                        lat=lat,
                        lon=lon,
                        reference_time=sample_time,  # ← ИСПРАВЛЕНО
                        region_data=region_df,
                        lst_data=lst_cache.get(region_name),
                        iono_data=iono_data_for_sample,
                        space_data=space_data_for_sample,
                        weather_data=weather_cache.get(event.get('id')),
                        data_end_time=data_end_time
                    )
                
                    X.append(features)
                    # Цели:
                    y_prob.append(1 if event['magnitude'] >= CONFIG.MAGNITUDE_THRESHOLD else 0)
                    y_mag.append(event['magnitude'])
                    y_days.append(days_before)  # ← сколько дней ДО события
                
                except Exception as e:
                    logger.debug(f"Пропуск точки: {e}")
                    continue
    
        if len(X) == 0:
            logger.warning("⚠️ Нет данных для обучения")
            return np.array([]), {}
    
        X = np.array(X)
        logger.info(f"✅ Собрано {len(X)} обучающих примеров")
        
        return X, {
            'probability': np.array(y_prob),
            'magnitude': np.array(y_mag),
            'days_before': np.array(y_days),  # ← переименовано
        }
    
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> bool:
        """Обучение моделей."""
        if len(X) < 20:
            logger.warning("⚠️ Слишком мало данных для обучения (< 20 образцов)")
            return False
        
        logger.info(f"🧠 Обучение на {len(X)} образцах...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Модель 1: Вероятность сильного землетрясения
        self.probability_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,  # ← уменьшено против переобучения
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            class_weight='balanced' if sum(y['probability']) / len(y['probability']) < 0.2 else None
        )
        self.probability_model.fit(X_scaled, y['probability'])
        prob_score = self.probability_model.score(X_scaled, y['probability'])
        logger.info(f"  📊 Accuracy (probability): {prob_score:.3f}")
        
        # Модель 2: Магнитуда (только для событий ≥ threshold)
        mask_strong = y['probability'] == 1
        if mask_strong.sum() >= 10:
            self.magnitude_model = HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=4,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15
            )
            self.magnitude_model.fit(X_scaled[mask_strong], y['magnitude'][mask_strong])
            mag_score = self.magnitude_model.score(X_scaled[mask_strong], y['magnitude'][mask_strong])
            logger.info(f"  📊 R² (magnitude): {mag_score:.3f}")
        else:
            logger.warning("⚠️ Мало сильных событий для обучения магнитуды")
            self.magnitude_model = None
        
        # Модель 3: days_before (для matching паттернов)
        self.days_before_model = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=5,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15
        )
        self.days_before_model.fit(X_scaled, y['days_before'])
        days
