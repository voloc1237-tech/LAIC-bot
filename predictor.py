#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py — LSTM-прогнозирование землетрясений (исправленная версия)
Предсказывает вероятность события в окнах: [1-3д, 4-7д, 8-14д, 15-30д, 30+д, нет]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Отключаем шум TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class EarthquakeLSTMPredictor:
    """
    LSTM для временного прогноза землетрясений.
    
    Вход: временной ряд признаков (без магнитуды!)
    Выход: вероятность события M≥threshold в окнах:
        - 0: нет события (30+ дней тишины)
        - 1: 1-3 дня
        - 2: 4-7 дней  
        - 3: 8-14 дней
        - 4: 15-30 дней
    """
    
    # Окна прогноза (дни)
    HORIZON_WINDOWS = [
        (1, 3),    # класс 1
        (4, 7),    # класс 2
        (8, 14),   # класс 3
        (15, 30),  # класс 4
    ]
    NO_EVENT_CLASS = 0
    N_CLASSES = 5
    
    def __init__(
        self,
        model_dir: str = "data/models",
        sequence_length: int = 14,  # ← увеличено с 7
        forecast_horizon: int = 30,  # ← сколько дней вперёд смотрим
        magnitude_threshold: float = 6.0
    ):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.magnitude_threshold = magnitude_threshold
        
        self.model = None
        self.scaler = StandardScaler()
        
        # Признаки (БЕЗ magnitude и energy!)
        self.feature_names = [
            'kp_mean', 'kp_max', 'kp_std',
            'dst_mean', 'dst_min',
            'f107_mean',
            'temp_mean', 'temp_range',
            'humidity_mean',
            'seismic_rate_7d', 'seismic_rate_30d',
            'max_mag_7d',  # ← макс магнитуда за 7 дней, не текущая!
            'b_value',
            'depth_mean',
            'iono_anomaly_score',  # ← 0-4 нормализованный уровень
            'iono_z_score_max',
            'lst_available',
            'day_of_year_sin', 'day_of_year_cos',  # ← циклическое время
        ]
        
        self._load_model()
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Строит LSTM для многоклассовой классификации."""
        model = Sequential([
            Input(shape=input_shape),
            
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            LSTM(32, return_sequences=False, dropout=0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            
            Dense(self.N_CLASSES, activation='softmax')  # ← многоклассовая!
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec')]
        )
        
        return model
    
    def _create_sequences(
        self,
        df: pd.DataFrame,
        event_times: List[datetime]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создаёт последовательности для обучения.
        
        Для каждой точки времени:
        - X: признаки за sequence_length дней ДО неё
        - y: класс окна, в котором произошло следующее событие (или 0)
        """
        # Сортируем по времени
        df = df.sort_values('time').reset_index(drop=True)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(df) - self.forecast_horizon):
            # Время "сейчас"
            current_time = df.loc[i, 'time']
            
            # Следующие события в горизонте
            future_mask = (
                (df['time'] > current_time) & 
                (df['time'] <= current_time + timedelta(days=self.forecast_horizon))
            )
            future_events = df[future_mask]
            
            # Фильтруем сильные события
            strong_events = future_events[future_events['magnitude'] >= self.magnitude_threshold]
            
            # Определяем класс
            if strong_events.empty:
                y_class = self.NO_EVENT_CLASS
            else:
                # Ближайшее событие
                next_event = strong_events.loc[strong_events['time'].idxmin()]
                days_to_event = (next_event['time'] - current_time).days
                
                y_class = self._days_to_class(days_to_event)
            
            # Признаки: sequence_length дней до current_time
            seq_data = df.loc[i - self.sequence_length : i - 1, self.feature_names]
            
            # Проверяем полноту
            if len(seq_data) < self.sequence_length or seq_data.isnull().any().any():
                continue
            
            X.append(seq_data.values)
            y.append(y_class)
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # One-hot encoding для y
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=self.N_CLASSES)
        
        return np.array(X), y_onehot
    
    def _days_to_class(self, days: int) -> int:
        """Конвертирует дни до события в класс окна."""
        for i, (start, end) in enumerate(self.HORIZON_WINDOWS, 1):
            if start <= days <= end:
                return i
        return self.NO_EVENT_CLASS  # > 30 или < 1
    
    def _class_to_days(self, class_idx: int) -> Tuple[int, int]:
        """Конвертирует класс в диапазон дней."""
        if class_idx == self.NO_EVENT_CLASS:
            return (31, 90)  # "далеко/никогда"
        return self.HORIZON_WINDOWS[class_idx - 1]
    
    def prepare_data(
        self,
        events_df: pd.DataFrame,
        region_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        daily_features_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Подготавливает ежедневные признаки из событий.
        
        Если daily_features_df уже есть — использует его.
        Иначе строит из events_df.
        """
        if daily_features_df is not None:
            return daily_features_df
        
        # Строим ежедневные агрегаты
        events_df = events_df.copy()
        events_df['time'] = pd.to_datetime(events_df['time'])
        events_df['date'] = events_df['time'].dt.floor('D')
        
        # Генерируем все даты
        date_range = pd.date_range(
            start=events_df['date'].min(),
            end=events_df['date'].max(),
            freq='D'
        )
        
        daily = pd.DataFrame({'date': date_range})
        daily['time'] = daily['date']  # для совместимости
        
        # Агрегаты по дням
        daily_stats = events_df.groupby('date').agg({
            'magnitude': ['count', 'max', 'mean', 'std'],
            'depth_km': 'mean',
        }).reset_index()
        daily_stats.columns = ['date', 'event_count', 'max_mag', 'mean_mag', 'mag_std', 'depth_mean']
        
        # Скользящие окна (сейсмичность)
        daily = daily.merge(daily_stats, on='date', how='left')
        daily = daily.fillna(0)
        
        # Скользящие признаки (важно: без заглядывания вперёд!)
        daily['seismic_rate_7d'] = daily['event_count'].rolling(7, min_periods=1).sum().shift(1)
        daily['seismic_rate_30d'] = daily['event_count'].rolling(30, min_periods=1).sum().shift(1)
        daily['max_mag_7d'] = daily['max_mag'].rolling(7, min_periods=1).max().shift(1)
        
        # b-value (упрощённо)
        daily['b_value'] = 1.0  # можно улучшить
        
        # Временные циклические признаки
        day_of_year = daily['date'].dt.dayofyear
        daily['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        daily['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Заглушки для остальных признаков (в реальности — из внешних источников)
        for col in self.feature_names:
            if col not in daily.columns:
                daily[col] = 0.0
        
        return daily[self.feature_names + ['time', 'date']]
    
    def train(
        self,
        daily_df: pd.DataFrame,
        epochs: int = 100,
        n_splits: int = 3
    ) -> bool:
        """
        Обучает LSTM с временной кросс-валидацией.
        """
        if len(daily_df) < self.sequence_length + self.forecast_horizon + 50:
            logger.warning(f"⚠️ Слишком мало данных: {len(daily_df)}")
            return False
        
        # Подготавливаем последовательности
        X, y = self._create_sequences(daily_df, [])
        
        if len(X) < 50:
            logger.warning(f"⚠️ Слишком мало последовательностей: {len(X)}")
            return False
        
        # Нормализация признаков (по каждому признаку отдельно)
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Временная кросс-валидация (не случайная!)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_val_loss = float('inf')
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"  Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Новая модель для каждого фолда
            self.model = self._build_model((seq_len, n_features))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            val_loss = min(history.history['val_loss'])
            logger.info(f"    Val loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = tf.keras.models.clone_model(self.model)
                best_model.set_weights(self.model.get_weights())
        
        self.model = best_model
        self._save_model()
        
        # Итоговая оценка
        y_pred = self.model.predict(X[-100:], verbose=0)  # последние 100 для оценки
        y_true_classes = np.argmax(y[-100:], axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = np.mean(y_pred_classes == y_true_classes)
        logger.info(f"🧠 LSTM обучена. Final accuracy (last 100): {accuracy:.3f}")
        
        # Распределение классов
        class_dist = pd.Series(y_true_classes).value_counts().sort_index()
        logger.info(f"   Распределение классов в данных: {dict(class_dist)}")
        
        return True
    
    def predict(
        self,
        recent_daily_df: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Прогноз на основе последних sequence_length дней.
        
        Возвращает распределение вероятностей по окнам.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                logger.warning("⚠️ Модель не загружена")
                return self._empty_forecast()
        
        if len(recent_daily_df) < self.sequence_length:
            logger.warning(f"⚠️ Нужно {self.sequence_length} дней, есть {len(recent_daily_df)}")
            return self._empty_forecast()
        
        current_time = current_time or datetime.now(timezone.utc)
        
        # Последние sequence_length дней
        recent = recent_daily_df.tail(self.sequence_length)
        
        # Проверяем свежесть
        last_date = recent['date'].iloc[-1] if 'date' in recent.columns else current_time
        if isinstance(last_date, pd.Timestamp):
            days_stale = (current_time - last_date.to_pydatetime()).days
        else:
            days_stale = 0
        
        # Нормализация
        X = recent[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_input = np.array([X_scaled])
        
        # Предсказание
        probs = self.model.predict(X_input, verbose=0)[0]
        
        # Формируем результат
        forecast = {
            'forecast_time': current_time.isoformat(),
            'data_staleness_days': days_stale,
            'windows': {}
        }
        
        for i, prob in enumerate(probs):
            if i == self.NO_EVENT_CLASS:
                forecast['windows']['no_event_30d+'] = round(float(prob), 3)
            else:
                start, end = self.HORIZON_WINDOWS[i - 1]
                forecast['windows'][f'{start}-{end}d'] = round(float(prob), 3)
        
        # Основной прогноз (максимальная вероятность, кроме "нет события")
        event_probs = {k: v for k, v in forecast['windows'].items() if k != 'no_event_30d+'}
        
        if event_probs and max(event_probs.values()) > 0.3:
            best_window = max(event_probs, key=event_probs.get)
            best_prob = event_probs[best_window]
            
            # Парсим окно для days_to_event
            if '-' in best_window:
                start_day = int(best_window.split('-')[0])
            else:
                start_day = 31
            
            forecast['primary_forecast'] = {
                'window': best_window,
                'probability': best_prob,
                'days_to_event_min': start_day,
                'days_to_event_max': start_day + 3 if start_day < 30 else 90,
                'is_significant': best_prob > 0.5
            }
        else:
            forecast['primary_forecast'] = None
        
        return forecast
    
    def predict_simple_probability(self, recent_daily_df: pd.DataFrame) -> Tuple[float, str]:
        """
        Упрощённый интерфейс: возвращает вероятность и текст.
        """
        forecast = self.predict(recent_daily_df)
        
        if forecast['primary_forecast'] is None:
            return 0.0, "✅ Ситуация стабильна"
        
        pf = forecast['primary_forecast']
        prob = pf['probability']
        window = pf['window']
        
        if prob > 0.7:
            status = f"🔴 ВЫСОКИЙ РИСК! Событие возможно в ближайшие {window}"
        elif prob > 0.4:
            status = f"🟠 Повышенный риск. Окно: {window}"
        else:
            status = f"🟡 Низкий риск. Окно: {window}"
        
        return prob, status
    
    def _save_model(self):
        """Сохраняет модель и скалер."""
        if self.model is not None:
            self.model.save(os.path.join(self.model_dir, 'lstm_forecast.keras'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'lstm_scaler.pkl'))
            # Сохраняем feature names для проверки
            joblib.dump(self.feature_names, os.path.join(self.model_dir, 'feature_names.pkl'))
            logger.info("💾 LSTM модель сохранена")
    
    def _load_model(self):
        """Загружает модель и скалер."""
        try:
            model_path = os.path.join(self.model_dir, 'lstm_forecast.keras')
            scaler_path = os.path.join(self.model_dir, 'lstm_scaler.pkl')
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                if os.path.exists(features_path):
                    saved_features = joblib.load(features_path)
                    if saved_features != self.feature_names:
                        logger.warning(f"⚠️ Признаки изменились! {len(saved_features)} → {len(self.feature_names)}")
                logger.info("📂 LSTM модель загружена")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки LSTM: {e}")
        
        return False

    def _empty_forecast(self) -> Dict:
        windows = {f'{w[0]}-{w[1]}d': 0.0 for w in self.HORIZON_WINDOWS}
        windows['no_event_30d+'] = 1.0
        
        return {
            'forecast_time': datetime.now(timezone.utc).isoformat(),
            'data_staleness_days': None,
            'windows': windows,
            'primary_forecast': None
        }
        
    


# ═══════════════════════════════════════════════════════════════
# ОБРАТНАЯ СОВМЕСТИМОСТЬ (если кто-то импортирует старый EarthquakePredictor)
# ═══════════════════════════════════════════════════════════════

class EarthquakePredictor(EarthquakeLSTMPredictor):
    """
    Псевдоним для обратной совместимости.
    """
    pass


# ═══════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Генерируем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='D')
    
    # События с кластерами (послешоки)
    events = []
    for _ in range(50):
        # Случайные события
        date = np.random.choice(dates)
        mag = np.random.exponential(2) + 3  # преимущественно малые
        
        # Иногда сильные (магнитуда 6+)
        if np.random.random() < 0.1:
            mag = 6 + np.random.exponential(1)
            # Послешоки
            for i in range(1, np.random.randint(3, 8)):
                events.append({
                    'time': date + timedelta(days=i),
                    'magnitude': 4 + np.random.random() * 2,
                    'depth_km': 10
                })
        
        events.append({'time': date, 'magnitude': mag, 'depth_km': 10})
    
    df_events = pd.DataFrame(events).sort_values('time')
    
    # Создаём и обучаем
    predictor = EarthquakeLSTMPredictor(model_dir="data/models_lstm_test")
    
    daily = predictor.prepare_data(df_events)
    print(f"Ежедневных записей: {len(daily)}")
    print(f"Класс распределение:")
    
    # Проверим
    success = predictor.train(daily, epochs=50)
    
    if success:
        # Прогноз
        forecast = predictor.predict(daily)
        print(f"\nПрогноз:")
        for window, prob in forecast['windows'].items():
            print(f"  {window}: {prob*100:.1f}%")
        
        if forecast['primary_forecast']:
            pf = forecast['primary_forecast']
            print(f"\nОсновной: {pf['window']} с вероятностью {pf['probability']*100:.1f}%")
        
