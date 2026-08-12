#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_backtest.py — быстрый бэктест для GitHub Actions
Использует только доступные API, заменяет недостающие данные на климатологию.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import yaml
import requests

# Импорты модулей бота (используем только те, что нужны)
from data_collector import USGSCollector
from aftershock_filter import filter_aftershocks
from weather_collector import WeatherCollector
from space_weather import SpaceWeatherCollector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Backtest')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FastBacktester:
    def __init__(self, settings_path="config/settings.yaml"):
        with open(settings_path, 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)
        self.weather = WeatherCollector()
        self.space = SpaceWeatherCollector()
        self.results = []

    def _get_climatology(self, lat, lon, event_time):
        """
        Возвращает климатологические средние для LST, TEC, влажности.
        """
        # Простая модель: температура зависит от широты и сезона
        lat_abs = abs(lat)
        # Базовая температура: экватор 30°C, полюса 0°C
        base_temp = 30 - lat_abs * 0.3
        # Сезонная поправка (северное полушарие)
        day_of_year = event_time.timetuple().tm_yday
        seasonal = 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        temp = base_temp + seasonal
        # Влажность: выше у экватора, ниже у полюсов
        humidity = 80 - lat_abs * 0.5
        # TEC: выше на экваторе, днём больше
        hour = event_time.hour
        tec = 20 + 10 * np.sin((hour - 8) * np.pi / 12) - lat_abs * 0.2
        return temp, humidity, max(tec, 5)

    def _prepare_features_fast(self, df_events):
        """
        Быстрое добавление признаков: погода (Open-Meteo), космос (Kp, Dst, F10.7),
        климатология для LST и TEC.
        """
        features = []
        total = len(df_events)
        for i, (_, row) in enumerate(df_events.iterrows()):
            if i % 10 == 0:
                logger.info(f"   Обработка событий: {i+1}/{total}")
            lat, lon = row['latitude'], row['longitude']
            event_time = row['time']

            # 1. Погода (Open-Meteo) — быстрый запрос
            try:
                wdf = self.weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                if not wdf.empty:
                    temp_mean = wdf['temp_2m_mean'].mean()
                    humidity_mean = wdf['humidity_2m_mean'].mean()
                else:
                    temp_mean = humidity_mean = None
            except:
                temp_mean = humidity_mean = None

            # 2. Космическая погода (Kp, Dst, F10.7)
            try:
                start_sp = event_time - timedelta(days=7)
                end_sp = event_time + timedelta(days=3)
                space_data = self.space.fetch_all_for_period(start_sp, end_sp)
                kp_mean = space_data['kp']['kp'].mean() if not space_data['kp'].empty else None
                dst_mean = space_data['dst']['dst'].mean() if not space_data['dst'].empty else None
                f107_mean = space_data['f107']['f107'].mean() if not space_data['f107'].empty else None
            except:
                kp_mean = dst_mean = f107_mean = None

            # 3. Климатология для LST и TEC (если погода не получена, тоже используем климатологию)
            if temp_mean is None or humidity_mean is None:
                temp_mean, humidity_mean, _ = self._get_climatology(lat, lon, event_time)
            # TEC (климатология)
            _, _, tec_mean = self._get_climatology(lat, lon, event_time)
            # LST (приблизительно как температура)
            lst_celsius = temp_mean - 2  # LST обычно на пару градусов выше температуры воздуха

            features.append({
                'id': row['id'],
                'time': event_time,
                'lat': lat,
                'lon': lon,
                'magnitude': row['magnitude'],
                'depth_km': row['depth_km'],
                'lst_celsius': lst_celsius,
                'temp_mean': temp_mean,
                'humidity_mean': humidity_mean,
                'tec_mean': tec_mean,
                'kp_mean': kp_mean if kp_mean is not None else 2.0,
                'dst_mean': dst_mean if dst_mean is not None else -10.0,
                'f107_mean': f107_mean if f107_mean is not None else 150.0,
            })

        return pd.DataFrame(features)

    def run(self, start_date, end_date, test_days=30, lookback_days=180, prediction_window=7):
        """
        Запускает бэктест.
        """
        # Загружаем USGS данные за период
        data_start = start_date - timedelta(days=lookback_days)
        data_end = end_date + timedelta(days=prediction_window)
        collector = USGSCollector(cache_enabled=True)
        all_events = collector.fetch_global(data_start, data_end, min_magnitude=5.0)  # берём только M≥5
        if all_events.empty:
            logger.error("Нет данных USGS")
            return

        all_events = all_events.sort_values('time').reset_index(drop=True)
        test_events = all_events[all_events['time'] >= start_date]
        if test_events.empty:
            logger.error("Нет тестовых данных")
            return

        # Цикл по дням
        current_date = start_date
        day_index = 0
        while current_date < end_date and day_index < test_days:
            logger.info(f"📅 День {day_index+1}/{test_days}: {current_date.date()}")

            # Данные до текущего дня
            window_events = all_events[
                (all_events['time'] >= current_date - timedelta(days=lookback_days)) &
                (all_events['time'] < current_date)
            ]
            if len(window_events) < 30:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Фильтруем афтершоки
            window_filtered, _ = filter_aftershocks(window_events)
            if window_filtered.empty:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Добавляем признаки
            logger.info("   Добавление признаков...")
            features_df = self._prepare_features_fast(window_filtered)
            if features_df.empty:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Обучаем модель
            X = features_df[['magnitude', 'depth_km', 'lst_celsius', 'temp_mean', 'humidity_mean',
                             'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']].fillna(0)
            # Целевая переменная: было ли сильное событие (M≥6) в следующие 7 дней
            y = []
            for _, row in features_df.iterrows():
                future = all_events[
                    (all_events['time'] > row['time']) &
                    (all_events['time'] <= row['time'] + timedelta(days=prediction_window)) &
                    (all_events['magnitude'] >= 6.0)
                ]
                y.append(1 if not future.empty else 0)
            y = np.array(y)

            if len(set(y)) < 2:
                logger.info("   Нет положительных примеров, пропускаем")
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Разделяем и обучаем
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)

            # Прогноз на последние 7 дней (для текущей даты)
            last_events = features_df.tail(7)
            if last_events.empty:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            X_pred = last_events[['magnitude', 'depth_km', 'lst_celsius', 'temp_mean', 'humidity_mean',
                                  'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']].fillna(0)
            prob = clf.predict_proba(X_pred)[:, 1].mean()
            pred_mag = last_events['magnitude'].max() * (0.5 + prob * 0.5)  # упрощённо

            # Фактические события в следующие prediction_window дней
            future_events = test_events[
                (test_events['time'] >= current_date) &
                (test_events['time'] < current_date + timedelta(days=prediction_window)) &
                (test_events['magnitude'] >= 6.0)
            ]
            actual_event = not future_events.empty
            actual_mag = future_events['magnitude'].max() if actual_event else None

            self.results.append({
                'date': current_date,
                'pred_prob': prob,
                'pred_mag': pred_mag,
                'actual_event': actual_event,
                'actual_mag': actual_mag,
            })

            logger.info(f"   Прогноз: P={prob:.2f}, M={pred_mag:.2f} | Факт: {'Да' if actual_event else 'Нет'}, M={actual_mag or 0:.1f}")

            current_date += timedelta(days=1)
            day_index += 1

    def evaluate(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("Нет результатов")
            return

        df['pred_binary'] = df['pred_prob'] >= 0.3
        df['actual_binary'] = df['actual_event']

        valid = df[df['pred_prob'].notna()]
        if valid.empty:
            logger.warning("Нет валидных прогнозов")
            return

        acc = accuracy_score(valid['actual_binary'], valid['pred_binary'])
        prec = precision_score(valid['actual_binary'], valid['pred_binary'], zero_division=0)
        rec = recall_score(valid['actual_binary'], valid['pred_binary'], zero_division=0)
        f1 = f1_score(valid['actual_binary'], valid['pred_binary'], zero_division=0)
        try:
            auc = roc_auc_score(valid['actual_binary'], valid['pred_prob'].fillna(0))
        except:
            auc = 0

        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
        print("="*60)
        print(f"Всего дней: {len(df)}")
        print(f"Событий M≥6.0: {df['actual_event'].sum()}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-score:  {f1:.3f}")
        print(f"ROC-AUC:   {auc:.3f}")
        print("="*60)

        false_alarms = valid[(valid['pred_binary'] == True) & (valid['actual_binary'] == False)]
        print(f"Ложных тревог: {len(false_alarms)} ({len(false_alarms)/len(valid)*100:.1f}%)")

        df.to_csv('backtest_results.csv', index=False)
        print("✅ Результаты сохранены в backtest_results.csv")

if __name__ == "__main__":
    # Параметры: тестируем 30 дней 2024 года
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    bt = FastBacktester()
    bt.run(start, end, test_days=30, lookback_days=180, prediction_window=7)
    bt.evaluate()
