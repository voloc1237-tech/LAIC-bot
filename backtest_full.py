#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py — быстрый бэктест с использованием кэшированных данных.
"""

import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Backtest')

class Backtester:
    def __init__(self, cache_path="data/cache_202308_202312.json"):
        self.cache_path = Path(cache_path)
        self.df = None
        self.load_cache()

    def load_cache(self):
        if not self.cache_path.exists():
            logger.error(f"❌ Файл кэша {self.cache_path} не найден")
            return False
        
        # Определяем формат по расширению файла
        if self.cache_path.suffix == '.json':
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:  # .pkl
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
        
        self.df = pd.DataFrame(data)
        # Исправленный парсинг времени с указанием формата ISO8601
        self.df['time'] = pd.to_datetime(self.df['time'], utc=True, format='ISO8601')
        self.df.sort_values('time', inplace=True)
        logger.info(f"✅ Загружено {len(self.df)} событий из кэша")
        return True

    def run(self, start_date, end_date, prediction_window=7):
        if self.df is None:
            return

        # Оставляем только события в тестовом периоде
        test_df = self.df[(self.df['time'] >= start_date) & (self.df['time'] <= end_date)]
        if test_df.empty:
            logger.error("Нет тестовых данных")
            return

        results = []
        current_date = start_date
        day_index = 0

        while current_date <= end_date and day_index < len(test_df):
            logger.info(f"Обработка {current_date.date()}...")
            
            # Данные до текущей даты (для обучения)
            train_df = self.df[self.df['time'] < current_date]
            if len(train_df) < 30:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Признаки для обучения
            X_train = train_df[['magnitude', 'depth_km', 'lst_celsius', 'temp_mean', 
                               'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']].fillna(0)
            
            # Целевая переменная: было ли событие M≥6.0 в следующие 7 дней после каждого события
            y_train = []
            for _, row in train_df.iterrows():
                future = self.df[
                    (self.df['time'] > row['time']) &
                    (self.df['time'] <= row['time'] + timedelta(days=prediction_window)) &
                    (self.df['magnitude'] >= 6.0)
                ]
                y_train.append(1 if not future.empty else 0)
            y_train = np.array(y_train)

            if len(set(y_train)) < 2:
                logger.info("  Нет положительных примеров, пропускаем")
                current_date += timedelta(days=1)
                day_index += 1
                continue

            # Обучаем модель
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)

            # Прогноз на последние события
            last_events = test_df[test_df['time'] < current_date + timedelta(days=prediction_window)].tail(7)
            if last_events.empty:
                current_date += timedelta(days=1)
                day_index += 1
                continue

            X_pred = last_events[['magnitude', 'depth_km', 'lst_celsius', 'temp_mean', 
                                 'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']].fillna(0)
            prob = clf.predict_proba(X_pred)[:, 1].mean()

            # Проверяем реальное событие
            actual = test_df[
                (test_df['time'] >= current_date) &
                (test_df['time'] <= current_date + timedelta(days=prediction_window)) &
                (test_df['magnitude'] >= 6.0)
            ]
            actual_event = not actual.empty
            actual_mag = actual['magnitude'].max() if actual_event else None

            results.append({
                'date': current_date,
                'pred_prob': prob,
                'actual_event': actual_event,
                'actual_mag': actual_mag
            })

            logger.info(f"  Прогноз: P={prob:.2f} | Факт: {'Да' if actual_event else 'Нет'}, M={actual_mag or 0:.1f}")

            current_date += timedelta(days=1)
            day_index += 1

        # Оценка
        self.evaluate(results)

    def evaluate(self, results):
        df = pd.DataFrame(results)
        if df.empty:
            return

        df['pred_binary'] = df['pred_prob'] >= 0.3
        df['actual_binary'] = df['actual_event']

        valid = df[df['pred_prob'].notna()]
        if valid.empty:
            return

        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА (с кэшем)")
        print("="*60)
        print(f"Всего дней: {len(df)}")
        print(f"Событий M≥6.0: {df['actual_event'].sum()}")
        print(f"Accuracy:  {accuracy_score(valid['actual_binary'], valid['pred_binary']):.3f}")
        print(f"Precision: {precision_score(valid['actual_binary'], valid['pred_binary'], zero_division=0):.3f}")
        print(f"Recall:    {recall_score(valid['actual_binary'], valid['pred_binary'], zero_division=0):.3f}")
        print(f"F1-score:  {f1_score(valid['actual_binary'], valid['pred_binary'], zero_division=0):.3f}")
        try:
            auc = roc_auc_score(valid['actual_binary'], valid['pred_prob'].fillna(0))
            print(f"ROC-AUC:   {auc:.3f}")
        except:
            pass
        false_alarms = valid[(valid['pred_binary'] == True) & (valid['actual_binary'] == False)]
        print(f"Ложных тревог: {len(false_alarms)} ({len(false_alarms)/len(valid)*100:.1f}%)")
        print("="*60)

if __name__ == "__main__":
    bt = Backtester(cache_path="data/cache_202308_202312.json")
    start = datetime(2023, 10, 1, tzinfo=timezone.utc)  # тестовый период (2 месяца)
    end = datetime(2023, 12, 31, tzinfo=timezone.utc)
    bt.run(start, end, prediction_window=7)
