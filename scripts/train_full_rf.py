#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_rf.py — обучение RF на полных годовых данных
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_full_year(year):
    data_dir = Path(__file__).parent.parent / "data"
    path = data_dir / f"cache_{year}01_{year}12.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
        return df
    return None


def prepare_rf_data(df, prediction_window=7):
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']
    X, y = [], []
    df_sorted = df.sort_values('time').reset_index(drop=True)
    for i in range(len(df_sorted) - prediction_window):
        X.append(df_sorted.iloc[i][feature_names].values)
        event_time = df_sorted.iloc[i]['time']
        future = df_sorted[
            (df_sorted['time'] > event_time) &
            (df_sorted['time'] <= event_time + timedelta(days=prediction_window)) &
            (df_sorted['magnitude'] >= 6.0)
        ]
        y.append(1 if not future.empty else 0)
    return np.array(X), np.array(y)


def main():
    logger.info("=" * 60)
    logger.info("🌲 ОБУЧЕНИЕ RF НА ПОЛНЫХ ГОДОВЫХ ДАННЫХ")
    logger.info("=" * 60)
    
    years = [2022, 2023, 2024]
    all_dfs = []
    for year in years:
        df = load_full_year(year)
        if df is not None:
            all_dfs.append(df)
            logger.info(f"✅ {year}: {len(df)} событий")
    
    if not all_dfs:
        return
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['id']).sort_values('time')
    logger.info(f"📊 Итого: {len(df_all)} событий")
    
    X, y = prepare_rf_data(df_all, prediction_window=7)
    logger.info(f"📊 Примеров: {len(X)}, положительных: {sum(y)}, отрицательных: {len(y)-sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    logger.info("⚡ Обучение RF...")
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    y_pred_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ RF НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    logger.info(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_prob):.3f}")
    logger.info("=" * 60)
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "rf_full_model.pkl", 'wb') as f:
        pickle.dump(rf, f)
    with open(models_dir / "rf_full_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"✅ Модель сохранена в {models_dir / 'rf_full_model.pkl'}")


if __name__ == "__main__":
    main()
