#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_zone_rf.py — обучение RandomForest для классификации регионов
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("🌲 ОБУЧЕНИЕ RANDOMFOREST ДЛЯ КЛАССИФИКАЦИИ РЕГИОНОВ")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Загружаем данные
    df = pd.read_pickle(data_dir / "events_with_regions.pkl")
    
    # Признаки
    feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                     'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean']
    
    X = df[feature_names].values
    y = df['region_id'].values

    logger.info(f"📊 Данные: {len(X)} событий, {df['region_id'].nunique()} регионов")

    # Масштабирование (для RF не обязательно, но помогает)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Сохраняем скейлер
    with open(models_dir / "scaler_rf.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Разделяем
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Обучаем RandomForest
    logger.info("⚡ Обучение RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Оценка
    y_pred = rf.predict(X_test)
    
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"  F1-score (weighted): {f1_score(y_test, y_pred, average='weighted'):.3f}")
    logger.info("=" * 60)
    
    # Детальный отчёт
    logger.info("\n📋 Подробный отчёт:")
    logger.info(classification_report(y_test, y_pred, zero_division=0))

    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n📊 Важность признаков:")
    for _, row in feature_importance.iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.3f}")

    # Сохраняем модель
    with open(models_dir / "zone_rf_model.pkl", 'wb') as f:
        pickle.dump(rf, f)
    
    with open(models_dir / "feature_importance.pkl", 'wb') as f:
        pickle.dump(feature_importance, f)

    logger.info(f"✅ Модель сохранена в {models_dir / 'zone_rf_model.pkl'}")


if __name__ == "__main__":
    main()
