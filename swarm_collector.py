#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_collector.py — сбор данных миссии ESA Swarm
Явная передача токена без сохранения в конфигурацию
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from viresclient import SwarmRequest
    VIRES_AVAILABLE = True
    logger.info("✅ viresclient загружен")
except ImportError:
    VIRES_AVAILABLE = False
    logger.warning("⚠️ viresclient не установлен")


class SwarmCollector:
    """Сборщик данных миссии ESA Swarm с явной передачей токена."""
    
    VIRES_URL = "https://vires.services/ows"
    
    def __init__(self, cache_dir="data/swarm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get('VIRES_ACCESS_TOKEN')
        
        if self.token:
            logger.info(f"✅ Swarm токен загружен (длина: {len(self.token)})")
        else:
            logger.warning("⚠️ Swarm токен не найден (VIRES_ACCESS_TOKEN)")
    
    def _create_request(self):
        """Создаёт SwarmRequest с явной передачей токена."""
        if not VIRES_AVAILABLE or not self.token:
            return None
        
        try:
            request = SwarmRequest(url=self.VIRES_URL, token=self.token)
            logger.debug("🛰️ SwarmRequest создан с токеном")
            return request
        except Exception as e:
            logger.error(f"❌ Ошибка создания SwarmRequest: {e}")
            return None
    
    def fetch_magnetic_data(self, lat, lon, start_time, end_time, spacecraft='A', use_fast=False):
        """
        Получает магнитные данные с указанного спутника.
        """
        request = self._create_request()
        if request is None:
            return pd.DataFrame()
        
        # Используем OPER (более стабильные данные)
        collection = f"SW_OPER_MAG{spacecraft}_LR_1B"
        if use_fast:
            collection = f"SW_FAST_MAG{spacecraft}_LR_1B"
        
        try:
            logger.info(f"🛰️ Swarm: запрос {collection} за {start_time.date()} — {end_time.date()}")
            
            request.set_collection(collection)
            request.set_products(
                measurements=["F", "B_NEC"],
                models=["CHAOS-Core"],
                auxiliaries=["QDLat", "QDLon", "Radius"],
                sampling_step="PT10S"
            )
            
            data = request.get_between(
                start_time=start_time,
                end_time=end_time,
                asynchronous=False,
                show_progress=False
            )
            
            df = data.as_dataframe()
            
            if df.empty:
                return pd.DataFrame()
            
            # Фильтр по координатам
            lat_range = 5.0
            lon_range = 5.0
            
            df_filtered = df[
                (df['QDLat'] >= lat - lat_range) &
                (df['QDLat'] <= lat + lat_range) &
                (df['QDLon'] >= lon - lon_range) &
                (df['QDLon'] <= lon + lon_range)
            ]
            
            if df_filtered.empty:
                return pd.DataFrame()
            
            # Аномалии
            if 'F_CHAOS-Core' in df_filtered.columns:
                df_filtered['F_anomaly'] = df_filtered['F'] - df_filtered['F_CHAOS-Core']
            else:
                df_filtered['F_anomaly'] = df_filtered['F'] - df_filtered['F'].mean()
            
            logger.info(f"🛰️ Swarm: получено {len(df_filtered)} записей")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Swarm ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """Собирает данные для события."""
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        # Пробуем спутник A (Alpha)
        mag_df = self.fetch_magnetic_data(lat, lon, start, end, spacecraft='A', use_fast=False)
        
        # Если A не дал данных, пробуем B или C
        if mag_df.empty:
            mag_df = self.fetch_magnetic_data(lat, lon, start, end, spacecraft='B', use_fast=False)
        if mag_df.empty:
            mag_df = self.fetch_magnetic_data(lat, lon, start, end, spacecraft='C', use_fast=False)
        
        return {
            'magnetic': mag_df,
            'start': start,
            'end': end,
            'lat': lat,
            'lon': lon,
            'event_time': event_time
        }
    
    def detect_anomaly(self, df, threshold=2.0):
        """Обнаруживает аномалии."""
        if df.empty or 'F_anomaly' not in df.columns:
            return {'has_anomaly': False, 'max_z_score': 0.0, 'anomaly_count': 0}
        
        z_scores = np.abs((df['F_anomaly'] - df['F_anomaly'].mean()) / df['F_anomaly'].std())
        anomalies = z_scores > threshold
        
        return {
            'has_anomaly': anomalies.any(),
            'max_z_score': float(z_scores.max()) if not z_scores.empty else 0.0,
            'anomaly_count': int(anomalies.sum()) if not z_scores.empty else 0
        }
    
    def get_status(self):
        return {
            'available': VIRES_AVAILABLE,
            'token_set': bool(self.token)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = SwarmCollector()
    status = collector.get_status()
    print(f"🛰️ Статус: {status}")
