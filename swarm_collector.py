#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_collector.py — сбор данных миссии ESA Swarm
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

try:
    from viresclient import SwarmRequest, ClientConfig
    VIRES_AVAILABLE = True
    logger.info("✅ viresclient загружен")
except ImportError:
    VIRES_AVAILABLE = False
    logger.warning("⚠️ viresclient не установлен")


class SwarmCollector:
    """
    Сборщик данных миссии ESA Swarm.
    """
    
    VIRES_URL = "https://vires.services/ows"
    
    def __init__(self, cache_dir="data/swarm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.token = os.environ.get('VIRES_ACCESS_TOKEN')
        self.request = None
        
        if self.token:
            logger.info(f"✅ Swarm токен загружен (длина: {len(self.token)})")
        else:
            logger.warning("⚠️ Swarm токен не найден (VIRES_ACCESS_TOKEN)")
        
        self._init_request()
    
    def _init_request(self):
        """Инициализирует запрос к Swarm с токеном."""
        if not VIRES_AVAILABLE or not self.token:
            return
        
        try:
            # Сохраняем токен в конфигурацию
            cc = ClientConfig()
            cc.set_site_config(self.VIRES_URL, token=self.token)
            cc.default_url = self.VIRES_URL
            cc.save()
            logger.info("🔑 Swarm: токен сохранён в конфигурацию")
            
            self.request = SwarmRequest()
            logger.info("🛰️ Swarm: соединение установлено")
        except Exception as e:
            logger.warning(f"⚠️ Swarm: ошибка соединения: {e}")
            self.request = None
    
    def _get_collection(self, spacecraft='A', use_fast=True, data_type='mag'):
        """Получить правильное имя коллекции."""
        # Для магнитных данных используем MAGx_LR_1B
        if data_type == 'mag':
            prefix = "SW_FAST" if use_fast else "SW_OPER"
            return f"{prefix}_MAG{spacecraft}_LR_1B"
        # Для TEC
        elif data_type == 'tec':
            prefix = "SW_FAST" if use_fast else "SW_OPER"
            return f"{prefix}_TEC_TMS_2F"
        return None
    
    def fetch_magnetic_data(self, lat, lon, start_time, end_time, spacecraft='A', use_fast=True):
        """Получает магнитные данные."""
        if not VIRES_AVAILABLE or self.request is None:
            return pd.DataFrame()
        
        collection = self._get_collection(spacecraft, use_fast, 'mag')
        if not collection:
            return pd.DataFrame()
        
        try:
            logger.info(f"🛰️ Swarm: запрос {collection} за {start_time.date()} — {end_time.date()}")
            
            self.request.set_collection(collection)
            self.request.set_products(
                measurements=["F", "B_NEC"],
                models=["CHAOS-Core"],
                auxiliaries=["QDLat", "QDLon", "Radius", "Spacecraft"],
                sampling_step="PT10S"
            )
            
            data = self.request.get_between(
                start_time=start_time,
                end_time=end_time,
                asynchronous=False  # синхронный режим для теста
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
            
            df_filtered['event_lat'] = lat
            df_filtered['event_lon'] = lon
            
            logger.info(f"🛰️ Swarm: получено {len(df_filtered)} записей")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Swarm ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_tec_data(self, lat, lon, start_time, end_time, use_fast=True):
        """Получает TEC данные."""
        if not VIRES_AVAILABLE or self.request is None:
            return pd.DataFrame()
        
        collection = self._get_collection('A', use_fast, 'tec')
        if not collection:
            return pd.DataFrame()
        
        try:
            logger.info(f"🛰️ Swarm TEC: запрос {collection}")
            
            self.request.set_collection(collection)
            self.request.set_products(
                measurements=["TEC"],
                auxiliaries=["Latitude", "Longitude"],
                sampling_step="PT10S"
            )
            
            data = self.request.get_between(
                start_time=start_time,
                end_time=end_time,
                asynchronous=False
            )
            
            df = data.as_dataframe()
            
            if df.empty:
                return pd.DataFrame()
            
            df_filtered = df[
                (df['Latitude'] >= lat - 5) &
                (df['Latitude'] <= lat + 5) &
                (df['Longitude'] >= lon - 5) &
                (df['Longitude'] <= lon + 5)
            ]
            
            if df_filtered.empty:
                return pd.DataFrame()
            
            df_filtered['event_lat'] = lat
            df_filtered['event_lon'] = lon
            
            logger.info(f"🛰️ Swarm TEC: получено {len(df_filtered)} записей")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Swarm TEC ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3, use_fast=True):
        """Собирает данные для события."""
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        # Пробуем все три спутника
        mag_dfs = []
        for spacecraft in ['A', 'B', 'C']:
            df = self.fetch_magnetic_data(lat, lon, start, end, spacecraft, use_fast)
            if not df.empty:
                mag_dfs.append(df)
        
        mag_df = pd.concat(mag_dfs, ignore_index=True) if mag_dfs else pd.DataFrame()
        tec_df = self.fetch_tec_data(lat, lon, start, end, use_fast)
        
        return {
            'magnetic': mag_df,
            'tec': tec_df,
            'start': start,
            'end': end,
            'lat': lat,
            'lon': lon,
            'event_time': event_time,
            'data_source': 'FAST' if use_fast else 'OPER'
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
            'token_set': bool(self.token),
            'request_ready': self.request is not None
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = SwarmCollector()
    status = collector.get_status()
    print(f"🛰️ Статус: {status}")
