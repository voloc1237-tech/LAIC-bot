#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_collector.py — сбор данных миссии ESA Swarm
Поддерживает: TEC, магнитные аномалии (F, B_NEC)
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
except ImportError:
    VIRES_AVAILABLE = False
    logger.warning("⚠️ viresclient не установлен")


class SwarmCollector:
    """Сборщик данных Swarm с приоритетом на магнитные аномалии и TEC."""

    VIRES_URL = "https://vires.services/ows"

    def __init__(self, cache_dir="data/swarm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get('VIRES_ACCESS_TOKEN')
        self.available = VIRES_AVAILABLE and bool(self.token)

        if self.token:
            logger.info(f"✅ Swarm токен загружен (длина: {len(self.token)})")
        if not self.available:
            logger.warning("⚠️ Swarm недоступен (токен или библиотека отсутствуют)")

    def _create_request(self):
        if not self.available:
            return None
        try:
            return SwarmRequest(url=self.VIRES_URL, token=self.token)
        except Exception as e:
            logger.error(f"❌ SwarmRequest ошибка: {e}")
            return None

    def fetch_magnetic_data(self, lat, lon, start_time, end_time, spacecraft='A'):
        """Получает магнитные данные (F, B_NEC) с указанного спутника."""
        request = self._create_request()
        if request is None:
            return pd.DataFrame()

        collection = f"SW_OPER_MAG{spacecraft}_LR_1B"
        try:
            logger.debug(f"🛰️ Swarm магнитные: {collection} за {start_time.date()} — {end_time.date()}")

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
            lat_range, lon_range = 5.0, 5.0
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

            logger.info(f"🛰️ Swarm магнитные: {len(df_filtered)} записей")
            return df_filtered

        except Exception as e:
            logger.warning(f"⚠️ Swarm магнитные ошибка: {e}")
            return pd.DataFrame()

    def fetch_tec_data(self, lat, lon, start_time, end_time):
        """Получает TEC из Swarm GNSS."""
        request = self._create_request()
        if request is None:
            return pd.DataFrame()

        try:
            # Список коллекций для TEC (по порядку приоритета)
            collections = [
                "SW_OPER_TEC_TMS_2F",     # Основная OPER
                "SW_FAST_TEC_TMS_2F",     # FAST (если есть)
                "SW_OPER_TEC_TMS_2F",     # Альтернатива
            ]
        
            df = pd.DataFrame()
            for collection in collections:
                try:
                    logger.debug(f"🛰️ Swarm TEC: пробуем {collection}")
                    request.set_collection(collection)
                    request.set_products(
                        measurements=["TEC"],
                        auxiliaries=["Latitude", "Longitude"],
                        sampling_step="PT10S"
                    )

                    data = request.get_between(
                        start_time=start_time,
                        end_time=end_time,
                        asynchronous=False,
                        show_progress=False
                    )

                    df = data.as_dataframe()
                    if not df.empty:
                        logger.info(f"🛰️ Swarm TEC: получено {len(df)} записей из {collection}")
                        break
                except Exception as e:
                    logger.debug(f"   {collection} не сработал: {e}")
                    continue

            if df.empty:
                logger.warning("⚠️ Swarm TEC: нет данных ни из одной коллекции")
                return pd.DataFrame()

            # Фильтр по координатам
            df_filtered = df[
                (df['Latitude'] >= lat - 5) &
                (df['Latitude'] <= lat + 5) &
                (df['Longitude'] >= lon - 5) &
                (df['Longitude'] <= lon + 5)
            ]

            if df_filtered.empty:
                logger.warning(f"⚠️ Swarm TEC: нет данных в радиусе 5°")
                return pd.DataFrame()

            df_filtered['event_lat'] = lat
            df_filtered['event_lon'] = lon

            logger.info(f"🛰️ Swarm TEC: {len(df_filtered)} записей")
            return df_filtered

        except Exception as e:
            logger.warning(f"⚠️ Swarm TEC ошибка: {e}")
            return pd.DataFrame()
    
    #def fetch_tec_data(self, lat, lon, start_time, end_time):
        #"""Получает TEC из Swarm GNSS."""
        #request = self._create_request()
        #if request is None:
            #return pd.DataFrame()

        #try:
            # Пробуем FAST TEC
            #collection = "SW_FAST_TEC_TMS_2F"
            #request.set_collection(collection)
            #request.set_products(
                #measurements=["TEC"],
                #auxiliaries=["Latitude", "Longitude"],
                #sampling_step="PT10S"
            #)

            #data = request.get_between(
                #start_time=start_time,
                #end_time=end_time,
                #asynchronous=False,
                #show_progress=False
            #)

            #df = data.as_dataframe()
            #if df.empty:
                # Пробуем OPER TEC
                #collection = "SW_OPER_TEC_TMS_2F"
                #request.set_collection(collection)
                #data = request.get_between(
                    #start_time=start_time,
                    #end_time=end_time,
                    #asynchronous=False,
                    #show_progress=False
                #)
                #df = data.as_dataframe()

            #if df.empty:
                #return pd.DataFrame()

            # Фильтр по координатам
            #df_filtered = df[
                #(df['Latitude'] >= lat - 5) &
                #(df['Latitude'] <= lat + 5) &
                #(df['Longitude'] >= lon - 5) &
                #(df['Longitude'] <= lon + 5)
            #]

            #if df_filtered.empty:
                #return pd.DataFrame()

            #logger.info(f"🛰️ Swarm TEC: {len(df_filtered)} записей")
            #return df_filtered

        #except Exception as e:
            #logger.warning(f"⚠️ Swarm TEC ошибка: {e}")
            #return pd.DataFrame()
    
   
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """Собирает магнитные и TEC данные для события."""
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)

        # Магнитные данные (все три спутника)
        mag_dfs = []
        for sc in ['A', 'B', 'C']:
            df = self.fetch_magnetic_data(lat, lon, start, end, spacecraft=sc)
            if not df.empty:
                mag_dfs.append(df)
        mag_df = pd.concat(mag_dfs, ignore_index=True) if mag_dfs else pd.DataFrame()

        # TEC
        tec_df = self.fetch_tec_data(lat, lon, start, end)

        return {
            'magnetic': mag_df,
            'tec': tec_df,
            'start': start,
            'end': end,
            'lat': lat,
            'lon': lon,
            'event_time': event_time,
            'source': 'Swarm'
        }

    def detect_magnetic_anomaly(self, df, threshold=2.0):
        """Обнаруживает аномалии в магнитных данных."""
        if df.empty or 'F_anomaly' not in df.columns:
            return {'has_anomaly': False, 'max_z_score': 0.0}

        z_scores = np.abs((df['F_anomaly'] - df['F_anomaly'].mean()) / df['F_anomaly'].std())
        anomalies = z_scores > threshold

        return {
            'has_anomaly': anomalies.any(),
            'max_z_score': float(z_scores.max()) if not z_scores.empty else 0.0,
            'anomaly_count': int(anomalies.sum()) if not z_scores.empty else 0
        }

    def get_tec_mean(self, df):
        """Возвращает средний TEC из данных."""
        if df.empty or 'TEC' not in df.columns:
            return None
        return float(df['TEC'].mean())

    def get_status(self):
        return {'available': self.available, 'token_set': bool(self.token)}
