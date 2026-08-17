#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_collector.py — сбор данных миссии ESA Swarm
Для обнаружения магнитных и ионосферных предвестников землетрясений.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# Проверка наличия viresclient
try:
    from viresclient import SwarmRequest
    VIRES_AVAILABLE = True
    logger.info("✅ viresclient загружен")
except ImportError:
    VIRES_AVAILABLE = False
    logger.warning("⚠️ viresclient не установлен. Установите: pip install viresclient")


class SwarmCollector:
    """
    Сборщик данных миссии ESA Swarm.
    - Магнитное поле (MFV, F)
    - Ионосферные параметры (Ne, Te)
    - TEC (GNSS)
    """
    
    def __init__(self, cache_dir="data/swarm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request = None
        self._init_request()
    
    def _init_request(self):
        """Инициализирует запрос к Swarm."""
        if VIRES_AVAILABLE:
            try:
                self.request = SwarmRequest()
                logger.info("🛰️ Swarm: соединение установлено")
            except Exception as e:
                logger.warning(f"⚠️ Swarm: ошибка соединения: {e}")
                self.request = None
    
    def fetch_magnetic_data(self, lat, lon, start_time, end_time, radius_km=500):
        """
        Получает магнитные данные (MFV) за период.
        Возвращает DataFrame с аномалиями магнитного поля.
        """
        if not VIRES_AVAILABLE or self.request is None:
            logger.warning("⚠️ Swarm недоступен, возвращаю пустые данные")
            return pd.DataFrame()
        
        # Проверяем кэш
        cache_key = f"swarm_mag_{lat:.2f}_{lon:.2f}_{start_time.date()}_{end_time.date()}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.info(f"📂 Swarm: загружено из кэша ({len(df)} записей)")
                    return df
            except:
                pass
        
        try:
            # Настраиваем запрос
            self.request.set_collection("SW_OPER_MAGA_LR_1B")  # Магнитные данные
            self.request.set_products(
                measurements=["F", "B_NEC"],
                models=["CHAOS-Core"],  # Для вычисления аномалий
                auxiliaries=["QDLat", "QDLon"],
                sampling_step="PT10S"
            )
            
            # Добавляем фильтр по региону
            # Swarm API: добавляем пространственный фильтр
            # Примечание: для простоты используем временной фильтр, а координаты фильтруем позже
            
            data = self.request.get_between(
                start_time=start_time,
                end_time=end_time
            )
            
            df = data.as_dataframe()
            
            if df.empty:
                logger.warning("⚠️ Swarm: нет данных за период")
                return pd.DataFrame()
            
            # Фильтруем по координатам (приблизительно)
            # Используем приближение: 1° ~ 111 км
            lat_range = radius_km / 111.0
            lon_range = radius_km / 111.0
            
            df = df[
                (df['QDLat'] >= lat - lat_range) &
                (df['QDLat'] <= lat + lat_range) &
                (df['QDLon'] >= lon - lon_range) &
                (df['QDLon'] <= lon + lon_range)
            ]
            
            if df.empty:
                logger.warning(f"⚠️ Swarm: нет данных в радиусе {radius_km} км")
                return pd.DataFrame()
            
            # Вычисляем аномалии (отклонение от модели CHAOS)
            # В данных уже есть модели, используем их
            if 'F_CHAOS-Core' in df.columns:
                df['F_anomaly'] = df['F'] - df['F_CHAOS-Core']
            else:
                # Если модели нет, используем среднее
                df['F_anomaly'] = df['F'] - df['F'].mean()
            
            # Сохраняем в кэш
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            logger.info(f"🛰️ Swarm: получено {len(df)} записей магнитных данных")
            return df
            
        except Exception as e:
            logger.error(f"❌ Swarm ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_tec_data(self, lat, lon, start_time, end_time):
        """
        Получает TEC из Swarm GNSS.
        """
        if not VIRES_AVAILABLE or self.request is None:
            return pd.DataFrame()
        
        cache_key = f"swarm_tec_{lat:.2f}_{lon:.2f}_{start_time.date()}_{end_time.date()}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    return df
            except:
                pass
        
        try:
            self.request.set_collection("SW_OPER_GNTA_1B")  # GNSS TEC
            self.request.set_products(
                measurements=["TEC"],
                auxiliaries=["Latitude", "Longitude"],
                sampling_step="PT10S"
            )
            
            data = self.request.get_between(
                start_time=start_time,
                end_time=end_time
            )
            
            df = data.as_dataframe()
            
            if df.empty:
                return pd.DataFrame()
            
            # Фильтруем по координатам
            df = df[
                (df['Latitude'] >= lat - 5) &
                (df['Latitude'] <= lat + 5) &
                (df['Longitude'] >= lon - 5) &
                (df['Longitude'] <= lon + 5)
            ]
            
            if df.empty:
                return pd.DataFrame()
            
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            logger.info(f"🛰️ Swarm TEC: получено {len(df)} записей")
            return df
            
        except Exception as e:
            logger.error(f"❌ Swarm TEC ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Собирает данные Swarm для события.
        Возвращает словарь с магнитными и TEC данными.
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        mag_df = self.fetch_magnetic_data(lat, lon, start, end)
        tec_df = self.fetch_tec_data(lat, lon, start, end)
        
        return {
            'magnetic': mag_df,
            'tec': tec_df,
            'start': start,
            'end': end,
            'lat': lat,
            'lon': lon
        }
    
    def detect_anomaly(self, df, threshold=2.0):
        """
        Обнаруживает аномалии в магнитных данных по Z-score.
        """
        if df.empty or 'F_anomaly' not in df.columns:
            return None
        
        z_scores = np.abs((df['F_anomaly'] - df['F_anomaly'].mean()) / df['F_anomaly'].std())
        anomalies = z_scores > threshold
        
        if anomalies.any():
            return {
                'has_anomaly': True,
                'max_z_score': z_scores.max(),
                'anomaly_count': anomalies.sum(),
                'anomaly_times': df.index[anomalies].tolist()
            }
        
        return {
            'has_anomaly': False,
            'max_z_score': z_scores.max(),
            'anomaly_count': 0
        }


if __name__ == "__main__":
    # Тест
    collector = SwarmCollector()
    
    # Тестовое событие (недавнее землетрясение)
    event_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    lat, lon = 35.0, 140.0  # Япония
    
    print(f"🛰️ Тест Swarm для ({lat}, {lon}) в {event_time}")
    data = collector.fetch_for_event(event_time, lat, lon)
    anomaly = collector.detect_anomaly(data['magnetic'])
    
    if anomaly:
        print(f"   Аномалия: {anomaly['has_anomaly']}, Z-max: {anomaly['max_z_score']:.2f}")
    else:
        print("   Данные не получены")
