#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_collector.py — сбор данных миссии ESA Swarm
Для обнаружения магнитных и ионосферных предвестников землетрясений.
Поддерживает авторизацию через токен из переменной окружения VIRES_ACCESS_TOKEN.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

# Проверка наличия viresclient
try:
    from viresclient import SwarmRequest, set_token
    VIRES_AVAILABLE = True
    logger.info("✅ viresclient загружен")
except ImportError:
    VIRES_AVAILABLE = False
    logger.warning("⚠️ viresclient не установлен. Установите: pip install viresclient")


class SwarmCollector:
    """
    Сборщик данных миссии ESA Swarm.
    - Магнитное поле (MFV, F) — для обнаружения предвестников
    - TEC (GNSS) — замена традиционным IONEX-данным
    - FAST-продукты для данных в реальном времени
    """
    
    VIRES_URL = "https://vires.services/ows"
    
    # Коллекции для разных типов данных
    COLLECTIONS = {
        'magnetic_fast': 'SW_FAST_MAGx_LR_1B',    # FAST (почти реальное время)
        'magnetic_oper': 'SW_OPER_MAGx_LR_1B',     # OPER (высокое качество, задержка 3 дня)
        'tec_fast': 'SW_FAST_TECxTMS_2F',          # FAST TEC
        'tec_oper': 'SW_OPER_TECxTMS_2F',          # OPER TEC
    }
    
    def __init__(self, cache_dir="data/swarm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем токен из переменной окружения
        self.token = os.environ.get('VIRES_ACCESS_TOKEN')
        self.request = None
        
        if self.token:
            logger.info(f"✅ Swarm токен загружен из окружения (длина: {len(self.token)} символов)")
        else:
            logger.warning("⚠️ Swarm токен не найден в окружении (VIRES_ACCESS_TOKEN)")
            logger.warning("   Установите: export VIRES_ACCESS_TOKEN='ваш_токен'")
        
        self._init_request()
    
    def _init_request(self):
        """Инициализирует запрос к Swarm с токеном."""
        if not VIRES_AVAILABLE:
            logger.warning("⚠️ viresclient не установлен, Swarm недоступен")
            return
        
        if not self.token:
            logger.warning("⚠️ Токен не установлен, Swarm недоступен")
            return
        
        try:
            # Устанавливаем токен
            set_token(self.token, self.VIRES_URL)
            logger.info("🛰️ Swarm: токен установлен")
            
            self.request = SwarmRequest()
            logger.info("🛰️ Swarm: соединение установлено")
        except Exception as e:
            logger.warning(f"⚠️ Swarm: ошибка соединения: {e}")
            self.request = None
    
    def _get_cache_path(self, prefix, lat, lon, start_time, end_time):
        """Формирует путь к файлу кэша."""
        cache_key = f"{prefix}_{lat:.2f}_{lon:.2f}_{start_time.date()}_{end_time.date()}"
        return self.cache_dir / f"{cache_key}.pkl"
    
    def fetch_magnetic_data(self, lat, lon, start_time, end_time, radius_km=500, use_fast=True):
        """
        Получает магнитные данные (MFV) за период.
        Использует FAST-продукты для данных в реальном времени.
        
        Args:
            lat, lon: координаты
            start_time, end_time: период
            radius_km: радиус поиска
            use_fast: True для FAST (почти реальное время), False для OPER (высокое качество)
        
        Returns:
            DataFrame с магнитными данными и аномалиями
        """
        if not VIRES_AVAILABLE or self.request is None:
            logger.warning("⚠️ Swarm недоступен, возвращаю пустые данные")
            return pd.DataFrame()
        
        # Проверяем кэш
        cache_prefix = "swarm_mag_fast" if use_fast else "swarm_mag_oper"
        cache_path = self._get_cache_path(cache_prefix, lat, lon, start_time, end_time)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.info(f"📂 Swarm: загружено из кэша ({len(df)} записей)")
                    return df
            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения кэша: {e}")
        
        try:
            collection = self.COLLECTIONS['magnetic_fast'] if use_fast else self.COLLECTIONS['magnetic_oper']
            logger.info(f"🛰️ Swarm: запрос магнитных данных ({collection}) за {start_time.date()} — {end_time.date()}")
            
            # Настраиваем запрос
            self.request.set_collection(collection)
            self.request.set_products(
                measurements=["F", "B_NEC"],
                models=["CHAOS-Core"],  # Для вычисления аномалий
                auxiliaries=["QDLat", "QDLon", "Radius", "Spacecraft"],
                sampling_step="PT10S"
            )
            
            data = self.request.get_between(
                start_time=start_time,
                end_time=end_time
            )
            
            df = data.as_dataframe()
            
            if df.empty:
                logger.warning("⚠️ Swarm: нет данных за период")
                return pd.DataFrame()
            
            # Фильтруем по координатам (в квази-дипольных координатах)
            lat_range = radius_km / 111.0
            lon_range = radius_km / 111.0
            
            df_filtered = df[
                (df['QDLat'] >= lat - lat_range) &
                (df['QDLat'] <= lat + lat_range) &
                (df['QDLon'] >= lon - lon_range) &
                (df['QDLon'] <= lon + lon_range)
            ]
            
            if df_filtered.empty:
                logger.warning(f"⚠️ Swarm: нет данных в радиусе {radius_km} км")
                return pd.DataFrame()
            
            # Вычисляем аномалии
            if 'F_CHAOS-Core' in df_filtered.columns:
                df_filtered['F_anomaly'] = df_filtered['F'] - df_filtered['F_CHAOS-Core']
            else:
                df_filtered['F_anomaly'] = df_filtered['F'] - df_filtered['F'].mean()
            
            # Добавляем метаданные
            df_filtered['event_lat'] = lat
            df_filtered['event_lon'] = lon
            df_filtered['data_source'] = 'FAST' if use_fast else 'OPER'
            
            # Сохраняем в кэш
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df_filtered, f)
                logger.info(f"💾 Swarm: данные сохранены в кэш ({len(df_filtered)} записей)")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка сохранения кэша: {e}")
            
            logger.info(f"🛰️ Swarm: получено {len(df_filtered)} записей магнитных данных")
            return df_filtered
            
        except Exception as e:
            # Если FAST не сработал, пробуем OPER
            if use_fast:
                logger.warning(f"⚠️ FAST данные не получены, пробуем OPER: {e}")
                return self.fetch_magnetic_data(lat, lon, start_time, end_time, radius_km, use_fast=False)
            else:
                logger.error(f"❌ Swarm ошибка: {e}")
                return pd.DataFrame()
    
    def fetch_tec_data(self, lat, lon, start_time, end_time, use_fast=True):
        """
        Получает TEC из Swarm GNSS.
        """
        if not VIRES_AVAILABLE or self.request is None:
            return pd.DataFrame()
        
        cache_prefix = "swarm_tec_fast" if use_fast else "swarm_tec_oper"
        cache_path = self._get_cache_path(cache_prefix, lat, lon, start_time, end_time)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.info(f"📂 Swarm TEC: загружено из кэша ({len(df)} записей)")
                    return df
            except:
                pass
        
        try:
            collection = self.COLLECTIONS['tec_fast'] if use_fast else self.COLLECTIONS['tec_oper']
            logger.info(f"🛰️ Swarm: запрос TEC ({collection}) за {start_time.date()} — {end_time.date()}")
            
            self.request.set_collection(collection)
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
            df_filtered['data_source'] = 'FAST' if use_fast else 'OPER'
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df_filtered, f)
            except:
                pass
            
            logger.info(f"🛰️ Swarm TEC: получено {len(df_filtered)} записей")
            return df_filtered
            
        except Exception as e:
            if use_fast:
                logger.warning(f"⚠️ FAST TEC не получен, пробуем OPER: {e}")
                return self.fetch_tec_data(lat, lon, start_time, end_time, use_fast=False)
            else:
                logger.error(f"❌ Swarm TEC ошибка: {e}")
                return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3, use_fast=True):
        """
        Собирает данные Swarm для события.
        Возвращает словарь с магнитными и TEC данными.
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        mag_df = self.fetch_magnetic_data(lat, lon, start, end, use_fast=use_fast)
        tec_df = self.fetch_tec_data(lat, lon, start, end, use_fast=use_fast)
        
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
        """
        Обнаруживает аномалии в магнитных данных по Z-score.
        Использует метод, описанный в научных работах по LAIC.
        """
        if df.empty or 'F_anomaly' not in df.columns:
            return {
                'has_anomaly': False,
                'max_z_score': 0.0,
                'anomaly_count': 0,
                'message': 'Нет данных или колонки F_anomaly'
            }
        
        try:
            # Z-score на основе аномалий
            z_scores = np.abs((df['F_anomaly'] - df['F_anomaly'].mean()) / df['F_anomaly'].std())
            anomalies = z_scores > threshold
            
            if anomalies.any():
                return {
                    'has_anomaly': True,
                    'max_z_score': float(z_scores.max()),
                    'anomaly_count': int(anomalies.sum()),
                    'anomaly_times': df.index[anomalies].tolist(),
                    'message': f"Обнаружено {anomalies.sum()} аномалий"
                }
            else:
                return {
                    'has_anomaly': False,
                    'max_z_score': float(z_scores.max()),
                    'anomaly_count': 0,
                    'message': 'Аномалий не обнаружено'
                }
        except Exception as e:
            return {
                'has_anomaly': False,
                'max_z_score': 0.0,
                'anomaly_count': 0,
                'message': f'Ошибка: {e}'
            }
    
    def get_status(self):
        """Возвращает статус подключения к Swarm."""
        return {
            'available': VIRES_AVAILABLE,
            'token_set': bool(self.token),
            'request_ready': self.request is not None
        }


if __name__ == "__main__":
    # Тест
    logging.basicConfig(level=logging.INFO)
    
    collector = SwarmCollector()
    status = collector.get_status()
    print(f"🛰️ Статус Swarm: {status}")
    
    # Тестовое событие
    event_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    lat, lon = 35.0, 140.0  # Япония
    
    print(f"\n🛰️ Тест Swarm для ({lat}, {lon}) в {event_time}")
    data = collector.fetch_for_event(event_time, lat, lon, days_before=3, days_after=1, use_fast=True)
    
    print(f"   Магнитных данных: {len(data['magnetic'])}")
    print(f"   TEC данных: {len(data['tec'])}")
    
    if not data['magnetic'].empty:
        anomaly = collector.detect_anomaly(data['magnetic'])
        print(f"   Аномалия: {anomaly['has_anomaly']}")
        if anomaly['has_anomaly']:
            print(f"   Z-max: {anomaly['max_z_score']:.2f}σ")
