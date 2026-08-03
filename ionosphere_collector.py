#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через публичный API SIMuRG
(альтернатива QuakeION, не требует установки дополнительных библиотек)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

class IonosphereCollector:
    """
    Сборщик ионосферных данных через API SIMuRG (https://simurg.space/).
    Предоставляет TEC вариации (2–10, 10–20, 20–60 минут) и ROTI.
    """

    BASE_URL = "https://simurg.space/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LAIC-Bot/1.0'})

    def fetch_tec_variations(self, lat, lon, start_time, end_time):
        """
        Получить TEC вариации (в TECU) для заданной точки и периода.
        Возвращает DataFrame с колонками: time, tec_2_10, tec_10_20, tec_20_60
        """
        params = {
            'lat': lat,
            'lon': lon,
            'start': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'format': 'json'
        }
        url = f"{self.BASE_URL}/tec"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                logger.warning("⚠️ Нет данных TEC от SIMuRG")
                return pd.DataFrame()
            
            # Парсим ответ (ожидаем список словарей с 'time', 'tec_2_10', 'tec_10_20', 'tec_20_60')
            records = []
            for item in data['data']:
                records.append({
                    'time': datetime.fromisoformat(item['time'].replace('Z', '+00:00')),
                    'tec_2_10': item.get('tec_2_10'),
                    'tec_10_20': item.get('tec_10_20'),
                    'tec_20_60': item.get('tec_20_60')
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                logger.info(f"🛰️ SIMuRG TEC: {len(df)} записей")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка SIMuRG TEC: {e}")
            return pd.DataFrame()

    def fetch_roti(self, lat, lon, start_time, end_time):
        """
        Получить ROTI (Rate of TEC Index) для заданной точки и периода.
        """
        params = {
            'lat': lat,
            'lon': lon,
            'start': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'format': 'json'
        }
        url = f"{self.BASE_URL}/roti"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                logger.warning("⚠️ Нет данных ROTI от SIMuRG")
                return pd.DataFrame()
            
            records = []
            for item in data['data']:
                records.append({
                    'time': datetime.fromisoformat(item['time'].replace('Z', '+00:00')),
                    'roti': item.get('roti')
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                logger.info(f"🛰️ SIMuRG ROTI: {len(df)} записей")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка SIMuRG ROTI: {e}")
            return pd.DataFrame()

    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Собирает TEC вариации и ROTI за период вокруг события.
        Возвращает словарь {'tec': df, 'roti': df}
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        tec_df = self.fetch_tec_variations(lat, lon, start, end)
        roti_df = self.fetch_roti(lat, lon, start, end)
        
        return {'tec': tec_df, 'roti': roti_df}
