#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_collector.py — сбор исторических погодных данных через Open-Meteo
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

class WeatherCollector:
    """
    Сборщик исторических данных о погоде с Open-Meteo.
    Бесплатный, без ключа, данные с 1940 года.
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LAIC-Bot/1.0'})
    
    def fetch_weather(self, lat, lon, start_date, end_date):
        """
        Возвращает DataFrame с температурой (2m) и влажностью (2m) за период.
        start_date, end_date: datetime или строки 'YYYY-MM-DD'
        """
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date,
            'end_date': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date,
            'daily': ['temperature_2m_mean', 'relative_humidity_2m_mean'],
            'timezone': 'UTC'
        }
        
        logger.debug(f"🌦️ Запрос погоды для ({lat}, {lon}) с {params['start_date']} по {params['end_date']}")
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            if not daily:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'temp_2m_mean': daily['temperature_2m_mean'],
                'humidity_2m_mean': daily['relative_humidity_2m_mean']
            })
            df.set_index('date', inplace=True)
            logger.info(f"🌡️ Погода: {len(df)} дней, средняя температура {df['temp_2m_mean'].mean():.1f}°C")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка Open-Meteo: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Удобный метод для землетрясения: возвращает погоду за [days_before] дней до
        и [days_after] после события.
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        return self.fetch_weather(lat, lon, start, end)
