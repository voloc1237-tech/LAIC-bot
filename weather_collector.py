#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_collector.py — сбор исторических погодных данных
Источники:
1. Open-Meteo (основной, без ключа)
2. WeatherAPI (резерв, требует ключа)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import os

logger = logging.getLogger(__name__)


class WeatherCollector:
    """
    Сборщик исторических данных о погоде с резервным источником.
    """
    
    # Open-Meteo (основной, бесплатный, без ключа)
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # WeatherAPI (резерв, бесплатный до 1 млн запросов/месяц)
    WEATHERAPI_URL = "https://api.weatherapi.com/v1/history.json"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LAIC-Bot/1.0'})
        
        # Ключ WeatherAPI из переменных окружения
        self.weatherapi_key = os.environ.get('WEATHERAPI_KEY', '')
        
        # Кэш для успешных ответов
        self.cache = {}
        
        logger.info("🌤️ WeatherCollector инициализирован")
        if self.weatherapi_key:
            logger.info("   ✅ WeatherAPI ключ найден")
        else:
            logger.info("   ℹ️ WeatherAPI ключ не найден, только Open-Meteo")
    
    def _get_cache_key(self, lat, lon, start_date, end_date):
        """Создаёт ключ для кэша."""
        return f"{lat:.2f}_{lon:.2f}_{start_date}_{end_date}"
    
    def fetch_weather(self, lat, lon, start_date, end_date):
        """
        Возвращает DataFrame с температурой и влажностью.
        Пытается получить данные из:
        1. Кэша
        2. Open-Meteo
        3. WeatherAPI (если есть ключ)
        """
        # Приводим даты к строкам
        if hasattr(start_date, 'strftime'):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = start_date
        
        if hasattr(end_date, 'strftime'):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = end_date
        
        # Проверка кэша
        cache_key = self._get_cache_key(lat, lon, start_str, end_str)
        if cache_key in self.cache:
            logger.debug(f"📂 Погода из кэша: {cache_key}")
            return self.cache[cache_key].copy()
        
        # Проверка координат
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.warning(f"⚠️ Некорректные координаты: ({lat}, {lon})")
            return pd.DataFrame()
        
        # 1. Open-Meteo (основной)
        df = self._fetch_open_meteo(lat, lon, start_str, end_str)
        if not df.empty:
            self.cache[cache_key] = df.copy()
            logger.info(f"🌡️ Open-Meteo: {len(df)} дней, средняя температура {df['temp_2m_mean'].mean():.1f}°C")
            return df
        
        # 2. WeatherAPI (резерв)
        if self.weatherapi_key:
            df = self._fetch_weatherapi(lat, lon, start_str, end_str)
            if not df.empty:
                self.cache[cache_key] = df.copy()
                logger.info(f"🌡️ WeatherAPI: {len(df)} дней, средняя температура {df['temp_2m_mean'].mean():.1f}°C")
                return df
        
        logger.warning(f"⚠️ Не удалось получить погоду для ({lat:.2f}, {lon:.2f})")
        return pd.DataFrame()
    
    def _fetch_open_meteo(self, lat, lon, start_date, end_date):
        """Запрос к Open-Meteo (основной источник)."""
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ['temperature_2m_mean', 'relative_humidity_2m_mean'],
            'timezone': 'UTC'
        }
        
        try:
            response = self.session.get(self.OPEN_METEO_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            if not daily or 'time' not in daily:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'temp_2m_mean': daily['temperature_2m_mean'],
                'humidity_2m_mean': daily['relative_humidity_2m_mean']
            })
            df.set_index('date', inplace=True)
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [502, 503, 504]:
                logger.warning(f"⚠️ Open-Meteo временно недоступен (HTTP {e.response.status_code})")
            else:
                logger.debug(f"⚠️ Open-Meteo ошибка: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"⚠️ Open-Meteo ошибка: {e}")
            return pd.DataFrame()
    
    def _fetch_weatherapi(self, lat, lon, start_date, end_date):
        """Запрос к WeatherAPI (резерв)."""
        if not self.weatherapi_key:
            return pd.DataFrame()
        
        # WeatherAPI требует по одному дню, поэтому собираем по дням
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end - start).days
            
            if days_diff > 15:
                logger.debug(f"⚠️ WeatherAPI: период {days_diff} дней > 15, пропускаем")
                return pd.DataFrame()
            
            records = []
            current = start
            
            while current <= end:
                date_str = current.strftime('%Y-%m-%d')
                params = {
                    'key': self.weatherapi_key,
                    'q': f"{lat},{lon}",
                    'dt': date_str
                }
                
                try:
                    response = self.session.get(self.WEATHERAPI_URL, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    
                    day_data = data.get('forecast', {}).get('forecastday', [])
                    if day_data:
                        day = day_data[0].get('day', {})
                        records.append({
                            'date': pd.to_datetime(date_str),
                            'temp_2m_mean': day.get('avgtemp_c', None),
                            'humidity_2m_mean': day.get('avghumidity', None)
                        })
                    
                except Exception as e:
                    logger.debug(f"⚠️ WeatherAPI ошибка для {date_str}: {e}")
                
                current += timedelta(days=1)
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            
            # Интерполяция пропусков
            if df['temp_2m_mean'].isnull().any():
                df['temp_2m_mean'] = df['temp_2m_mean'].interpolate(method='time')
            if df['humidity_2m_mean'].isnull().any():
                df['humidity_2m_mean'] = df['humidity_2m_mean'].interpolate(method='time')
            
            return df
            
        except Exception as e:
            logger.debug(f"⚠️ WeatherAPI ошибка: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Удобный метод для землетрясения: возвращает погоду за [days_before] дней до
        и [days_after] после события.
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        return self.fetch_weather(lat, lon, start, end)
