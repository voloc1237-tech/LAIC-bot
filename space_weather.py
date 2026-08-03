#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — сбор индексов Kp, Dst, F10.7 с NOAA SWPC
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SpaceWeatherCollector:
    """
    Сборщик данных о солнечной и геомагнитной активности.
    Использует публичные JSON API NOAA SWPC.
    """
    
    BASE_URL = "https://services.swpc.noaa.gov/json"
    
    def __init__(self):
        self.session = requests.Session()
    
    def fetch_kp(self, start_date, end_date):
        """
        Получить 3-часовые Kp-индексы.
        start_date, end_date: строки 'YYYY-MM-DD'
        """
        # NOAA предоставляет архив по дням
        # Используем список за последние 30 дней, но можно расширить
        # Здесь используем предварительно собранный файл для упрощения
        url = f"{self.BASE_URL}/planetary_kp_archive.json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Парсим: каждый элемент содержит 'time_tag' и 'kp_index'
            records = []
            for item in data:
                dt = datetime.strptime(item['time_tag'], '%Y-%m-%d %H:%M:%S')
                if start_date <= dt <= end_date:
                    records.append({
                        'time': dt,
                        'kp': item['kp_index']
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                logger.info(f"☀️ Kp: {len(df)} записей")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка получения Kp: {e}")
            return pd.DataFrame()
    
    def fetch_dst(self, start_date, end_date):
        """
        Dst-индекс (геомагнитная буря) – из Kyoto или NOAA.
        У NOAA есть Dst в том же архиве.
        """
        # Используем альтернативный источник: NOAA Dst
        url = f"{self.BASE_URL}/dst_archive.json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for item in data:
                dt = datetime.strptime(item['time_tag'], '%Y-%m-%d %H:%M:%S')
                if start_date <= dt <= end_date:
                    records.append({
                        'time': dt,
                        'dst': item['dst']
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                logger.info(f"🧲 Dst: {len(df)} записей")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка получения Dst: {e}")
            return pd.DataFrame()
    
    def fetch_f107(self, start_date, end_date):
        """
        Солнечный поток F10.7 (ежедневный).
        """
        url = f"{self.BASE_URL}/f10_7_archive.json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for item in data:
                dt = datetime.strptime(item['time_tag'], '%Y-%m-%d')
                if start_date <= dt <= end_date:
                    records.append({
                        'time': dt,
                        'f107': item['f10_7']
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                logger.info(f"☀️ F10.7: {len(df)} записей")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка получения F10.7: {e}")
            return pd.DataFrame()
    
    def fetch_all_for_period(self, start_date, end_date):
        """
        Возвращает словарь {'kp': df, 'dst': df, 'f107': df}
        """
        return {
            'kp': self.fetch_kp(start_date, end_date),
            'dst': self.fetch_dst(start_date, end_date),
            'f107': self.fetch_f107(start_date, end_date)
        }
