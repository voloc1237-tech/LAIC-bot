#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через QuakeION
"""

import pandas as pd
from quakeion import get_tec_data, get_roti_data
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class IonosphereCollector:
    """
    Сборщик данных ионосферы (TEC, ROTI) с помощью библиотеки QuakeION.
    Требуется интернет для загрузки GNSS-файлов.
    """
    
    def __init__(self, sampling='hourly'):
        self.sampling = sampling  # 'hourly' или 'daily'
    
    def fetch_tec(self, lat, lon, start_time, end_time, station='auto'):
        """
        Получить TEC (Total Electron Content) в указанной точке за период.
        station: 'auto' — автоматический выбор ближайшей станции.
        """
        logger.info(f"🛰️ Запрос TEC для ({lat:.2f}, {lon:.2f}) с {start_time} по {end_time}")
        try:
            # QuakeION возвращает DataFrame с datetime индексом и столбцом 'tec'
            df_tec = get_tec_data(
                lat=lat,
                lon=lon,
                start_time=start_time,
                end_time=end_time,
                sampling=self.sampling,
                station=station
            )
            if df_tec is None or df_tec.empty:
                logger.warning("⚠️ TEC данные не получены")
                return pd.DataFrame()
            logger.info(f"✅ TEC: {len(df_tec)} записей")
            return df_tec
        except Exception as e:
            logger.error(f"❌ Ошибка TEC: {e}")
            return pd.DataFrame()
    
    def fetch_roti(self, lat, lon, start_time, end_time, station='auto'):
        """
        Получить ROTI (Rate of TEC Index) – показатель ионосферных возмущений.
        """
        logger.info(f"🛰️ Запрос ROTI для ({lat:.2f}, {lon:.2f})")
        try:
            df_roti = get_roti_data(
                lat=lat,
                lon=lon,
                start_time=start_time,
                end_time=end_time,
                sampling=self.sampling,
                station=station
            )
            if df_roti is None or df_roti.empty:
                logger.warning("⚠️ ROTI данные не получены")
                return pd.DataFrame()
            logger.info(f"✅ ROTI: {len(df_roti)} записей")
            return df_roti
        except Exception as e:
            logger.error(f"❌ Ошибка ROTI: {e}")
            return pd.DataFrame()
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Собирает TEC и ROTI за период вокруг события.
        Возвращает словарь {'tec': df, 'roti': df}
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        tec_df = self.fetch_tec(lat, lon, start, end)
        roti_df = self.fetch_roti(lat, lon, start, end)
        
        return {'tec': tec_df, 'roti': roti_df}
