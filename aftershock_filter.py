#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aftershock_filter.py — фильтрация афтершоков для LAIC-анализа
"""

import pandas as pd
import logging
from datetime import timedelta

# Настройка логгера
logger = logging.getLogger(__name__)


def filter_aftershocks(df, time_window_days=7, distance_km=50, mag_threshold=0.5):
    """
    Удаляет афтершоки из DataFrame.
    
    Аргументы:
        df: DataFrame с колонками 'time', 'latitude', 'longitude', 'magnitude'
        time_window_days: временное окно (дней) для поиска более сильных событий
        distance_km: радиус (км) для поиска в том же районе
        mag_threshold: на сколько магнитуда должна быть меньше главного толчка
    
    Возвращает:
        df_filtered: DataFrame без афтершоков
        df_aftershocks: DataFrame только с афтершоками
    """
    if df.empty:
        return df, pd.DataFrame()
    
    df_sorted = df.sort_values('time').reset_index(drop=True)
    is_aftershock = [False] * len(df_sorted)
    
    for i, row in df_sorted.iterrows():
        # Ищем более сильные события ДО этого
        for j in range(i):
            prev = df_sorted.iloc[j]
            # Проверяем время
            time_diff = (row['time'] - prev['time']).total_seconds() / 86400  # в днях
            if time_diff > time_window_days:
                continue
            
            # Проверяем расстояние (приблизительно, через гаверсинус)
            distance = haversine_distance(
                row['latitude'], row['longitude'],
                prev['latitude'], prev['longitude']
            )
            if distance > distance_km:
                continue
            
            # Проверяем магнитуду (предыдущее событие должно быть сильнее)
            if prev['magnitude'] - row['magnitude'] >= mag_threshold:
                is_aftershock[i] = True
                break
    
    df['is_aftershock'] = is_aftershock
    df_aftershocks = df[df['is_aftershock'] == True].copy()
    df_filtered = df[df['is_aftershock'] == False].copy()
    
    logger.info(f"🔍 Афтершоков: {len(df_aftershocks)} из {len(df)} событий")
    
    return df_filtered, df_aftershocks


def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние между двумя точками в км (приблизительно)."""
    import math
    R = 6371  # радиус Земли в км
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c
