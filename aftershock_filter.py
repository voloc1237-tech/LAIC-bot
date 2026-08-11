#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aftershock_filter.py — адаптивная фильтрация афтершоков для LAIC-анализа
Использует формулы Гарднера-Кнопова (1974) для динамического окна:
    время (дней) = 10^(0.032*M + 0.18)
    расстояние (км) = 10^(0.037*M + 1.77)
"""

import pandas as pd
import logging
from datetime import timedelta
import math

logger = logging.getLogger(__name__)


def _adaptive_window(magnitude: float):
    """
    Вычисляет адаптивные временное окно и радиус по магнитуде.
    Возвращает (time_window_days, distance_km).
    """
    if magnitude < 4.0:
        # Для слабых событий используем минимальные значения
        return 3.0, 30.0
    elif magnitude > 8.0:
        # Для очень сильных ограничиваем максимальные значения
        return 40.0, 200.0
    else:
        days = 10 ** (0.032 * magnitude + 0.18)
        km = 10 ** (0.037 * magnitude + 1.77)
        return round(days, 1), round(km, 1)


def filter_aftershocks(df, time_window_days=None, distance_km=None, mag_threshold=0.5, adaptive=True):
    """
    Удаляет афтершоки из DataFrame.

    Аргументы:
        df: DataFrame с колонками 'time', 'latitude', 'longitude', 'magnitude'
        time_window_days: фиксированное временное окно (если None и adaptive=True, вычисляется адаптивно)
        distance_km: фиксированный радиус (если None и adaptive=True, вычисляется адаптивно)
        mag_threshold: минимальная разница магнитуд, чтобы считать событие афтершоком
        adaptive: если True, использует адаптивные параметры в зависимости от магнитуды главного толчка

    Возвращает:
        df_filtered: DataFrame без афтершоков
        df_aftershocks: DataFrame только с афтершоками
    """
    if df.empty:
        return df, pd.DataFrame()

    df_sorted = df.sort_values('time').reset_index(drop=True)
    is_aftershock = [False] * len(df_sorted)

    # Если adaptive=True, вычисляем параметры для каждого потенциального главного события
    # В противном случае используем переданные фиксированные значения
    fixed_time_window = time_window_days
    fixed_distance = distance_km

    for i, row in df_sorted.iterrows():
        # Ищем более сильные события ДО этого
        for j in range(i):
            prev = df_sorted.iloc[j]
            mag_main = prev['magnitude']

            # Определяем параметры для этого главного события
            if adaptive:
                tw, dist = _adaptive_window(mag_main)
            else:
                tw = fixed_time_window if fixed_time_window is not None else 7.0
                dist = fixed_distance if fixed_distance is not None else 50.0

            # Проверяем время
            time_diff = (row['time'] - prev['time']).total_seconds() / 86400  # в днях
            if time_diff > tw:
                continue

            # Проверяем расстояние
            distance = haversine_distance(
                row['latitude'], row['longitude'],
                prev['latitude'], prev['longitude']
            )
            if distance > dist:
                continue

            # Проверяем магнитуду (предыдущее событие должно быть сильнее)
            if prev['magnitude'] - row['magnitude'] >= mag_threshold:
                is_aftershock[i] = True
                break

    df['is_aftershock'] = is_aftershock
    df_aftershocks = df[df['is_aftershock'] == True].copy()
    df_filtered = df[df['is_aftershock'] == False].copy()

    # Логируем результат с информацией о параметрах
    if adaptive:
        logger.info(f"🔍 Афтершоков: {len(df_aftershocks)} из {len(df)} событий (адаптивные параметры)")
    else:
        logger.info(f"🔍 Афтершоков: {len(df_aftershocks)} из {len(df)} событий (фиксированные: окно={fixed_time_window} дн, радиус={fixed_distance} км)")

    return df_filtered, df_aftershocks


def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние между двумя точками в км (приблизительно)."""
    R = 6371  # радиус Земли в км
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c
