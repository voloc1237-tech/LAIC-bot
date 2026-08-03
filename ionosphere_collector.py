#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC через gnssanalysis (IONEX)
Использует библиотеку gnssanalysis от Geoscience Australia.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import requests
from gnssanalysis.ionex import IonexMap

logger = logging.getLogger(__name__)


class IonosphereCollector:
    """
    Сборщик ионосферных данных (TEC) через IONEX-файлы.
    Использует gnssanalysis для парсинга.
    """
    
    # CDDIS FTP сервер для IONEX (требуется анонимный доступ)
    CDDIS_URL = "https://cddis.nasa.gov/archive/gnss/products/ionex"
    
    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {cache_dir}")
    
    def _get_ionex_filename(self, date, version='codg'):
        """
        Формирует имя IONEX-файла по дате.
        Формат: codgYYYYDDD.01i (например, codg2023001.01i)
        """
        day_of_year = date.timetuple().tm_yday
        year_short = str(date.year)[-2:]
        return f"{version}{year_short}{day_of_year:03d}.01i"
    
    def _download_ionex(self, date):
        """
        Скачивает IONEX-файл с CDDIS.
        """
        filename = self._get_ionex_filename(date)
        local_path = os.path.join(self.cache_dir, filename)
        
        # Проверяем кэш
        if os.path.exists(local_path):
            logger.debug(f"📂 IONEX файл в кэше: {filename}")
            return local_path
        
        # Формируем URL
        # IONEX файлы организованы по годам и дням
        year = date.strftime('%Y')
        day_of_year = date.timetuple().tm_yday
        url = f"{self.CDDIS_URL}/{year}/{day_of_year:03d}/{filename}"
        
        try:
            logger.info(f"📥 Скачивание IONEX: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ IONEX сохранён: {local_path}")
            return local_path
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️ IONEX файл не найден: {filename}")
            else:
                logger.warning(f"⚠️ Ошибка скачивания IONEX: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Ошибка скачивания IONEX: {e}")
            return None
    
    def fetch_tec_for_point(self, lat, lon, start_time, end_time):
        """
        Получить VTEC (вертикальный TEC) для указанной точки за период.
        Возвращает DataFrame с индексами времени и колонкой 'tec'.
        """
        if not isinstance(start_time, datetime):
            start_time = pd.to_datetime(start_time).to_pydatetime()
        if not isinstance(end_time, datetime):
            end_time = pd.to_datetime(end_time).to_pydatetime()
        
        # Для IONEX выбираем середину периода (или каждый день)
        current_date = start_time.replace(hour=12, minute=0, second=0)
        records = []
        
        while current_date <= end_time:
            # Скачиваем IONEX-файл для этой даты
            ionex_path = self._download_ionex(current_date)
            
            if ionex_path:
                try:
                    # Парсим IONEX
                    ionmap = IonexMap.from_file(ionex_path)
                    
                    # Получаем TEC для заданной точки
                    # IONEX обычно содержит карты каждый 2 часа
                    for hour in range(0, 24, 2):
                        time_query = current_date.replace(hour=hour, minute=0, second=0)
                        try:
                            tec_val = ionmap.get_tec(lat=lat, lon=lon, time=time_query)
                            if tec_val is not None and not np.isnan(tec_val):
                                records.append({
                                    'time': time_query,
                                    'tec': float(tec_val),
                                    'lat': lat,
                                    'lon': lon
                                })
                        except Exception as e:
                            logger.debug(f"Ошибка получения TEC для {time_query}: {e}")
                            continue
                    
                    logger.info(f"🛰️ TEC: {len(records)} записей для ({lat:.2f}, {lon:.2f}) на {current_date.date()}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка парсинга IONEX {ionex_path}: {e}")
            
            # Переход к следующему дню
            current_date += timedelta(days=1)
        
        if not records:
            logger.warning(f"⚠️ Нет TEC данных для ({lat:.2f}, {lon:.2f}) за период")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Интерполяция пропущенных значений (если есть)
        if df['tec'].isnull().any():
            df['tec'] = df['tec'].interpolate(method='time')
        
        logger.info(f"🛰️ TEC: {len(df)} записей для ({lat:.2f}, {lon:.2f})")
        return df
    
    def fetch_roti(self, lat, lon, start_time, end_time):
        """
        Получить ROTI (Rate of TEC Index) — показатель ионосферных возмущений.
        Вычисляется как скользящее стандартное отклонение TEC.
        """
        tec_df = self.fetch_tec_for_point(lat, lon, start_time, end_time)
        if tec_df.empty:
            return pd.DataFrame()
        
        # ROTI — стандартное отклонение TEC за окно (для часовых данных — окно 4 часа)
        tec_df['roti'] = tec_df['tec'].rolling(window=4, min_periods=2).std()
        
        logger.info(f"🛰️ ROTI: {len(tec_df)} записей")
        return tec_df[['roti']]
    
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """
        Собирает TEC и ROTI за период вокруг события.
        Возвращает словарь {'tec': df, 'roti': df}
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)
        
        tec_df = self.fetch_tec_for_point(lat, lon, start, end)
        roti_df = self.fetch_roti(lat, lon, start, end) if not tec_df.empty else pd.DataFrame()
        
        return {
            'tec': tec_df,
            'roti': roti_df
        }
