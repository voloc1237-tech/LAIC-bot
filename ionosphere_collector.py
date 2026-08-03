#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через tec.io (NASA CDDIS)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import os
import tempfile
from tec import Tec, download_ionex

logger = logging.getLogger(__name__)


class IonosphereCollector:
    """
    Сборщик ионосферных данных через NASA CDDIS.
    Использует библиотеку tec.io для парсинга IONEX-файлов.
    """
    
    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {cache_dir}")
    
    def fetch_tec_for_point(self, lat, lon, start_time, end_time, sampling='hourly'):
        """
        Получить VTEC (вертикальный TEC) для указанной точки за период.
        Возвращает DataFrame с индексами времени и колонкой 'tec'.
        """
        try:
            # Скачиваем IONEX-файлы за период
            # tec.io умеет скачивать файлы с NASA CDDIS
            ionex_files = download_ionex(
                start=start_time,
                end=end_time,
                output_dir=self.cache_dir,
                source='cddis'
            )
            
            if not ionex_files:
                logger.warning("⚠️ IONEX-файлы не найдены")
                return pd.DataFrame()
            
            records = []
            for file_path in ionex_files:
                try:
                    # Парсим IONEX-файл
                    tec_data = Tec(file_path)
                    
                    # Получаем TEC в указанной точке
                    # tec.io возвращает TEC в TECU (1 TECU = 10^16 эл/м²)
                    tec_value = tec_data.get_value(lat, lon)
                    
                    # Время из имени файла или из данных
                    # Извлекаем дату из имени файла (например, igs12345.20i)
                    date_str = os.path.basename(file_path).split('.')[0][3:8]
                    year = int(f"20{date_str[4:6]}")  # 20 + последние 2 цифры
                    day_of_year = int(date_str[0:3])
                    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
                    
                    if tec_value is not None:
                        records.append({
                            'time': dt,
                            'tec': tec_value,
                            'lat': lat,
                            'lon': lon
                        })
                
                except Exception as e:
                    logger.debug(f"Ошибка парсинга {file_path}: {e}")
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                df = df.resample('1h').mean() if sampling == 'hourly' else df
                logger.info(f"🛰️ TEC: {len(df)} записей для ({lat:.2f}, {lon:.2f})")
                return df
            
            return pd.DataFrame()
            
        except ImportError:
            logger.warning("⚠️ Библиотека 'tec' не установлена. Установите: pip install tec")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка TEC: {e}")
            return pd.DataFrame()
    
    def fetch_roti(self, lat, lon, start_time, end_time):
        """
        Получить ROTI (Rate of TEC Index) — показатель ионосферных возмущений.
        ROTI вычисляется как стандартное отклонение TEC за 15-минутные окна.
        """
        tec_df = self.fetch_tec_for_point(lat, lon, start_time, end_time, sampling='hourly')
        if tec_df.empty:
            return pd.DataFrame()
        
        # ROTI = скользящее стандартное отклонение TEC за 15 минут
        # При часовых данных используем окно в 4 точки (1 час)
        # Для более точного ROTI нужны данные с частотой 30-60 секунд
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
