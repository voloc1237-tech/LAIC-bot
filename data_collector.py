#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_collector.py — сбор данных USGS с кэшированием
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import logging

from utils import load_cache, save_cache, ensure_dir

logger = logging.getLogger(__name__)


class USGSCollector:
    """
    Сборщик данных USGS с интеллектуальным кэшированием.
    """
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self, cache_enabled=True, cache_max_age_hours=6):
        self.cache_enabled = cache_enabled
        self.cache_max_age = cache_max_age_hours
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LAIC-Bot/1.0 (Research; GitHub Actions)'
        })
    
    def _make_cache_key(self, params):
        """Создаёт ключ кэша из параметров запроса."""
        sorted_params = sorted(params.items())
        return 'usgs_' + '_'.join(f"{k}={v}" for k, v in sorted_params)
    
    def fetch(self, start_time, end_time, min_magnitude=4.0,
              min_lat=None, max_lat=None, min_lon=None, max_lon=None):
        """
        Качает землетрясения с USGS. Использует кэш если возможно.
        """
        
        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'minmagnitude': min_magnitude,
            'orderby': 'time-asc',
            'limit': 20000
        }
        
        if min_lat is not None:
            params.update({
                'minlatitude': min_lat,
                'maxlatitude': max_lat,
                'minlongitude': min_lon,
                'maxlongitude': max_lon
            })
        
        cache_key = self._make_cache_key(params)
        
        # Пробуем кэш
        if self.cache_enabled:
            cached = load_cache(cache_key, self.cache_max_age)
            if cached:
                logger.info(f"💾 КЭШ: использованы кэшированные данные ({len(cached)} событий)")
                return pd.DataFrame(cached)
        
        # Запрос к USGS
        logger.info(f"🌐 USGS запрос: {start_time.date()} — {end_time.date()}, M≥{min_magnitude}")
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Ошибка USGS: {e}")
            # Пробуем старый кэш даже если просрочен
            cached = load_cache(cache_key, max_age_hours=168)  # 7 дней
            if cached:
                logger.warning("⚠️ Использован ПРОСРОЧЕННЫЙ кэш")
                return pd.DataFrame(cached)
            return pd.DataFrame()
        
        # Парсим ответ
        data = response.json()
        features = data.get('features', [])
        
        if not features:
            logger.info("ℹ️ USGS: нет событий")
            return pd.DataFrame()
        
        # Преобразуем в записи
        records = []
        for f in features:
            props = f['properties']
            coords = f['geometry']['coordinates']
            
            records.append({
                'id': f['id'],
                'time': props['time'],  # timestamp в ms
                'latitude': coords[1],
                'longitude': coords[0],
                'depth_km': coords[2],
                'magnitude': props['mag'],
                'mag_type': props.get('magType', 'unknown'),
                'place': props.get('place', 'Unknown'),
                'status': props.get('status', ''),
                'tsunami': bool(props.get('tsunami', 0)),
                'alert': props.get('alert'),
                'felt': props.get('felt'),
                'cdi': props.get('cdi'),
                'mmi': props.get('mmi'),
                'significance': props.get('sig', 0),
                'nst': props.get('nst'),
                'dmin': props.get('dmin'),
                'rms': props.get('rms'),
                'gap': props.get('gap'),
                'url': props.get('url', '')
            })
        
        # Сохраняем в кэш
        if self.cache_enabled:
            save_cache(cache_key, records)
            logger.info(f"💾 КЭШ сохранён: {len(records)} событий")
        
        # Создаём DataFrame
        df = pd.DataFrame(records)
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        
        logger.info(f"✅ USGS: получено {len(df)} событий")
        return df.sort_values('time').reset_index(drop=True)

    # Добавьте в класс USGSCollector:

    def fetch_global(self, start_time, end_time, min_magnitude=4.5):
        """
        Собирает ВСЕ землетрясения по всему миру за период.
        Без ограничений по широте/долготе.
        """
        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'minmagnitude': min_magnitude,
            'orderby': 'time-asc',
            'limit': 20000
        }
    
        cache_key = self._make_cache_key(params)
    
        # Кэш
        if self.cache_enabled:
            cached = load_cache(cache_key, self.cache_max_age)
            if cached is not None:
                logger.info(f"💾 КЭШ: глобальные данные ({len(cached)} событий)")
                return pd.DataFrame(cached)
    
        logger.info(f"🌍 Глобальный USGS запрос: {start_time.date()} — {end_time.date()}, M≥{min_magnitude}")
    
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Ошибка USGS: {e}")
            cached = load_cache(cache_key, max_age_hours=168)
            if cached is not None:
                logger.warning("⚠️ Использован просроченный кэш")
                return pd.DataFrame(cached)
            return pd.DataFrame()
    
        data = response.json()
        features = data.get('features', [])
        if not features:
            return pd.DataFrame()
    
        # Преобразуем в записи (как в существующем fetch)
        records = []
        for f in features:
            props = f['properties']
            coords = f['geometry']['coordinates']
            records.append({
                'id': f['id'],
                'time': props['time'],
                'latitude': coords[1],
                'longitude': coords[0],
                'depth_km': coords[2],
                'magnitude': props['mag'],
                'mag_type': props.get('magType', 'unknown'),
                'place': props.get('place', 'Unknown'),
                'status': props.get('status', ''),
                'tsunami': bool(props.get('tsunami', 0)),
                'alert': props.get('alert'),
                'felt': props.get('felt'),
                'cdi': props.get('cdi'),
                'mmi': props.get('mmi'),
                'significance': props.get('sig', 0),
                'nst': props.get('nst'),
                'dmin': props.get('dmin'),
                'rms': props.get('rms'),
                'gap': props.get('gap'),
                'url': props.get('url', '')
            })
    
        if self.cache_enabled:
            save_cache(cache_key, records)
    
        df = pd.DataFrame(records)
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        logger.info(f"✅ USGS: получено {len(df)} глобальных событий")
        return df.sort_values('time').reset_index(drop=True)
    
    def fetch_for_region(self, region_config, days_back=30):
        """Сбор данных для конкретного региона."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        
        return self.fetch(
            start, end,
            min_magnitude=region_config.get('min_magnitude', 4.0),
            min_lat=region_config.get('min_lat'),
            max_lat=region_config.get('max_lat'),
            min_lon=region_config.get('min_lon'),
            max_lon=region_config.get('max_lon')
        )

def collect_all_regions(settings, global_mode=True):
    """
    global_mode=True: собирает события по всему миру (без регионов)
    global_mode=False: старый режим по регионам из settings
    """
    collector = USGSCollector(
        cache_enabled=settings.get('cache', {}).get('enabled', True),
        cache_max_age_hours=settings.get('cache', {}).get('max_age_hours', 6)
    )
    
    results = {}
    
    if global_mode:
        # Глобальный сбор
        min_mag = settings.get('analysis', {}).get('global_min_magnitude', 4.5)
        days_back = settings['analysis']['history_days']
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        df = collector.fetch_global(start, end, min_magnitude=min_mag)
        if not df.empty:
            # Добавляем искусственный регион 'global'
            df['region_key'] = 'global'
            df['region_name'] = '🌍 Global'
            results['global'] = df
        else:
            results['global'] = pd.DataFrame()
    else:
        # Старый региональный режим
        for key, config in settings['regions'].items():
            df = collector.fetch_for_region(
                config,
                days_back=settings['analysis']['history_days']
            )
            if not df.empty:
                df['region_key'] = key
                df['region_name'] = config['name']
            results[key] = df
    
    total = sum(len(df) for df in results.values())
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 ИТОГО: {total} событий по {len(results)} регионам")
    return results

  
