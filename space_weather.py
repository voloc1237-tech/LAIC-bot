#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — сбор данных космической погоды
Источники:
- Kp: GFZ Potsdam (основной, работает стабильно)
- Dst: NOAA SWPC (JSON)
- F10.7: LASP (университет Колорадо)
"""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import logging
import time

logger = logging.getLogger(__name__)


class SpaceWeatherCollector:
    """Сборщик данных о космической погоде."""
    
    def __init__(self, cache_ttl=3600):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LAIC-Bot/1.0'})
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def _get_cache(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key, data):
        self.cache[key] = (data, time.time())
    
    # ═══════════════════════════════════════════════════════
    # 1. Kp-индекс (GFZ Potsdam — основной, стабильный)
    # ═══════════════════════════════════════════════════════
    
    def fetch_kp(self, start_date, end_date):
        """Получить 3-часовые Kp-индексы из архива GFZ Potsdam."""
        cache_key = f'kp_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ Kp: из кэша ({len(cached)} записей)")
            return cached
        
        try:
            url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            
            records = []
            for line in resp.text.splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 14:  # минимум для Kp данных
                    continue
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    for i in range(8):
                        kp_val = float(parts[5 + i]) if parts[5 + i] else None
                        if kp_val is not None:
                            hour = i * 3
                            dt = datetime(year, month, day, hour, 0, tzinfo=timezone.utc)
                            if start_date <= dt <= end_date:
                                records.append({'time': dt, 'kp': kp_val})
                except (ValueError, IndexError):
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ Kp (GFZ): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка GFZ Kp: {e}")
        
        logger.warning("⚠️ Kp данные не получены")
        return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════
    # 2. Dst-индекс (NOAA SWPC)
    # ═══════════════════════════════════════════════════════
    
    def fetch_dst(self, start_date, end_date):
        """Получить Dst-индекс из NOAA SWPC."""
        cache_key = f'dst_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"🧲 Dst: из кэша ({len(cached)} записей)")
            return cached
        
        try:
            url = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data[1:]:
                try:
                    dt = datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S')
                    if start_date <= dt <= end_date:
                        dst_val = float(item[1]) if item[1] else None
                        if dst_val is not None:
                            records.append({'time': dt, 'dst': dst_val})
                except (ValueError, IndexError):
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"🧲 Dst (NOAA): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка Dst NOAA: {e}")
        
        logger.warning("⚠️ Dst данные не получены")
        return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════
    # 3. F10.7 (LASP — университет Колорадо)
    # ═══════════════════════════════════════════════════════
    
    def fetch_f107(self, start_date, end_date):
        """Получить ежедневный поток F10.7 из архива LASP."""
        cache_key = f'f107_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ F10.7: из кэша ({len(cached)} записей)")
            return cached
        
        try:
            url = "https://lasp.colorado.edu/lisird/data/570_daily.txt"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            
            records = []
            for line in resp.text.splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    year = int(parts[0])
                    day_of_year = int(parts[1])
                    f107_val = float(parts[2]) if parts[2] else None
                    if f107_val is not None:
                        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
                        if start_date <= dt <= end_date:
                            records.append({'time': dt, 'f107': f107_val})
                except (ValueError, IndexError):
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (LASP): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка F10.7 LASP: {e}")
        
        logger.warning("⚠️ F10.7 данные не получены")
        return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════
    # 4. Универсальный метод
    # ═══════════════════════════════════════════════════════
    
    def fetch_all_for_period(self, start_date, end_date):
        """Возвращает {'kp': df, 'dst': df, 'f107': df}."""
        return {
            'kp': self.fetch_kp(start_date, end_date),
            'dst': self.fetch_dst(start_date, end_date),
            'f107': self.fetch_f107(start_date, end_date)
        }
