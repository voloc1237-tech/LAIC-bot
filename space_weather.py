#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — сбор данных космической погоды
Источники:
- Kp: GFZ Potsdam (основной) + NOAA SWPC (резерв)
- Dst: Kyoto WDC (основной) + NOAA SWPC (резерв)
- F10.7: GFZ (основной) + LASP (резерв) + NOAA NCEI (резерв)
"""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import logging
import time

logger = logging.getLogger(__name__)


class SpaceWeatherCollector:
    """Сборщик данных о космической погоде с резервными источниками."""
    
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
    # 1. Kp-индекс (GFZ + NOAA SWPC резерв)
    # ═══════════════════════════════════════════════════════
    
    def fetch_kp(self, start_date, end_date):
        """
        Получить 3-часовые Kp-индексы.
        Источник: GFZ (основной) → NOAA SWPC (резерв)
        """
        cache_key = f'kp_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ Kp: из кэша ({len(cached)} записей)")
            return cached
        
        # ---- ИСТОЧНИК 1: GFZ (основной) ----
        try:
            url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            
            records = []
            for line in resp.text.splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 14:
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
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ Kp (GFZ): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ GFZ Kp ошибка: {e}")
        
        # ---- ИСТОЧНИК 2: NOAA SWPC (резерв) ----
        try:
            url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data:
                if 'time_tag' not in item or 'kp' not in item:
                    continue
                dt = datetime.strptime(item['time_tag'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                if start_date <= dt <= end_date:
                    kp_val = float(item['kp']) if item['kp'] is not None else None
                    if kp_val is not None:
                        records.append({'time': dt, 'kp': kp_val})
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ Kp (NOAA SWPC): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ NOAA Kp ошибка: {e}")
        
        logger.warning("⚠️ Kp данные не получены")
        return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════
    # 2. Dst-индекс (Kyoto WDC + NOAA SWPC резерв)
    # ═══════════════════════════════════════════════════════
    
    def fetch_dst(self, start_date, end_date):
        """
        Получить Dst-индекс.
        Источник: Kyoto WDC (основной) → NOAA SWPC (резерв)
        """
        cache_key = f'dst_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"🧲 Dst: из кэша ({len(cached)} записей)")
            return cached
        
        # ---- ИСТОЧНИК 1: Kyoto WDC (основной) ----
        try:
            year = start_date.strftime('%Y')
            month = start_date.strftime('%m')
            
            # Пробуем provisional, затем final
            for url_template in [
                f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month}/{year}{month}.dst",
                f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}/{year}{month}.dst"
            ]:
                try:
                    resp = self.session.get(url_template, timeout=30)
                    if resp.status_code != 200:
                        continue
                    
                    records = []
                    for line in resp.text.splitlines():
                        if not line.strip() or line.startswith('#'):
                            continue
                        try:
                            year_str = line[2:6].strip()
                            day_of_year = int(line[7:10].strip())
                            hour = int(line[11:13].strip())
                            if not year_str:
                                continue
                            year = int(year_str)
                            dst_values = []
                            for i in range(24):
                                pos = 20 + i * 4
                                val_str = line[pos:pos+4].strip()
                                if val_str and val_str != '9999':
                                    dst_values.append(float(val_str))
                            for i, dst_val in enumerate(dst_values):
                                if dst_val is not None:
                                    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1, hours=i)
                                    if start_date <= dt <= end_date:
                                        records.append({'time': dt, 'dst': dst_val})
                        except (ValueError, IndexError):
                            continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df.set_index('time', inplace=True)
                        df.sort_index(inplace=True)
                        self._set_cache(cache_key, df)
                        logger.info(f"🧲 Dst (Kyoto): {len(df)} записей")
                        return df
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"⚠️ Kyoto Dst ошибка: {e}")
        
        # ---- ИСТОЧНИК 2: NOAA SWPC (резерв) ----
        try:
            url = "https://services.swpc.noaa.gov/json/kyoto-dst.json"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data[1:]:  # пропускаем заголовок
                if len(item) < 2:
                    continue
                time_str = item[0]
                dst_val = float(item[1]) if item[1] else None
                if dst_val is None:
                    continue
                dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if start_date <= dt <= end_date:
                    records.append({'time': dt, 'dst': dst_val})
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"🧲 Dst (NOAA SWPC): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ NOAA Dst ошибка: {e}")
        
        logger.warning("⚠️ Dst данные не получены")
        return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════
    # 3. F10.7 (GFZ + LASP + NOAA NCEI)
    # ═══════════════════════════════════════════════════════
    
    def fetch_f107(self, start_date, end_date):
        """
        Получить ежедневный поток F10.7.
        Источники: GFZ (основной) → LASP (резерв) → NOAA NCEI (резерв)
        """
        cache_key = f'f107_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ F10.7: из кэша ({len(cached)} записей)")
            return cached
        
        # ---- ИСТОЧНИК 1: GFZ (основной) ----
        try:
            url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            
            records = []
            for line in resp.text.splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 13:
                    continue
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    f107_val = float(parts[12]) if parts[12] else None
                    if f107_val is not None and f107_val > 0:
                        dt = datetime(year, month, day, tzinfo=timezone.utc)
                        if start_date <= dt <= end_date:
                            records.append({'time': dt, 'f107': f107_val})
                except (ValueError, IndexError):
                    continue
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (GFZ): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ GFZ F10.7 ошибка: {e}")
        
        # ---- ИСТОЧНИК 2: LASP (резерв) ----
        try:
            url = "https://lasp.colorado.edu/lisird/data/570_daily.txt"
            resp = self.session.get(url, timeout=60)
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
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (LASP): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ LASP F10.7 ошибка: {e}")
        
        # ---- ИСТОЧНИК 3: NOAA NCEI (последний резерв) ----
        try:
            url = "https://services.swpc.noaa.gov/json/solar-radio-flux/f10.7.json"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data:
                if 'time_tag' not in item or 'f10.7' not in item:
                    continue
                dt = datetime.strptime(item['time_tag'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if start_date <= dt <= end_date:
                    f107_val = float(item['f10.7']) if item['f10.7'] is not None else None
                    if f107_val is not None:
                        records.append({'time': dt, 'f107': f107_val})
            
            if records:
                df = pd.DataFrame(records)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (NOAA NCEI): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ NOAA NCEI F10.7 ошибка: {e}")
        
        logger.warning("⚠️ F10.7 данные не получены ни из одного источника")
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
