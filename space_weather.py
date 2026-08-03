#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — сбор данных космической погоды
Источники:
- Kp: NOAA SWPC (JSON) + резерв: GFZ Potsdam
- Dst: Kyoto WDC (текстовый архив)
- F10.7: NOAA SWPC (JSON) + резерв: LASP
"""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import io
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
        """Получить данные из кэша."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key, data):
        """Сохранить в кэш."""
        self.cache[key] = (data, time.time())
    
    # ═══════════════════════════════════════════════════════
    # 1. Kp-индекс (NOAA SWPC + GFZ Potsdam)
    # ═══════════════════════════════════════════════════════
    
    def fetch_kp(self, start_date, end_date):
        """
        Получить 3-часовые Kp-индексы.
        Основной источник: NOAA SWPC JSON.
        Резерв: GFZ Potsdam (если NOAA недоступен).
        """
        cache_key = f'kp_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ Kp: загружено из кэша ({len(cached)} записей)")
            return cached
        
        # Попытка 1: NOAA SWPC
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data[1:]:  # пропускаем заголовок
                try:
                    dt = datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S')
                    if start_date <= dt <= end_date:
                        kp_val = float(item[1]) if item[1] else None
                        if kp_val is not None:
                            records.append({'time': dt, 'kp': kp_val})
                except (ValueError, IndexError):
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ Kp (NOAA): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка NOAA Kp: {e}")
        
        # Попытка 2: GFZ Potsdam (резерв)
        try:
            # GFZ предоставляет Kp в текстовом формате
            url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            
            records = []
            for line in resp.text.splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    # Kp за каждый 3-часовой интервал (8 значений в сутки)
                    for i in range(8):
                        kp_val = float(parts[5 + i]) if parts[5 + i] else None
                        if kp_val is not None:
                            # Время: 00:00, 03:00, ... 21:00
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
    # 2. Dst-индекс (Kyoto WDC)
    # ═══════════════════════════════════════════════════════
    
    def fetch_dst(self, start_date, end_date):
        """
        Получить Dst-индекс (геомагнитная буря).
        Источник: Kyoto WDC (текстовый архив).
        """
        cache_key = f'dst_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"🧲 Dst: загружено из кэша ({len(cached)} записей)")
            return cached
        
        try:
            # Kyoto WDC предоставляет часовые данные
            # Формируем URL для запроса (за последний месяц для простоты)
            # Для полного архива используем прямой файл
            url = "https://wdc.kugi.kyoto-u.ac.jp/dst_final/2026/202608.dst"
            # В реальности нужно формировать URL по годам/месяцам
            # Пока используем упрощённый подход
            
            # Альтернатива: NOAA Dst (если Kyoto недоступен)
            url_noaa = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
            resp = self.session.get(url_noaa, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data[1:]:  # пропускаем заголовок
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
            logger.warning(f"⚠️ Ошибка Dst: {e}")
        
        logger.warning("⚠️ Dst данные не получены")
        return pd.DataFrame()

    def fetch_dst_kyoto(self, start_date, end_date):
        """Получить Dst из Kyoto WDC (альтернативный источник)."""
        try:
            # Используем архив Kyoto WDC в текстовом формате
            # (для примера – заглушка, нужно парсить реальный файл)
            url = "https://wdc.kugi.kyoto-u.ac.jp/dst_final/2026/202608.dst"
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                # Парсим текстовый файл
                records = []
                for line in resp.text.splitlines():
                    if not line.strip() or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            year = int(parts[0][:4])
                            day_of_year = int(parts[0][4:7])
                            hour = int(parts[0][7:9])
                            dst_val = float(parts[1]) if parts[1] else None
                            if dst_val is not None:
                                dt = datetime(year, 1, 1) + timedelta(days=day_of_year-1, hours=hour)
                                if start_date <= dt <= end_date:
                                    records.append({'time': dt, 'dst': dst_val})
                        except (ValueError, IndexError):
                            continue
                if records:
                    df = pd.DataFrame(records)
                    df.set_index('time', inplace=True)
                    return df
        except Exception as e:
            logger.warning(f"⚠️ Dst Kyoto: {e}")
        return pd.DataFrame()
    # ═══════════════════════════════════════════════════════
    # 3. F10.7 (солнечный поток)
    # ═══════════════════════════════════════════════════════
    
    def fetch_f107(self, start_date, end_date):
        """
        Получить ежедневный поток F10.7.
        Источник: NOAA SWPC (JSON).
        """
        cache_key = f'f107_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ F10.7: загружено из кэша ({len(cached)} записей)")
            return cached
        
        # Попытка 1: NOAA SWPC JSON
        url = "https://services.swpc.noaa.gov/products/solar-radio-flux.json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data[1:]:  # пропускаем заголовок
                try:
                    dt = datetime.strptime(item[0], '%Y-%m-%d')
                    if start_date <= dt <= end_date:
                        f107_val = float(item[1]) if item[1] else None
                        if f107_val is not None:
                            records.append({'time': dt, 'f107': f107_val})
                except (ValueError, IndexError):
                    continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (NOAA): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка F10.7 NOAA: {e}")
        
        # Попытка 2: LASP (университет Колорадо)
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
    # 4. Универсальный метод для всех данных
    # ═══════════════════════════════════════════════════════
    
    def fetch_all_for_period(self, start_date, end_date):
        """
        Возвращает словарь {'kp': df, 'dst': df, 'f107': df}
        для указанного периода.
        """
        return {
            'kp': self.fetch_kp(start_date, end_date),
            'dst': self.fetch_dst(start_date, end_date),
            'f107': self.fetch_f107(start_date, end_date)
        }
