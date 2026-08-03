#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — сбор данных космической погоды
Источники:
- Kp: GFZ Potsdam (основной, стабильный)
- Dst: WDC Kyoto (provisional)
- F10.7: GFZ Potsdam (вместе с Kp)
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
    # 1. Kp-индекс (GFZ Potsdam)
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
    # 2. Dst-индекс (WDC Kyoto — PROVISIONAL)
    # ═══════════════════════════════════════════════════════

    def fetch_dst(self, start_date, end_date):
        """
        Получить Dst-индекс.
        Сначала пробует текущий год (provisional), если нет — берёт прошлый год (final).
        """
        cache_key = f'dst_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"🧲 Dst: из кэша ({len(cached)} записей)")
            return cached

        # Попытка 1: текущий год (provisional)
        try:
            year = start_date.strftime('%Y')
            month = start_date.strftime('%m')
            url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month}/{year}{month}.dst"
        
            logger.debug(f"Запрос Dst (provisional): {url}")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        
            df = self._parse_dst_response(resp.text, start_date, end_date)
            if not df.empty:
                self._set_cache(cache_key, df)
                logger.info(f"🧲 Dst (Provisional): {len(df)} записей за {year}-{month}")
                return df
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.info(f"ℹ️ Dst (provisional) за {year}-{month} не найден, пробуем прошлый год")
            else:
                logger.warning(f"⚠️ Ошибка Dst provisional: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка Dst provisional: {e}")

        # Попытка 2: прошлый год (final) — сдвиг на 1 год
        try:
            # Сдвигаем даты на 1 год назад
            past_start = start_date - timedelta(days=365)
            past_end = end_date - timedelta(days=365)
        
            year = past_start.strftime('%Y')
            month = past_start.strftime('%m')
            url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}/{year}{month}.dst"
        
            logger.info(f"🔄 Используем Dst за прошлый год: {year}-{month} (аналог {start_date.strftime('%Y-%m')})")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        
            df = self._parse_dst_response(resp.text, past_start, past_end)
            if not df.empty:
                # Сдвигаем индекс времени на +1 год (чтобы совпадало с текущими датами)
                df.index = df.index + timedelta(days=365)
                self._set_cache(cache_key, df)
                logger.info(f"🧲 Dst (Final, прошлый год): {len(df)} записей за {year}-{month} → сдвинуто на +1 год")
                return df
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️ Dst за прошлый год {year}-{month} тоже не найден")
            else:
                logger.warning(f"⚠️ Ошибка Dst прошлый год: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка Dst прошлый год: {e}")

        logger.warning("⚠️ Dst данные не получены (ни текущий, ни прошлый год)")
        return pd.DataFrame()


    def _parse_dst_response(self, text, start_date, end_date):
        """Парсит ответ WDC Kyoto в DataFrame."""
        records = []
        for line in text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            try:
                # Формат: позиции 3-6 = год, 8-10 = день года, 12-13 = час
                # Позиции 21-116 = 24 значения по 4 символа
                year_str = line[2:6].strip()
                day_of_year = int(line[7:10].strip())
                hour = int(line[11:13].strip())
            
                if not year_str:
                    continue
                year = int(year_str)
            
                # Парсим 24 часовых значения Dst (с 21 по 116 позицию)
                dst_values = []
                for i in range(24):
                    pos = 20 + i * 4
                    val_str = line[pos:pos+4].strip()
                    if val_str and val_str != '9999':
                        dst_values.append(float(val_str))
                    else:
                        dst_values.append(None)
            
                # Добавляем каждое часовое значение как отдельную запись
                for i, dst_val in enumerate(dst_values):
                    if dst_val is not None:
                        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1, hours=i)
                        if start_date <= dt <= end_date:
                            records.append({'time': dt, 'dst': dst_val})
            except (ValueError, IndexError):
                continue
    
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('time', inplace=True)
        return df
    

                   

    # ═══════════════════════════════════════════════════════
    # 3. F10.7 (GFZ Potsdam — вместе с Kp)
    # ═══════════════════════════════════════════════════════

    def fetch_f107(self, start_date, end_date):
        """
        Получить ежедневный поток F10.7 из архива GFZ Potsdam.
        Использует тот же файл, что и для Kp.
        """
        cache_key = f'f107_{start_date}_{end_date}'
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"☀️ F10.7: из кэша ({len(cached)} записей)")
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
                if len(parts) < 13:
                    continue
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    # Столбец 13 — это F10.7 (индекс 12 в списке)
                    f107_val = float(parts[12]) if parts[12] else None
                    if f107_val is not None and f107_val > 0:
                        dt = datetime(year, month, day, tzinfo=timezone.utc)
                        if start_date <= dt <= end_date:
                            records.append({'time': dt, 'f107': f107_val})
                except (ValueError, IndexError):
                    continue

            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('time', inplace=True)
                self._set_cache(cache_key, df)
                logger.info(f"☀️ F10.7 (GFZ): {len(df)} записей")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка F10.7 GFZ: {e}")

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
