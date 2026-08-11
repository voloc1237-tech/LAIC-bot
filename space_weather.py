#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — Сбор космической погоды: Dst, Kp, F10.7
Стратегия: в первую очередь реальные данные, даже устаревшие.
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple
import os
import time
import re

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

SWPC_DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_F107_URL = "https://services.swpc.noaa.gov/products/solar-radio-flux.json"

WDC_KYOTO_DST_REALTIME = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/index.html"
WDC_KYOTO_DST_PROV = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/{year}{month:02d}.dst"

OMNIWEB_FTP = "spdf.gsfc.nasa.gov"
OMNIWEB_PATH = "/pub/data/omni/low_res_omni/"
OMNIWEB_WEB = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "space_weather_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# КЭШ ПОСЛЕДНИХ РЕАЛЬНЫХ ДАННЫХ (в памяти)
# ═══════════════════════════════════════════════════════════════

_last_real_dst = None   # (DataFrame, timestamp)
_last_real_kp = None
_last_real_f107 = None

def _save_last_real(param: str, df: pd.DataFrame):
    """Сохраняет последние реальные данные в кэш."""
    global _last_real_dst, _last_real_kp, _last_real_f107
    if df.empty:
        return
    if param == 'dst':
        _last_real_dst = (df.copy(), datetime.now(timezone.utc))
    elif param == 'kp':
        _last_real_kp = (df.copy(), datetime.now(timezone.utc))
    elif param == 'f107':
        _last_real_f107 = (df.copy(), datetime.now(timezone.utc))

def _get_last_real(param: str) -> Optional[pd.DataFrame]:
    """Возвращает последние реальные данные и задержку в днях."""
    global _last_real_dst, _last_real_kp, _last_real_f107
    cache = {'dst': _last_real_dst, 'kp': _last_real_kp, 'f107': _last_real_f107}.get(param)
    if cache is None:
        return None, None
    df, timestamp = cache
    delay = (datetime.now(timezone.utc) - timestamp).days
    return df, delay

# ═══════════════════════════════════════════════════════════════
# ПАРСЕРЫ SWPC
# ═══════════════════════════════════════════════════════════════

def _parse_swpc_dst(data: list) -> pd.DataFrame:
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    records = []
    for item in data[1:]:
        try:
            time_str = item[0] if isinstance(item, list) else item.get('time_tag')
            dst_val = item[1] if isinstance(item, list) else item.get('dst')
            if time_str and dst_val is not None:
                dt = pd.to_datetime(time_str).tz_localize('UTC')
                records.append({'time': dt, 'dst': float(dst_val)})
        except (ValueError, TypeError, IndexError):
            continue
    return pd.DataFrame(records)

def _parse_swpc_kp(data: list) -> pd.DataFrame:
    if not isinstance(data, list):
        return pd.DataFrame()
    records = []
    for item in data:
        try:
            time_str = item.get('time_tag')
            kp_val = item.get('Kp') or item.get('kp') or item.get('kp_index')
            if time_str and kp_val is not None:
                dt = pd.to_datetime(time_str).tz_localize('UTC')
                records.append({'time': dt, 'kp': float(kp_val)})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(records)

def _parse_swpc_f107(data: list) -> pd.DataFrame:
    if not isinstance(data, list):
        return pd.DataFrame()
    records = []
    for item in data:
        try:
            time_str = item.get('time_tag')
            flux_val = item.get('flux')
            if time_str and flux_val is not None:
                dt = pd.to_datetime(time_str).tz_localize('UTC')
                records.append({'time': dt, 'f107': float(flux_val)})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════
# WDC KYOTO DST (REALTIME + PROVISIONAL)
# ═══════════════════════════════════════════════════════════════

def _fetch_wdc_dst_realtime(year: int, month: int) -> Optional[pd.DataFrame]:
    """Парсит HTML-таблицу Real-time Dst с WDC Kyoto."""
    url = WDC_KYOTO_DST_REALTIME.format(year=year, month=month)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text
        # Ищем таблицу с Dst значениями (упрощённый парсинг)
        # Формат: строки с датами и значениями
        # Пример: <tr><td>2026-04-01 00:00</td><td>-12</td>...
        import re
        pattern = r'<tr>\s*<td>(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})</td>\s*<td>(-?\d+)</td>'
        matches = re.findall(pattern, html)
        records = []
        for time_str, dst_val in matches:
            dt = pd.to_datetime(time_str).tz_localize('UTC')
            records.append({'time': dt, 'dst': float(dst_val)})
        if records:
            df = pd.DataFrame(records)
            logger.info(f"✅ WDC Real-time Dst: {len(df)} записей за {year}-{month:02d}")
            return df
    except Exception as e:
        logger.debug(f"WDC Real-time Dst недоступен: {e}")
    return None

def _fetch_wdc_dst_provisional(year: int, month: int) -> Optional[pd.DataFrame]:
    """Скачивает и парсит Provisional Dst (фиксированная ширина)."""
    url = WDC_KYOTO_DST_PROV.format(year=year, month=month)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        records = []
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                year_str = line[2:6].strip()
                day_of_year = int(line[7:10].strip())
                hour = int(line[11:13].strip())
                if not year_str:
                    continue
                year_val = int(year_str)
                # 24 значения по 4 символа
                dst_vals = []
                for i in range(24):
                    pos = 20 + i * 4
                    val_str = line[pos:pos+4].strip()
                    if val_str and val_str != '9999':
                        dst_vals.append(float(val_str))
                    else:
                        dst_vals.append(None)
                for i, dst_val in enumerate(dst_vals):
                    if dst_val is not None:
                        dt = datetime(year_val, 1, 1, hour, 0, 0, tzinfo=timezone.utc) + timedelta(days=day_of_year-1, hours=i)
                        records.append({'time': dt, 'dst': dst_val})
            except (ValueError, IndexError):
                continue
        if records:
            df = pd.DataFrame(records)
            logger.info(f"✅ WDC Provisional Dst: {len(df)} записей за {year}-{month:02d}")
            return df
    except Exception as e:
        logger.debug(f"WDC Provisional Dst недоступен: {e}")
    return None

# ═══════════════════════════════════════════════════════════════
# ОСНОВНЫЕ МЕТОДЫ КОЛЛЕКТОРА
# ═══════════════════════════════════════════════════════════════

class SpaceWeatherCollector:
    """
    Сборщик космической погоды с приоритетом на реальные данные.
    """

    def __init__(self):
        self._omni_available = True

    def _try_swpc(self, url: str, parser, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = parser(resp.json())
            if not df.empty:
                mask = (df['time'] >= start_time) & (df['time'] <= end_time)
                df = df[mask]
                if len(df) > 0:
                    df['data_source'] = 'SWPC'
                    logger.debug(f"✅ SWPC: {len(df)} записей")
                    return df
        except Exception as e:
            logger.debug(f"SWPC недоступен: {e}")
        return None

    def _try_omni(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """OMNIWeb — только для исторических данных (до 2025)."""
        if not self._omni_available:
            return {}
        try:
            from space_weather_omni import OMNIWebClient  # импортируем из отдельного модуля
            client = OMNIWebClient()
            result = client.fetch_range(start_time, end_time)
            if any(not df.empty for df in result.values()):
                logger.info("✅ OMNIWeb: загружены исторические данные")
                return result
            else:
                return {}
        except Exception as e:
            logger.warning(f"⚠️ OMNIWeb недоступен: {e}")
            self._omni_available = False
            return {}

    def _generate_synthetic(self, param: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Синтетические данные — самый крайний случай."""
        logger.error(f"🔴 Нет реальных данных {param.upper()}, использую синтетику")
        if param == 'dst':
            return self._synthetic_dst(start_time, end_time)
        elif param == 'kp':
            return self._synthetic_kp(start_time, end_time)
        else:
            return self._synthetic_f107(start_time, end_time)

    def fetch_dst(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Dst: SWPC → WDC → кэш → OMNI → синтетика."""
        # 1. SWPC
        df = self._try_swpc(SWPC_DST_URL, _parse_swpc_dst, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('dst', df)
            return df

        # 2. WDC Kyoto (Realtime + Provisional)
        for year in range(start_time.year, end_time.year + 1):
            for month in range(1, 13):
                if (year, month) < (start_time.year, start_time.month):
                    continue
                if (year, month) > (end_time.year, end_time.month):
                    break
                # Realtime
                df = _fetch_wdc_dst_realtime(year, month)
                if df is not None and not df.empty:
                    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                    if not df.empty:
                        df['data_source'] = 'WDC_Realtime'
                        _save_last_real('dst', df)
                        return df
                # Provisional (если Realtime нет)
                df = _fetch_wdc_dst_provisional(year, month)
                if df is not None and not df.empty:
                    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                    if not df.empty:
                        df['data_source'] = 'WDC_Provisional'
                        _save_last_real('dst', df)
                        return df

        # 3. Кэш последних реальных
        cached, delay = _get_last_real('dst')
        if cached is not None:
            # Обрезаем по запрошенному диапазону (если данные есть за этот период)
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ Dst: используются кэшированные данные (задержка {delay} дн.)")
                df_cached['data_source'] = 'CACHE'
                return df_cached
            else:
                # Если данных за нужный период нет — берём всё, что есть, и предупреждаем
                logger.warning(f"⚠️ Dst: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                cached['data_source'] = 'CACHE_PARTIAL'
                return cached

        # 4. OMNIWeb (исторические)
        omni = self._try_omni(start_time, end_time)
        if 'dst' in omni and not omni['dst'].empty:
            df = omni['dst']
            df['data_source'] = 'OMNIWeb'
            _save_last_real('dst', df)
            return df

        # 5. Синтетика
        return self._generate_synthetic('dst', start_time, end_time)

    def fetch_kp(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Kp: SWPC → кэш → OMNI → синтетика."""
        # 1. SWPC
        df = self._try_swpc(SWPC_KP_URL, _parse_swpc_kp, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('kp', df)
            return df

        # 2. Кэш
        cached, delay = _get_last_real('kp')
        if cached is not None:
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ Kp: используются кэшированные данные (задержка {delay} дн.)")
                df_cached['data_source'] = 'CACHE'
                return df_cached
            else:
                logger.warning(f"⚠️ Kp: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                cached['data_source'] = 'CACHE_PARTIAL'
                return cached

        # 3. OMNIWeb
        omni = self._try_omni(start_time, end_time)
        if 'kp' in omni and not omni['kp'].empty:
            df = omni['kp']
            df['data_source'] = 'OMNIWeb'
            _save_last_real('kp', df)
            return df

        # 4. Синтетика
        return self._generate_synthetic('kp', start_time, end_time)

    def fetch_f107(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """F10.7: SWPC → кэш → OMNI → синтетика."""
        # 1. SWPC
        df = self._try_swpc(SWPC_F107_URL, _parse_swpc_f107, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('f107', df)
            return df

        # 2. Кэш
        cached, delay = _get_last_real('f107')
        if cached is not None:
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ F10.7: используются кэшированные данные (задержка {delay} дн.)")
                df_cached['data_source'] = 'CACHE'
                return df_cached
            else:
                logger.warning(f"⚠️ F10.7: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                cached['data_source'] = 'CACHE_PARTIAL'
                return cached

        # 3. OMNIWeb
        omni = self._try_omni(start_time, end_time)
        if 'f107' in omni and not omni['f107'].empty:
            df = omni['f107']
            df['data_source'] = 'OMNIWeb'
            _save_last_real('f107', df)
            return df

        # 4. Синтетика
        return self._generate_synthetic('f107', start_time, end_time)

    def fetch_all_for_period(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        logger.info(f"🛰️ Space weather: {start_time.date()} — {end_time.date()}")
        return {
            'dst': self.fetch_dst(start_time, end_time),
            'kp': self.fetch_kp(start_time, end_time),
            'f107': self.fetch_f107(start_time, end_time),
        }

    # ---------- Синтетические генераторы ----------
    @staticmethod
    def _synthetic_dst(start, end):
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ Dst: {start.date()} — {end.date()}")
        dates = pd.date_range(start=start, end=end, freq='h', tz='UTC')
        records = []
        for dt in dates:
            base = -10 + 15*np.sin(2*np.pi*dt.timetuple().tm_yday/365.25)
            noise = np.random.normal(0, 8)
            dst = max(-150, min(50, base + noise))
            records.append({'time': dt, 'dst': round(dst, 1), 'data_source': 'SYNTHETIC'})
        return pd.DataFrame(records)

    @staticmethod
    def _synthetic_kp(start, end):
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ Kp: {start.date()} — {end.date()}")
        dates = pd.date_range(start=start, end=end, freq='3H', tz='UTC')
        records = []
        for dt in dates:
            mu, sigma = 1.5, 0.5
            kp = np.random.lognormal(mu, sigma)
            records.append({'time': dt, 'kp': round(min(9.0, kp), 1), 'data_source': 'SYNTHETIC'})
        return pd.DataFrame(records)

    @staticmethod
    def _synthetic_f107(start, end):
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ F10.7: {start.date()} — {end.date()}")
        dates = pd.date_range(start=start, end=end, freq='D', tz='UTC')
        records = []
        for dt in dates:
            base = 150 + 20*np.sin(2*np.pi*dt.timetuple().tm_yday/27)
            noise = np.random.normal(0, 15)
            f107 = max(50, min(300, base + noise))
            records.append({'time': dt, 'f107': round(f107, 1), 'data_source': 'SYNTHETIC'})
        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_collector_instance = None

def get_collector() -> SpaceWeatherCollector:
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = SpaceWeatherCollector()
    return _collector_instance
