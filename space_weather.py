#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — Сбор космической погоды: Dst, Kp, F10.7
Стратегия: SWPC → альтернативные источники → OMNIWeb → кэш → синтетика
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

# NOAA SWPC (свежие данные)
SWPC_DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_F107_URL = "https://services.swpc.noaa.gov/products/solar-radio-flux.json"

# Альтернативные источники
# Dst: WDC Kyoto (realtime + provisional)
WDC_KYOTO_DST_REALTIME = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/index.html"
WDC_KYOTO_DST_PROV = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/{year}{month:02d}.dst"

# Kp: GFZ Potsdam (исторический архив, но тоже реальные данные)
GFZ_KP_URL = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"

# F10.7: LASP (университет Колорадо)
LASP_F107_URL = "https://lasp.colorado.edu/lisird/data/570_daily.txt"

# OMNIWeb (исторические данные, задержка 2-6 месяцев) — используем прямой HTTP
OMNIWEB_BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"

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

def _get_last_real(param: str) -> Optional[Tuple[pd.DataFrame, int]]:
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
# ПАРСЕРЫ АЛЬТЕРНАТИВНЫХ ИСТОЧНИКОВ
# ═══════════════════════════════════════════════════════════════

def _parse_gfz_kp(text: str) -> pd.DataFrame:
    """Парсинг Kp из GFZ Potsdam (фиксированная ширина)."""
    records = []
    for line in text.splitlines():
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
                kp_val = float(parts[5+i]) if parts[5+i] else None
                if kp_val is not None:
                    hour = i * 3
                    dt = datetime(year, month, day, hour, 0, tzinfo=timezone.utc)
                    records.append({'time': dt, 'kp': kp_val})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)

def _parse_lasp_f107(text: str) -> pd.DataFrame:
    """Парсинг F10.7 из LASP (фиксированная ширина)."""
    records = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year = int(parts[0])
            day_of_year = int(parts[1])
            f107 = float(parts[2]) if parts[2] else None
            if f107 is not None:
                dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
                records.append({'time': dt, 'f107': f107})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════
# OMNIWEB (исторические данные)
# ═══════════════════════════════════════════════════════════════

def _fetch_omni_year(year: int) -> Optional[pd.DataFrame]:
    """Скачивает OMNI данные за год и возвращает DataFrame."""
    url = f"{OMNIWEB_BASE}/omni2_{year}.dat"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        records = []
        for line in lines:
            if len(line) < 50:
                continue
            try:
                year_val = int(line[0:4].strip())
                doy = int(line[4:7].strip())
                hour = int(line[7:10].strip())
                dt = datetime(year_val, 1, 1, hour, 0, 0, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                dst = float(line[13:19].strip()) if line[13:19].strip() and line[13:19].strip() != '99999' else None
                kp = float(line[25:31].strip()) / 10 if line[25:31].strip() and line[25:31].strip() != '999' else None
                f107 = float(line[31:37].strip()) / 10 if line[31:37].strip() and line[31:37].strip() != '9999' else None
                if dst is not None or kp is not None or f107 is not None:
                    record = {'time': dt, 'data_source': 'OMNIWeb'}
                    if dst is not None:
                        record['dst'] = dst
                    if kp is not None:
                        record['kp'] = kp
                    if f107 is not None:
                        record['f107'] = f107
                    records.append(record)
            except (ValueError, IndexError):
                continue
        if records:
            df = pd.DataFrame(records)
            logger.info(f"✅ OMNIWeb {year}: {len(df)} записей")
            return df
    except Exception as e:
        logger.debug(f"OMNIWeb {year} недоступен: {e}")
    return None

def _fetch_omni_range(start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
    """Загружает OMNI данные за диапазон лет."""
    start_year = start_time.year
    end_year = end_time.year
    all_dfs = []
    for year in range(start_year, end_year + 1):
        df = _fetch_omni_year(year)
        if df is not None:
            all_dfs.append(df)
        time.sleep(0.5)
    if not all_dfs:
        return {}
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('time').reset_index(drop=True)
    mask = (combined['time'] >= start_time) & (combined['time'] <= end_time)
    filtered = combined[mask]
    result = {}
    for param in ['dst', 'kp', 'f107']:
        if param in filtered.columns:
            subset = filtered[['time', param, 'data_source']].dropna(subset=[param])
            if not subset.empty:
                subset = subset.rename(columns={param: param})
                result[param] = subset
    return result

# ═══════════════════════════════════════════════════════════════
# ОСНОВНЫЕ МЕТОДЫ КОЛЛЕКТОРА
# ═══════════════════════════════════════════════════════════════

class SpaceWeatherCollector:
    """
    Сборщик космической погоды с резервными источниками.
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

    # ---------- Dst ----------
    def fetch_dst(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # 1. SWPC
        df = self._try_swpc(SWPC_DST_URL, _parse_swpc_dst, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('dst', df)
            return df

        # 2. WDC Kyoto (realtime + provisional)
        for year in range(start_time.year, end_time.year + 1):
            for month in range(1, 13):
                if (year, month) < (start_time.year, start_time.month):
                    continue
                if (year, month) > (end_time.year, end_time.month):
                    break
                for url_func, src_name in [
                    (lambda y, m: WDC_KYOTO_DST_REALTIME.format(year=y, month=m), 'WDC_Realtime'),
                    (lambda y, m: WDC_KYOTO_DST_PROV.format(year=y, month=m), 'WDC_Provisional')
                ]:
                    try:
                        resp = requests.get(url_func(year, month), timeout=30)
                        if resp.status_code != 200:
                            continue
                        # Парсинг (упрощённый)
                        if 'realtime' in src_name.lower():
                            # HTML парсинг
                            html = resp.text
                            pattern = r'<tr>\s*<td>(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})</td>\s*<td>(-?\d+)</td>'
                            matches = re.findall(pattern, html)
                            records = []
                            for time_str, dst_val in matches:
                                dt = pd.to_datetime(time_str).tz_localize('UTC')
                                records.append({'time': dt, 'dst': float(dst_val)})
                            if records:
                                df = pd.DataFrame(records)
                                df['data_source'] = src_name
                                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                                if not df.empty:
                                    _save_last_real('dst', df)
                                    return df
                        else:
                            # Provisional: fixed-width
                            lines = resp.text.splitlines()
                            records = []
                            for line in lines:
                                if not line.strip() or line.startswith('#'):
                                    continue
                                try:
                                    year_str = line[2:6].strip()
                                    day_of_year = int(line[7:10].strip())
                                    hour = int(line[11:13].strip())
                                    year_val = int(year_str)
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
                                df['data_source'] = src_name
                                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                                if not df.empty:
                                    _save_last_real('dst', df)
                                    return df
                    except Exception as e:
                        logger.debug(f"WDC {src_name} недоступен: {e}")

        # 3. OMNIWeb (исторические)
        omni = _fetch_omni_range(start_time, end_time)
        if 'dst' in omni and not omni['dst'].empty:
            df = omni['dst']
            _save_last_real('dst', df)
            return df

        # 4. Кэш последних реальных
        cached, delay = _get_last_real('dst')
        if cached is not None:
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ Dst: используются кэшированные данные (задержка {delay} дн.)")
                return df_cached
            else:
                logger.warning(f"⚠️ Dst: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                return cached

        # 5. Синтетика
        logger.error(f"🔴 Нет реальных данных Dst, использую синтетику")
        return self._synthetic_dst(start_time, end_time)

    # ---------- Kp ----------
    def fetch_kp(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # 1. SWPC
        df = self._try_swpc(SWPC_KP_URL, _parse_swpc_kp, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('kp', df)
            return df

        # 2. GFZ Potsdam (альтернатива)
        try:
            resp = requests.get(GFZ_KP_URL, timeout=30)
            resp.raise_for_status()
            df = _parse_gfz_kp(resp.text)
            if not df.empty:
                df['data_source'] = 'GFZ'
                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                if not df.empty:
                    _save_last_real('kp', df)
                    return df
        except Exception as e:
            logger.debug(f"GFZ Kp недоступен: {e}")

        # 3. OMNIWeb
        omni = _fetch_omni_range(start_time, end_time)
        if 'kp' in omni and not omni['kp'].empty:
            df = omni['kp']
            _save_last_real('kp', df)
            return df

        # 4. Кэш
        cached, delay = _get_last_real('kp')
        if cached is not None:
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ Kp: используются кэшированные данные (задержка {delay} дн.)")
                return df_cached
            else:
                logger.warning(f"⚠️ Kp: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                return cached

        # 5. Синтетика
        logger.error(f"🔴 Нет реальных данных Kp, использую синтетику")
        return self._synthetic_kp(start_time, end_time)

    # ---------- F10.7 ----------
    def fetch_f107(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # 1. SWPC
        df = self._try_swpc(SWPC_F107_URL, _parse_swpc_f107, start_time, end_time)
        if df is not None and not df.empty:
            _save_last_real('f107', df)
            return df

        # 2. LASP (альтернатива)
        try:
            resp = requests.get(LASP_F107_URL, timeout=30)
            resp.raise_for_status()
            df = _parse_lasp_f107(resp.text)
            if not df.empty:
                df['data_source'] = 'LASP'
                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                if not df.empty:
                    _save_last_real('f107', df)
                    return df
        except Exception as e:
            logger.debug(f"LASP F10.7 недоступен: {e}")

        # 3. OMNIWeb
        omni = _fetch_omni_range(start_time, end_time)
        if 'f107' in omni and not omni['f107'].empty:
            df = omni['f107']
            _save_last_real('f107', df)
            return df

        # 4. Кэш
        cached, delay = _get_last_real('f107')
        if cached is not None:
            df_cached = cached[(cached['time'] >= start_time) & (cached['time'] <= end_time)]
            if not df_cached.empty:
                logger.warning(f"⚠️ F10.7: используются кэшированные данные (задержка {delay} дн.)")
                return df_cached
            else:
                logger.warning(f"⚠️ F10.7: кэшированные данные не покрывают запрошенный период, возвращаю последние {len(cached)} записей")
                return cached

        # 5. Синтетика
        logger.error(f"🔴 Нет реальных данных F10.7, использую синтетику")
        return self._synthetic_f107(start_time, end_time)

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
# SINGLETON И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

_collector_instance = None

def get_collector() -> SpaceWeatherCollector:
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = SpaceWeatherCollector()
    return _collector_instance

def get_data_quality(start_time: datetime, end_time: datetime) -> Dict[str, Dict]:
    collector = get_collector()
    data = collector.fetch_all_for_period(start_time, end_time)
    report = {}
    for param, df in data.items():
        if df.empty:
            report[param] = {'status': 'NO_DATA', 'real_pct': 0.0}
            continue
        total = len(df)
        real = len(df[df['data_source'] != 'SYNTHETIC'])
        real_pct = real / total * 100
        status = 'GOOD' if real_pct > 80 else 'PARTIAL' if real_pct > 20 else 'SYNTHETIC'
        report[param] = {
            'status': status,
            'real_pct': real_pct,
            'records': total,
            'sources': df['data_source'].value_counts().to_dict()
        }
    return report
