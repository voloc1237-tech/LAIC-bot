#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_weather.py — Сбор космической погоды: Dst, Kp, F10.7

Источники:
    1. NOAA SWPC — реальные данные (свежие, ~7-30 дней)
    2. NASA OMNIWeb — исторические данные (1963-настоящее)
    3. Синтетический fallback — только если оба источника недоступны

OMNIWeb: https://omniweb.gsfc.nasa.gov/
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple
import os
import io
import time
import ftplib

# Подавление шумных логов
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ИСТОЧНИКОВ
# ═══════════════════════════════════════════════════════════════

# NOAA SWPC (свежие данные)
SWPC_DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_F107_URL = "https://services.swpc.noaa.gov/products/10cm-flux-30-day.json"

SWPC_DST_DAYS = 30
SWPC_KP_DAYS = 7
SWPC_F107_DAYS = 30

# NASA OMNIWeb (исторические данные)
OMNIWEB_FTP = "spdf.gsfc.nasa.gov"
OMNIWEB_PATH = "/pub/data/omni/low_res_omni/"
OMNIWEB_WEB = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Кэш директория
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "space_weather_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# КЭШИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

class DataCache:
    """Простой файловый кэш для OMNI данных."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _cache_path(self, year: int, param: str) -> str:
        return os.path.join(self.cache_dir, f"omni_{param}_{year}.csv")
    
    def has(self, year: int, param: str) -> bool:
        return os.path.exists(self._cache_path(year, param))
    
    def get(self, year: int, param: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(year, param)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=['time'])
            logger.debug(f"📂 Кэш загружен: {year} {param}")
            return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения кэша {path}: {e}")
            return None
    
    def put(self, year: int, param: str, df: pd.DataFrame):
        path = self._cache_path(year, param)
        df.to_csv(path, index=False)
        logger.debug(f"💾 Кэш сохранён: {year} {param}")


# ═══════════════════════════════════════════════════════════════
# ПАРСЕРЫ SWPC
# ═══════════════════════════════════════════════════════════════

def _parse_swpc_dst(data: list) -> pd.DataFrame:
    """Парсинг Dst из SWPC JSON."""
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    
    # Пропускаем заголовок [0]
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
    """Парсинг Kp из SWPC JSON."""
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
    """Парсинг F10.7 из SWPC JSON."""
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
# OMNIWEB HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════

class OMNIWebClient:
    """
    Клиент для NASA OMNIWeb historical data.
    
    OMNI 1-min и 5-min данные доступны через FTP и веб-форму.
    Используем hourly (низкое разрешение) — достаточно для LAIC.
    """
    
    # Формат OMNI hourly ASCII:
    # Колонки: Year, DOY, Hour, BST, ... Dst, Kp*10, F10.7, ...
    # Подробнее: https://omniweb.gsfc.nasa.gov/html/ow_data.html
    
    OMNI_COLUMNS = [
        'year', 'doy', 'hour', 'bst', 'dst', 'ae', 'kp', 'f107',
        'proton_density', 'plasma_speed', 'imf_bz'
    ]
    
    # Позиции колонок в fixed-width формате (hourly)
    OMNI_COLSPECS = [
        (0, 4),    # year
        (4, 7),    # doy
        (7, 10),   # hour
        (10, 13),  # bst ( Bartels solar rotation number)
        (13, 19),  # dst (nT)
        (19, 25),  # ae (nT)
        (25, 31),  # kp (*10)
        (31, 37),  # f107 (*10)
        # ... можно добавить больше
    ]
    
    def __init__(self, cache: DataCache = None):
        self.cache = cache or DataCache()
    
    def _year_to_file_url(self, year: int) -> str:
        """URL для скачивания файла года."""
        return f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
    
    def _download_year(self, year: int) -> Optional[pd.DataFrame]:
        """Скачать данные за год с OMNIWeb."""
        # Сначала проверяем кэш
        cached = self.cache.get(year, 'all')
        if cached is not None:
            return cached
        
        url = self._year_to_file_url(year)
        logger.info(f"🌐 Загрузка OMNI {year}...")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Парсинг fixed-width формата
            lines = response.text.strip().split('\n')
            records = []
            
            for line in lines:
                if len(line) < 50:
                    continue
                
                try:
                    year_val = int(line[0:4].strip())
                    doy = int(line[4:7].strip())
                    hour = int(line[7:10].strip())
                    
                    # Конвертируем year+doy+hour в datetime
                    dt = datetime(year_val, 1, 1, hour, 0, 0, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                    
                    # Парсим значения (9999 = missing)
                    dst = self._parse_value(line[13:19], scale=1.0, missing=99999)
                    kp = self._parse_value(line[25:31], scale=0.1, missing=999)  # Kp хранится *10
                    f107 = self._parse_value(line[31:37], scale=0.1, missing=9999)  # F10.7 *10
                    
                    records.append({
                        'time': dt,
                        'dst': dst,
                        'kp': kp,
                        'f107': f107,
                        'data_source': 'OMNIWeb'
                    })
                    
                except (ValueError, IndexError):
                    continue
            
            if not records:
                logger.warning(f"⚠️ Нет данных OMNI для {year}")
                return None
            
            df = pd.DataFrame(records)
            
            # Сохраняем в кэш
            self.cache.put(year, 'all', df)
            
            logger.info(f"✅ OMNI {year}: {len(df)} записей")
            return df
            
        except requests.RequestException as e:
            logger.warning(f"⚠️ Ошибка загрузки OMNI {year}: {e}")
            return None
    
    def _parse_value(self, s: str, scale: float = 1.0, missing: int = 9999) -> Optional[float]:
        """Парсинг значения с обработкой missing data."""
        try:
            val = int(s.strip())
            if abs(val) >= missing:
                return None
            return val * scale
        except ValueError:
            return None
    
    def fetch_range(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Получить данные за диапазон дат."""
        start_year = start_time.year
        end_year = end_time.year
        
        all_data = []
        for year in range(start_year, end_year + 1):
            df = self._download_year(year)
            if df is not None:
                all_data.append(df)
            
            # Задержка чтобы не DDoSить NASA
            time.sleep(0.5)
        
        if not all_data:
            logger.error("❌ Не удалось загрузить данные OMNI")
            return {'dst': pd.DataFrame(), 'kp': pd.DataFrame(), 'f107': pd.DataFrame()}
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('time').reset_index(drop=True)
        
        # Фильтруем по диапазону
        mask = (combined['time'] >= start_time) & (combined['time'] <= end_time)
        filtered = combined[mask]
        
        # Разделяем по параметрам (пропускаем None)
        result = {}
        for param in ['dst', 'kp', 'f107']:
            param_df = filtered[['time', param, 'data_source']].dropna(subset=[param])
            
            # Проверка что данные есть
            if param_df.empty:
                logger.warning(f"⚠️ OMNI: нет данных {param} после фильтрации")
            
            result[param] = param_df.rename(columns={param: param})
        
        return result

    


# ═══════════════════════════════════════════════════════════════
# СИНТЕТИЧЕСКИЙ FALLBACK (последняя надежда)
# ═══════════════════════════════════════════════════════════════

class SyntheticSpaceWeather:
    """
    Синтетические данные ТОЛЬКО когда нет реальных.
    С предупреждениями и маркировкой.
    """
    
    @staticmethod
    def generate_dst(start_time: datetime, end_time: datetime, freq: str = 'h') -> pd.DataFrame:
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ Dst: {start_time.date()} — {end_time.date()}")
        
        dates = pd.date_range(start=start_time, end=end_time, freq=freq, tz='UTC')
        records = []
        
        for dt in dates:
            day_of_year = dt.timetuple().tm_yday
            base = -10
            seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365.25)
            # Случайность без фиксированного seed!
            noise = np.random.normal(0, 8)
            dst = max(-150, min(50, base + seasonal + noise))
            
            records.append({
                'time': dt,
                'dst': round(dst, 1),
                'data_source': 'SYNTHETIC'
            })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def generate_kp(start_time: datetime, end_time: datetime, freq: str = '3H') -> pd.DataFrame:
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ Kp: {start_time.date()} — {end_time.date()}")
        
        dates = pd.date_range(start=start_time, end=end_time, freq=freq, tz='UTC')
        records = []
        
        for dt in dates:
            # Логнормальное распределение (более реалистичное)
            mu, sigma = 1.5, 0.5
            kp = np.random.lognormal(mu, sigma)
            
            records.append({
                'time': dt,
                'kp': round(min(9.0, kp), 1),
                'data_source': 'SYNTHETIC'
            })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def generate_f107(start_time: datetime, end_time: datetime, freq: str = 'D') -> pd.DataFrame:
        logger.warning(f"🔴 СИНТЕТИЧЕСКИЙ F10.7: {start_time.date()} — {end_time.date()}")
        
        dates = pd.date_range(start=start_time, end=end_time, freq=freq, tz='UTC')
        records = []
        
        for dt in dates:
            day_of_year = dt.timetuple().tm_yday
            base = 150
            rotation = 20 * np.sin(2 * np.pi * day_of_year / 27)  # Солнечная ротация
            noise = np.random.normal(0, 15)
            f107 = max(50, min(300, base + rotation + noise))
            
            records.append({
                'time': dt,
                'f107': round(f107, 1),
                'data_source': 'SYNTHETIC'
            })
        
        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС-КОЛЛЕКТОР
# ═══════════════════════════════════════════════════════════════

class SpaceWeatherCollector:
    """
    Унифицированный сборщик космической погоды.
    
    Приоритет источников:
        1. NOAA SWPC (свежие данные)
        2. NASA OMNIWeb (исторические)
        3. Синтетический fallback (предупреждение)
    """
    
    def __init__(self):
        self.omni = OMNIWebClient()
        self.synthetic = SyntheticSpaceWeather()
        self.cache = DataCache()
        self._omni_available = True  # Проверяем при первом использовании
    
    def _is_recent(self, end_time: datetime, days_limit: int) -> bool:
        """Проверяет, достаточно ли свежая дата для SWPC."""
        days_diff = (datetime.now(timezone.utc) - end_time).days
        return days_diff <= days_limit
    
    def _try_swpc(self, url: str, parser, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Попытка загрузки из SWPC."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = parser(response.json())
            
            if not df.empty:
                # Фильтруем по диапазону
                mask = (df['time'] >= start_time) & (df['time'] <= end_time)
                df = df[mask]
                
                if len(df) > 0:
                    df['data_source'] = 'SWPC'
                    logger.info(f"✅ SWPC: {len(df)} записей")
                    return df
            
        except Exception as e:
            logger.debug(f"SWPC недоступен: {e}")
        
        return None
    
    def _try_omni(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Попытка загрузки из OMNIWeb."""
        if not self._omni_available:
            return {}
        
        try:
            result = self.omni.fetch_range(start_time, end_time)
            
            # Проверяем, есть ли данные
            has_data = any(not df.empty for df in result.values())
            
            if has_data:
                logger.info(f"✅ OMNIWeb: загружены исторические данные")
                return result
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"⚠️ OMNIWeb недоступен: {e}")
            self._omni_available = False  # Не пытаться больше
            return {}
    
    def fetch_dst(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Dst: SWPC → OMNI → Synthetic
        """
        # 1. Пробуем SWPC (если свежие данные)
        if self._is_recent(end_time, SWPC_DST_DAYS):
            df = self._try_swpc(SWPC_DST_URL, _parse_swpc_dst, start_time, end_time)
            if df is not None and not df.empty:
                return df
        
        # 2. Пробуем OMNIWeb (исторические)
        omni_data = self._try_omni(start_time, end_time)
        logger.debug(f"OMNI результат: {omni_data.keys() if omni_data else 'пусто'}")
        if 'dst' in omni_data and not omni_data['dst'].empty:
            logger.info(f"✅ Dst из OMNI: {len(omni_data['dst'])} записей")
            return omni_data['dst']
    
        # 3. Синтетический fallback
        logger.error(f"🔴 Нет реальных данных Dst для {start_time.date()}-{end_time.date()}")
        return self.synthetic.generate_dst(start_time, end_time)
    
    def fetch_kp(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Kp: SWPC → OMNI → Synthetic
        """
        # 1. Пробуем SWPC (только 7 дней!)
        if self._is_recent(end_time, SWPC_KP_DAYS):
            df = self._try_swpc(SWPC_KP_URL, _parse_swpc_kp, start_time, end_time)
            if df is not None and not df.empty:
                return df
        
        # 2. Пробуем OMNIWeb (критично для Kp — только 7 дней SWPC!)
        omni_data = self._try_omni(start_time, end_time)
        if 'kp' in omni_data and not omni_data['kp'].empty:
            return omni_data['kp']
        
        # 3. Синтетический fallback
        logger.error(f"🔴 Нет реальных данных Kp для {start_time.date()}-{end_time.date()}")
        return self.synthetic.generate_kp(start_time, end_time)
    
    def fetch_f107(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        F10.7: SWPC → OMNI → Synthetic
        """
        # 1. Пробуем SWPC
        if self._is_recent(end_time, SWPC_F107_DAYS):
            df = self._try_swpc(SWPC_F107_URL, _parse_swpc_f107, start_time, end_time)
            if df is not None and not df.empty:
                return df
        
        # 2. Пробуем OMNIWeb
        omni_data = self._try_omni(start_time, end_time)
        if 'f107' in omni_data and not omni_data['f107'].empty:
            return omni_data['f107']
        
        # 3. Синтетический fallback
        logger.error(f"🔴 Нет реальных данных F10.7 для {start_time.date()}-{end_time.date()}")
        return self.synthetic.generate_f107(start_time, end_time)
    
    def fetch_all_for_period(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Получить все параметры за период."""
        logger.info(f"🛰️ Space weather: {start_time.date()} — {end_time.date()}")
        
        return {
            'dst': self.fetch_dst(start_time, end_time),
            'kp': self.fetch_kp(start_time, end_time),
            'f107': self.fetch_f107(start_time, end_time)
        }
    
    def get_data_quality_report(self, start_time: datetime, end_time: datetime) -> Dict:
        """Отчёт о качестве данных за период."""
        result = self.fetch_all_for_period(start_time, end_time)
        
        report = {}
        for param, df in result.items():
            if df.empty:
                report[param] = {'status': 'NO_DATA', 'real_pct': 0.0}
                continue
            
            total = len(df)
            real = len(df[df['data_source'] != 'SYNTHETIC'])
            real_pct = real / total * 100
            
            report[param] = {
                'status': 'GOOD' if real_pct > 80 else 'PARTIAL' if real_pct > 20 else 'SYNTHETIC',
                'real_pct': real_pct,
                'sources': df['data_source'].value_counts().to_dict(),
                'records': total
            }
        
        return report


# ═══════════════════════════════════════════════════════════════
# SINGLETON И ОБРАТНАЯ СОВМЕСТИМОСТЬ
# ═══════════════════════════════════════════════════════════════

_collector_instance = None

def get_collector() -> SpaceWeatherCollector:
    """Получить singleton экземпляр коллектора."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = SpaceWeatherCollector()
    return _collector_instance

def fetch_all_for_period(start_time, end_time):
    """Обратная совместимость."""
    return get_collector().fetch_all_for_period(start_time, end_time)

def get_data_quality(start_time, end_time):
    """Отчёт о качестве данных."""
    return get_collector().get_data_quality_report(start_time, end_time)


# ═══════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = SpaceWeatherCollector()
    
    # Тест 1: Свежие данные (SWPC)
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Свежие данные (5 дней)")
    end1 = datetime.now(timezone.utc) - timedelta(days=1)
    start1 = end1 - timedelta(days=5)
    r1 = collector.fetch_all_for_period(start1, end1)
    for k, df in r1.items():
        sources = df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        print(f"{k.upper()}: {len(df)} записей, источники: {sources}")
    
    # Тест 2: Исторические данные (OMNIWeb)
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Исторические данные 2020 год")
    start2 = datetime(2020, 6, 1, tzinfo=timezone.utc)
    end2 = datetime(2020, 6, 10, tzinfo=timezone.utc)
    r2 = collector.fetch_all_for_period(start2, end2)
    for k, df in r2.items():
        sources = df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        print(f"{k.upper()}: {len(df)} записей, источники: {sources}")
    
    # Тест 3: Очень старые данные (Synthetic)
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Очень старые данные 1990 год")
    start3 = datetime(1990, 1, 1, tzinfo=timezone.utc)
    end3 = datetime(1990, 1, 5, tzinfo=timezone.utc)
    r3 = collector.fetch_all_for_period(start3, end3)
    for k, df in r3.items():
        sources = df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        print(f"{k.upper()}: {len(df)} записей, источники: {sources}")
    
    # Тест 4: Отчёт о качестве
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Отчёт о качестве данных")
    quality = collector.get_data_quality_report(start1, end1)
    for param, info in quality.items():
        print(f"{param}: {info['status']} (реальных: {info['real_pct']:.1f}%)")
