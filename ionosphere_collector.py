#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор реальных TEC-данных из IONEX-файлов.
Использует прямые HTTP-запросы к CDDIS и ручной парсинг.

✅ ИСПРАВЛЕНО в этой версии:
  1. Корректный парсер IONEX (использует lat_idx/lon_idx, читает ВСЕ эпохи)
  2. Оптимизация: загрузка IONEX один раз на событие (не 6 раз на 6 days_before)
  3. Кэширование результатов fetch_for_event по (event_id, lat, lon)
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IonosphereCollector:
    """
    Сборщик ионосферных данных (TEC) из IONEX-файлов.
    Скачивает файлы с CDDIS и парсит их вручную.
    """

    CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # ✅ ИСПРАВЛЕНО: кэш результатов в памяти для одного прогона
        self._event_cache = {}
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_ionex_filename(self, date: datetime) -> str:
        """Формирует имя IONEX-файла."""
        doy = date.timetuple().tm_yday
        return f"codg{date.year}{doy:03d}0.10i.Z"        # ✅ ИСПРАВЛЕНО: корректное имя файла CODG (с нулём перед .10i.Z)

    def _download_ionex(self, date: datetime) -> str | None:
        """Скачивает IONEX-файл с CDDIS."""
        filename = self._get_ionex_filename(date)
        # Сохраняем без .Z для простоты, но ищем оба варианта
        local_base = self.cache_dir / f"codg{date.year}{date.timetuple().tm_yday:03d}"
        local_path = local_base.with_suffix(".10i")
        local_path_z = local_base.with_suffix(".10i.Z")

        # Проверяем, есть ли уже распакованный файл
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"📂 IONEX в кэше: {local_path.name}")
            return str(local_path)

        # Если есть только запакованный — распаковываем (CDDIS часто даёт .Z)
        if local_path_z.exists() and local_path_z.stat().st_size > 0:
            try:
                import gzip
                import shutil
                with gzip.open(local_path_z, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if local_path.exists() and local_path.stat().st_size > 0:
                    logger.debug(f"📦 Распакован: {local_path.name}")
                    return str(local_path)
            except Exception as e:
                logger.debug(f"Не удалось распаковать {local_path_z}: {e}")

        year = date.strftime("%Y")
        doy = date.timetuple().tm_yday
        url = f"{self.CDDIS_BASE}/{year}/{doy:03d}/{filename}"

        try:
            logger.info(f"📥 Скачивание IONEX: {url}")
            urllib.request.urlretrieve(url, local_path_z, timeout=60)

            if local_path_z.stat().st_size == 0:
                local_path_z.unlink()
                return None

            # Распаковываем
            try:
                import gzip
                import shutil
                with gzip.open(local_path_z, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if local_path.exists() and local_path.stat().st_size > 0:
                    logger.debug(f"✅ Скачан и распакован: {local_path.name}")                    return str(local_path)
                else:
                    # Возможно, файл не был упакован — переименовываем
                    local_path_z.rename(local_path)
                    return str(local_path)
            except Exception as e:
                logger.debug(f"⚠️ Ошибка распаковки, пробуем как есть: {e}")
                local_path_z.rename(local_path)
                return str(local_path)

        except Exception as e:
            logger.debug(f"Не удалось скачать {filename}: {e}")
            return None

    def _parse_ionex_file(self, filepath: str, lat: float, lon: float, date: datetime) -> pd.DataFrame:
        """
        ✅ ИСПРАВЛЕННЫЙ ПАРСЕР IONEX.
        
        Правильно:
          - Считывает сетку из заголовка (LAT/DLAT, LON/DLON)
          - Использует lat_idx/lon_idx для извлечения значения именно в нужной точке
          - Читает ВСЕ эпохи (каждые 2 часа), а не только первую
          - Учитывает единицы (CODG хранит TEC в 0.1 TECU)
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения файла {filepath}: {e}")
            return pd.DataFrame()

        # ═══════════════════════════════════════════════════════════════
        # 1. Читаем параметры сетки из заголовка
        # ═══════════════════════════════════════════════════════════════
        lat_min, lat_max, lat_step = -87.5, 87.5, 2.5
        lon_min, lon_max, lon_step = -180.0, 180.0, 5.0
        exponent = -1  # масштабный коэффициент (обычно -1 → делим на 10)

        header_end = False
        for line in lines:
            if 'END OF HEADER' in line:
                header_end = True
                break
            if 'LAT1 / LAT2 / DLAT' in line or ('LAT' in line and 'DLAT' in line):
                parts = line.split()
                try:
                    lat_min = float(parts[0])
                    lat_max = float(parts[1])
                    lat_step = float(parts[2])
                except (ValueError, IndexError):                    pass
            elif 'LON1 / LON2 / DLON' in line or ('LON' in line and 'DLON' in line):
                parts = line.split()
                try:
                    lon_min = float(parts[0])
                    lon_max = float(parts[1])
                    lon_step = float(parts[2])
                except (ValueError, IndexError):
                    pass
            elif 'EXPONENT' in line:
                parts = line.split()
                try:
                    exponent = int(parts[0])
                except (ValueError, IndexError):
                    pass

        if not header_end:
            logger.debug(f"⚠️ Нет END OF HEADER в {filepath}")
            return pd.DataFrame()

        # Строим сетку
        lats = np.arange(lat_min, lat_max + lat_step / 2, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step / 2, lon_step)
        n_lon = len(lons)

        # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: вычисляем индексы целевой точки
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)) % n_lon)
        scale = 10 ** exponent  # обычно 0.1

        # ═══════════════════════════════════════════════════════════════
        # 2. Читаем карты TEC (после END OF HEADER)
        # ═══════════════════════════════════════════════════════════════
        records = []
        current_epoch_time = None
        in_tec_map = False
        current_lat_idx = -1
        header_idx = 0

        # Ищем END OF HEADER
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_idx = i
                break

        # Парсим данные
        for line in lines[header_idx + 1:]:
            # Начало карты
            if 'START OF TEC MAP' in line:
                in_tec_map = True                continue
            # Конец карты
            if 'END OF TEC MAP' in line:
                in_tec_map = False
                continue

            if not in_tec_map:
                continue

            # Эпоха карты (время)
            if 'EPOCH OF CURRENT MAP' in line:
                try:
                    parts = line.split()
                    y = int(parts[0])
                    m = int(parts[1])
                    d = int(parts[2])
                    h = int(parts[3])
                    mi = int(parts[4]) if len(parts) > 4 else 0
                    s = int(parts[5]) if len(parts) > 5 else 0
                    current_epoch_time = datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)
                except (ValueError, IndexError) as e:
                    logger.debug(f"⚠️ Ошибка парсинга эпохи: {e}")
                    continue
                continue

            # Строка широты: "  LAT   LON1   LON2   DLON   H"
            stripped = line.rstrip()
            if len(stripped) < 8:
                continue

            parts = stripped.split()
            if len(parts) >= 5 and parts[0] == 'LAT':
                try:
                    current_lat = float(parts[1])
                    current_lat_idx = int(np.argmin(np.abs(lats - current_lat)))
                except (ValueError, IndexError):
                    current_lat_idx = -1
                continue

            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: читаем строку с данными TEC
            # Только если текущая широта совпадает с целевой
            if current_lat_idx == lat_idx and len(stripped) >= 16:
                # Формат CODG: значения в формате I5 (5 символов на значение)
                try:
                    values_str = [stripped[i:i + 5].strip()
                                  for i in range(0, len(stripped), 5)
                                  if stripped[i:i + 5].strip()]
                    values = []
                    for v in values_str:
                        try:                            iv = int(v)
                            if iv != 9999 and iv != -9999:
                                values.append(iv * scale)  # применяем масштаб
                            else:
                                values.append(np.nan)
                        except ValueError:
                            values.append(np.nan)

                    if lon_idx < len(values) and not np.isnan(values[lon_idx]):
                        tec_val = values[lon_idx]
                        if 0 <= tec_val <= 300 and current_epoch_time is not None:
                            records.append({
                                'time': current_epoch_time,
                                'tec_value': float(tec_val),
                                'lat': float(lat),
                                'lon': float(lon),
                            })
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка чтения строки TEC: {e}")
                    continue

        if not records:
            logger.debug(f"⚠️ Нет валидных TEC для ({lat:.2f}, {lon:.2f}) в {filepath}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset=['time'])
        df = df.set_index("time")
        df = df.sort_index()
        return df

    def _generate_synthetic_tec(self, lat, lon, start_time, end_time) -> pd.DataFrame:
        """Генерирует синтетические TEC-данные (запасной вариант)."""
        records = []
        current = start_time
        base_tec = 30 - abs(lat) * 0.3

        # ✅ ИСПРАВЛЕНО: детерминированный seed по координатам
        # (чтобы один и тот же результат для одной точки в разных вызовах)
        seed = int(abs(lat) * 100 + abs(lon) * 10) % (2**31)
        rng = np.random.RandomState(seed)

        while current <= end_time:
            hour = current.hour
            daily = 5 * np.sin((hour - 10) * np.pi / 12)
            noise = rng.normal(0, 2)
            tec = max(0, base_tec + daily + noise)
            records.append({'time': current, 'tec_value': tec})
            current += timedelta(hours=2)
        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        logger.debug(f"🔄 Сгенерированы синтетические TEC для ({lat:.2f}, {lon:.2f})")
        return df

    def fetch_tec_for_point(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Получает TEC для точки. Сначала пробует реальные данные, при ошибке — синтетику."""
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        # ✅ ИСПРАВЛЕНО: кэш по диапазону дат, чтобы не скачивать повторно
        cache_key = f"tec_{lat:.1f}_{lon:.1f}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"

        all_dfs = []
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        files_downloaded = 0
        files_parsed_ok = 0

        while current_date <= end_time:
            filepath = self._download_ionex(current_date)
            if filepath:
                files_downloaded += 1
                df = self._parse_ionex_file(filepath, lat, lon, current_date)
                if not df.empty:
                    files_parsed_ok += 1
                    all_dfs.append(df)
                else:
                    logger.debug(f"⚠️ Файл {filepath} не дал TEC для ({lat:.2f}, {lon:.2f})")
            current_date += timedelta(days=1)

        if not all_dfs:
            logger.warning(
                f"⚠️ Нет реальных TEC для ({lat:.2f}, {lon:.2f}) "
                f"(файлов: {files_downloaded}, с данными: {files_parsed_ok}) → использую синтетику"
            )
            return self._generate_synthetic_tec(lat, lon, start_time, end_time)

        df_combined = pd.concat(all_dfs)
        df_combined = df_combined.sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

        if df_combined['tec_value'].isnull().any():
            df_combined['tec_value'] = df_combined['tec_value'].interpolate(method='time')

        logger.info(
            f"🛰️ TEC: {len(df_combined)} записей для ({lat:.2f}, {lon:.2f}) "
            f"[{start_time.date()} — {end_time.date()}]"
        )
        return df_combined
    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3) -> dict:
        """
        Собирает TEC и анализирует аномалии.
        
        ✅ ИСПРАВЛЕНО: использует in-memory кэш по событию,
        чтобы prepare_training_data не вызывал это 6 раз для одного события.
        """
        event_time = self._to_utc(event_time)

        # Ключ кэша
        cache_key = (round(lat, 2), round(lon, 2), event_time.date().isoformat())
        if cache_key in self._event_cache:
            logger.debug(f"♻️ Ионосфера из кэша: {cache_key}")
            return self._event_cache[cache_key]

        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)

        # Target
        target_df = self.fetch_tec_for_point(lat, lon, start, end)

        if target_df.empty:
            result = {
                'tec': pd.DataFrame(),
                'background': pd.DataFrame(),
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': 0.0,
                'std_bg': 0.0,
                'level': 'normal',
                'n_historical_years': 0,
                'year_over_year_change': 0.0,
                'details': 'Нет данных Target'
            }
            self._event_cache[cache_key] = result
            return result

        # Background (прошлый год)
        try:
            bg_start = start.replace(year=start.year - 1)
            bg_end = end.replace(year=end.year - 1)
        except ValueError:
            # 29 февраля в високосный год
            bg_start = start.replace(year=start.year - 1, day=28)
            bg_end = end.replace(year=end.year - 1, day=28)

        bg_df = self.fetch_tec_for_point(lat, lon, bg_start, bg_end)
        if bg_df.empty:
            result = {
                'tec': target_df,
                'background': pd.DataFrame(),
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': 0.0,
                'std_bg': 0.0,
                'level': 'normal',
                'n_historical_years': 0,
                'year_over_year_change': 0.0,
                'details': 'Нет данных фона (прошлый год)'
            }
            self._event_cache[cache_key] = result
            return result

        mean_bg = float(bg_df['tec_value'].mean())
        std_bg = float(bg_df['tec_value'].std())

        if std_bg <= 0:
            result = {
                'tec': target_df,
                'background': bg_df,
                'is_anomaly': False,
                'z_score_max': 0.0,
                'z_score_min': 0.0,
                'mean_bg': mean_bg,
                'std_bg': std_bg,
                'level': 'normal',
                'n_historical_years': 1,
                'year_over_year_change': 0.0,
                'details': 'Стандартное отклонение фона = 0'
            }
            self._event_cache[cache_key] = result
            return result

        target_df = target_df.copy()
        target_df['z_score'] = (target_df['tec_value'] - mean_bg) / std_bg
        z_max = float(target_df['z_score'].max())
        z_min = float(target_df['z_score'].min())

        anomaly_detected = z_max > 2.0 or z_min < -2.0

        # Определение уровня аномалии
        max_abs_z = max(abs(z_max), abs(z_min))
        if max_abs_z > 3.0:
            level = 'critical'
        elif max_abs_z > 2.5:
            level = 'high'        elif max_abs_z > 2.0:
            level = 'moderate'
        else:
            level = 'normal'

        # YoY change
        yoy_change = 0.0
        target_mean = float(target_df['tec_value'].mean())
        if mean_bg > 0:
            yoy_change = ((target_mean - mean_bg) / mean_bg) * 100

        if anomaly_detected:
            details = f"Аномалия: max={z_max:.2f}σ, min={z_min:.2f}σ"
            logger.critical(f"   🔴🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ: {details}")
        else:
            details = f"В норме (max={z_max:.2f}σ)"
            logger.debug(f"   ✅ Ионосфера: {details}")

        result = {
            'tec': target_df,
            'background': bg_df,
            'is_anomaly': anomaly_detected,
            'z_score_max': z_max,
            'z_score_min': z_min,
            'mean_bg': mean_bg,
            'std_bg': std_bg,
            'level': level,
            'n_historical_years': 1,
            'year_over_year_change': round(yoy_change, 2),
            'details': details
        }

        self._event_cache[cache_key] = result
        return result

    def clear_cache(self):
        """Очистить in-memory кэш."""
        self._event_cache.clear()
        logger.info("🗑️ Кэш ионосферы очищен")


# ═══════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = IonosphereCollector()
    # Тест: реальные данные за последние 3 дня
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Реальные TEC (3 дня назад)")
    end = datetime.now(timezone.utc) - timedelta(days=3)
    start = end - timedelta(days=2)
    df = collector.fetch_tec_for_point(35.0, 140.0, start, end)
    print(f"Записей: {len(df)}")
    if not df.empty:
        print(f"Среднее TEC: {df['tec_value'].mean():.2f} TECU")
        print(f"Максимум: {df['tec_value'].max():.2f} TECU")

    # Тест: аномалии
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Анализ аномалий")
    result = collector.fetch_for_event(
        event_time=datetime.now(timezone.utc) - timedelta(days=3),
        lat=35.0, lon=140.0,
        days_before=5, days_after=0
    )
    print(f"Аномалия: {result['is_anomaly']}")
    print(f"Уровень: {result['level']}")
    print(f"Z-max: {result['z_score_max']:.2f}")
    print(f"Z-min: {result['z_score_min']:.2f}")
    print(f"Детали: {result['details']}")

    # Тест: кэширование
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Кэширование (должно быть мгновенно)")
    result2 = collector.fetch_for_event(
        event_time=datetime.now(timezone.utc) - timedelta(days=3),
        lat=35.0, lon=140.0,
        days_before=5, days_after=0
    )
    print(f"Из кэша: {result2['details']}")
