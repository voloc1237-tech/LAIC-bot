#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор TEC и ROTI через IONEX-файлы.
Исправленная и рабочая версия.

Зависимости:
    pip install pandas numpy gnssanalysis
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
from gnssanalysis.ionex import IonexMap

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class IonosphereCollector:
    """
    Сборщик ионосферных данных (TEC, ROTI) через IONEX-файлы.
    Использует FTP-зеркало IGS для скачивания (открытый доступ).
    """

    # FTP-зеркало IGS (открытое, не требует авторизации)
    CDDIS_FTP_BASE = "ftp://igs.ensg.ign.fr/pub/igs/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {cache_dir}")

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Приводит datetime к UTC."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_ionex_filename(self, date: datetime, version: str = "codg") -> str:
        """
        Формирует имя IONEX-файла по дате.
        Формат: codgYYYYDDD.01i (например, codg2023001.01i)
        """
        doy = date.timetuple().tm_yday
        return f"{version}{date.year}{doy:03d}.01i"

    def _download_ionex(self, date: datetime) -> str | None:
        """Скачивает IONEX-файл с FTP-зеркала IGS."""
        filename = self._get_ionex_filename(date)
        local_path = os.path.join(self.cache_dir, filename)

        # Проверка кэша (только если файл существует и не пустой)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.debug(f"📂 IONEX файл в кэше: {filename}")
            return local_path

        year = date.strftime("%Y")
        doy = date.timetuple().tm_yday
        url = f"{self.CDDIS_FTP_BASE}/{year}/{doy:03d}/{filename}"

        try:
            logger.info(f"📥 Скачивание IONEX: {url}")
            # timeout в секундах
            urllib.request.urlretrieve(url, local_path, timeout=60)

            # Дополнительная проверка: файл должен быть не пустым
            if os.path.getsize(local_path) == 0:
                logger.warning(f"⚠️ Скачанный файл пуст: {local_path}. Удаляю.")
                os.remove(local_path)
                return None

            logger.info(f"✅ IONEX сохранён: {local_path}")
            return local_path

        except urllib.error.HTTPError as e:
            # Для FTP это редкость, но обрабатываем
            if e.code == 404:
                logger.warning(f"⚠️ IONEX файл не найден (404): {filename}")
            else:
                logger.warning(f"⚠️ Ошибка HTTP при скачивании: {e}")
            return None
        except urllib.error.URLError as e:
            logger.warning(f"⚠️ Ошибка сети при скачивании: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Неожиданная ошибка скачивания: {e}")
            # Если файл частично скачался — удаляем, чтобы не использовать битый
            if os.path.exists(local_path):
                os.remove(local_path)
            return None

    def fetch_tec_for_point(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получить VTEC (вертикальный TEC) для указанной точки за период.
        Возвращает DataFrame с индексом времени и колонкой 'tec'.
        """
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        records = []
        # Начинаем с 00:00 даты начала
        current_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_time:
            ionex_path = self._download_ionex(current_date)

            if not ionex_path:
                # Нет файла — переходим к следующему дню
                current_date += timedelta(days=1)
                continue

            try:
                ionmap = IonexMap.from_file(ionex_path)

                # IONEX обычно содержит карты каждые 2 часа (0, 2, ..., 22 UTC)
                for hour in range(0, 24, 2):
                    time_query = current_date.replace(hour=hour)
                    try:
                        tec_val = ionmap.get_tec(lat=lat, lon=lon, time=time_query)
                        if tec_val is not None and not np.isnan(tec_val):
                            records.append({
                                "time": time_query,
                                "tec": float(tec_val),
                                "lat": lat,
                                "lon": lon
                            })
                    except Exception as e:
                        # Это нормально, если точка вне покрытия карты
                        logger.debug(f"Нет TEC для ({lat}, {lon}) на {time_query}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"⚠️ Ошибка парсинга IONEX {ionex_path}: {e}")

            current_date += timedelta(days=1)

        if not records:
            logger.warning(f"⚠️ Нет TEC данных для ({lat:.2f}, {lon:.2f}) за период")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Интерполяция пропущенных значений (если есть)
        if df["tec"].isnull().any():
            df["tec"] = df["tec"].interpolate(method="time")

        logger.info(f"🛰️ TEC: {len(df)} записей для ({lat:.2f}, {lon:.2f})")
        return df

    def fetch_roti(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получить ROTI (Rate of TEC Index) — показатель ионосферных возмущений.
        ROTI рассчитывается как скользящее стандартное отклонение TEC за окно 8 часов.
        """
        tec_df = self.fetch_tec_for_point(lat, lon, start_time, end_time)
        if tec_df.empty:
            return pd.DataFrame()

        # Приводим к регулярной сетке 2 часа, чтобы rolling работал корректно
        tec_df = tec_df.asfreq("2H")

        # ROTI = std(TEC) за окно 8 часов (4 точки по 2 часа)
        tec_df["roti"] = tec_df["tec"].rolling("8H", min_periods=2).std()

        logger.info(f"🛰️ ROTI: {len(tec_df)} записей")
        return tec_df[["roti"]]

    def fetch_for_event(self, event_time: datetime, lat: float, lon: float, days_before: int = 7, days_after: int = 3) -> dict:
        """
        Собирает TEC и ROTI за период вокруг события.
        Возвращает словарь: {'tec': df, 'roti': df}
        """
        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)

        tec_df = self.fetch_tec_for_point(lat, lon, start, end)
        roti_df = (
            self.fetch_roti(lat, lon, start, end) if not tec_df.empty else pd.DataFrame()
        )

        return {"tec": tec_df, "roti": roti_df}


if __name__ == "__main__":
    # Пример использования
    collector = IonosphereCollector(cache_dir="data/ionex_cache")

    # Координаты (например, Москва)
    lat, lon = 55.75, 37.62

    # Время события (в UTC!)
    event_time = datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    print(f"🚀 Запуск сбора данных для ({lat}, {lon}) вокруг события {event_time}")
    result = collector.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)

    tec_df = result["tec"]
    roti_df = result["roti"]

    if not tec_df.empty:
        print("\n--- Первые 10 записей TEC ---")
        print(tec_df.head(10))
        print(f"\nДиапазон времени: {tec_df.index.min()} — {tec_df.index.max()}")
        print(f"Среднее TEC: {tec_df['tec'].mean():.2f} TECU")
    else:
        print("❌ TEC данные не получены.")

    if not roti_df.empty:
        print("\n--- Первые 10 записей ROTI ---")
        print(roti_df.head(10))
        print(f"Максимальный ROTI: {roti_df['roti'].max():.2f}")
    else:
        print("❌ ROTI данные не получены.")
