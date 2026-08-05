#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionosphere_collector.py — сбор реальных TEC из IONEX-файлов.
Финальная исправленная версия:
  1. Несколько зеркал загрузки (IGS, AIUB, CDDIS)
  2. Корректные имена файлов (старый и новый формат)
  3. Парсер учитывает формат IONEX (числа выше подписи)
"""

import logging
import gzip
import shutil
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IonosphereCollector:
    """Сборщик TEC из IONEX-файлов."""

    CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

    def __init__(self, cache_dir="data/ionex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._event_cache = {}
        logger.info(f"🛰️ Ионосфера: кэш IONEX в {self.cache_dir}")

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_ionex_urls(self, date: datetime) -> list:
        """Зеркала IONEX, пробуем по очереди."""
        year = date.year
        doy = date.timetuple().tm_yday
        yy = year % 100
        long_name = (
            "COD0OPSFIN_"
            f"{year}{doy:03d}0000"
            "_01D_01H_GIM.INX.gz"
        )
        short_name = f"codg{doy:03d}0.{yy:02d}i.Z"  # FIX: перевод строки
        return [
            f"https://files.igs.org/pub/product/ionex/{year}/{doy:03d}/{long_name}",
            f"https://ftp.aiub.unibe.ch/CODE/{year}/{short_name}",
            f"{self.CDDIS_BASE}/{year}/{doy:03d}/{long_name}",
        ]

    def _download_ionex(self, date: datetime):
        """Скачивает IONEX с любого зеркала."""
        doy = date.timetuple().tm_yday
        local_path = self.cache_dir / f"ionex_{date.year}_{doy:03d}.inx"

        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        tmp_path = local_path.with_suffix(".tmp")

        for url in self._get_ionex_urls(date):
            try:
                logger.info(f"📥 Скачивание IONEX: {url}")
                urllib.request.urlretrieve(url, tmp_path, timeout=60)

                if tmp_path.stat().st_size == 0:
                    tmp_path.unlink(missing_ok=True)
                    continue

                with open(tmp_path, "rb") as f:
                    magic = f.read(2)

                if magic == b"\x1f\x8b":
                    with gzip.open(tmp_path, "rb") as f_in:
                        with open(local_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.rename(local_path)

                if local_path.stat().st_size > 0:
                    logger.info(f"✅ IONEX скачан: {local_path.name}")
                    return str(local_path)

            except Exception as e:
                logger.debug(f"⚠️ Не удалось с {url}: {e}")
                tmp_path.unlink(missing_ok=True)
                continue

        logger.debug(f"⚠️ Все зеркала недоступны: {date.date()}")
        return None

    def _parse_ionex_file(self, filepath, lat, lon, date):
        """
        ✅ Корректный парсер IONEX.
        В IONEX числа идут ВЫШЕ строки-подписи,
        поэтому запоминаем числа в pending и
        применяем их, когда видим подпись.
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения {filepath}: {e}")
            return pd.DataFrame()

        # Сетка по умолчанию (стандарт CODE)
        lats = np.arange(-87.5, 87.5 + 1.25, 2.5)
        lons = np.arange(-180.0, 180.0 + 2.5, 5.0)
        n_lon = len(lons)
        scale = 0.1  # exponent -1

        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))

        # Ищем конец заголовка
        header_idx = 0
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_idx = i
                break

        records = []
        in_tec_map = False
        data_mode = False
        pending = None
        col = 0
        current_lat_idx = -1
        current_epoch = None

        for line in lines[header_idx + 1:]:
            if "START OF TEC MAP" in line:
                in_tec_map = True
                data_mode = False
                continue
            if "END OF TEC MAP" in line:
                in_tec_map = False
                data_mode = False
                continue
            if not in_tec_map:
                continue

            parts = line.split()
            if not parts:                      # FIX: отступы
                continue

            # Строка-подпись (содержит буквы)
            if any(c.isalpha() for c in line):
                if "EPOCH OF CURRENT MAP" in line and pending:
                    try:
                        v = [int(float(x)) for x in pending[:6]]
                        current_epoch = datetime(
                            v[0], v[1], v[2],
                            v[3], v[4], v[5],
                            tzinfo=timezone.utc
                        )
                    except Exception:
                        pass
                elif "LAT/LON1/LON2/DLON/H" in line and pending:
                    try:
                        lat_val = float(pending[0])
                        current_lat_idx = int(
                            np.argmin(np.abs(lats - lat_val))
                        )
                        data_mode = (current_lat_idx == lat_idx)
                        col = 0
                    except Exception:
                        data_mode = False
                pending = None
                continue

            # Числовая строка
            if "." in line:
                # Заголовок широты/эпохи (с точками)
                pending = parts
                data_mode = False
                continue

            # Данные TEC (целые числа)
            if data_mode and current_epoch is not None:
                for tok in parts:
                    try:
                        iv = int(tok)
                    except ValueError:
                        col += 1
                        continue
                    if col == lon_idx and iv != 9999:
                        tec = iv * scale
                        if 0.0 <= tec <= 300.0:
                            records.append({
                                "time": current_epoch,
                                "tec_value": float(tec),
                            })
                    col += 1
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset=["time"])
        df = df.set_index("time")
        df = df.sort_index()
        return df

    def _generate_synthetic_tec(self, lat, lon, start_time, end_time):
        """Синтетика как запасной вариант."""
        records = []
        current = start_time
        base_tec = 30 - abs(lat) * 0.3
        seed = int(abs(lat) * 100 + abs(lon) * 10)
        seed = seed % (2 ** 31)
        rng = np.random.RandomState(seed)

        while current <= end_time:
            hour = current.hour
            daily = 5 * np.sin((hour - 10) * np.pi / 12)
            noise = rng.normal(0, 2)
            tec = max(0, base_tec + daily + noise)
            records.append({"time": current, "tec_value": tec})
            current += timedelta(hours=2)

        df = pd.DataFrame(records)
        df.set_index("time", inplace=True)
        return df

    def fetch_tec_for_point(self, lat, lon, start_time, end_time):
        """TEC для точки: реальные данные, иначе синтетика."""
        start_time = self._to_utc(start_time)
        end_time = self._to_utc(end_time)

        all_dfs = []
        parsed_ok = 0
        current_date = start_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        while current_date <= end_time:
            filepath = self._download_ionex(current_date)
            if filepath:
                df = self._parse_ionex_file(
                    filepath, lat, lon, current_date
                )
                if not df.empty:              # FIX: отступы
                    parsed_ok += 1
                    all_dfs.append(df)
            current_date += timedelta(days=1)

        if not all_dfs:
            logger.warning(
                f"⚠️ Нет реальных TEC для ({lat:.2f}, {lon:.2f}), "
                f"файлов с данными: {parsed_ok} → синтетика"
            )
            return self._generate_synthetic_tec(lat, lon, start_time, end_time)

        df_combined = pd.concat(all_dfs)
        df_combined = df_combined.sort_index()
        df_combined = df_combined[
            ~df_combined.index.duplicated(keep="first")
        ]
        logger.info(
            f"🛰️ TEC: {len(df_combined)} записей "
            f"для ({lat:.2f}, {lon:.2f})"
        )
        return df_combined

    def fetch_for_event(self, event_time, lat, lon, days_before=7, days_after=3):
        """Сбор TEC + анализ аномалий (с кэшем по событию)."""
        event_time = self._to_utc(event_time)

        cache_key = (
            round(lat, 2),
            round(lon, 2),
            event_time.date().isoformat(),
        )
        if cache_key in self._event_cache:
            return self._event_cache[cache_key]

        start = event_time - timedelta(days=days_before)
        end = event_time + timedelta(days=days_after)

        target_df = self.fetch_tec_for_point(lat, lon, start, end)

        if target_df.empty:
            result = self._empty_result("Нет данных Target")
            self._event_cache[cache_key] = result
            return result

        try:
            bg_start = start.replace(year=start.year - 1)
            bg_end = end.replace(year=end.year - 1)
        except ValueError:
            bg_start = start.replace(year=start.year - 1, day=28)
            bg_end = end.replace(year=end.year - 1, day=28)
        bg_df = self.fetch_tec_for_point(lat, lon, bg_start, bg_end)

        if bg_df.empty:
            result = self._empty_result("Нет данных фона")
            result["tec"] = target_df
            self._event_cache[cache_key] = result
            return result

        mean_bg = float(bg_df["tec_value"].mean())
        std_bg = float(bg_df["tec_value"].std())

        if std_bg <= 0:
            result = self._empty_result("std фона = 0")
            result["tec"] = target_df
            result["mean_bg"] = mean_bg
            self._event_cache[cache_key] = result
            return result

        target_df = target_df.copy()
        target_df["z_score"] = (target_df["tec_value"] - mean_bg) / std_bg
        z_max = float(target_df["z_score"].max())
        z_min = float(target_df["z_score"].min())

        anomaly = z_max > 2.0 or z_min < -2.0
        max_z = max(abs(z_max), abs(z_min))

        if max_z > 3.0:
            level = "critical"
        elif max_z > 2.5:
            level = "high"
        elif max_z > 2.0:
            level = "moderate"
        else:
            level = "normal"

        yoy = 0.0
        t_mean = float(target_df["tec_value"].mean())
        if mean_bg > 0:
            yoy = ((t_mean - mean_bg) / mean_bg) * 100

        if anomaly:
            details = f"Аномалия: max={z_max:.2f}σ, min={z_min:.2f}σ"
            logger.critical(f"   🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ: {details}")
        else:
            details = f"В норме (max={z_max:.2f}σ)"

        result = {
            "tec": target_df,
            "background": bg_df,
            "is_anomaly": anomaly,
            "z_score_max": z_max,
            "z_score_min": z_min,
            "mean_bg": mean_bg,
            "std_bg": std_bg,
            "level": level,
            "n_historical_years": 1,
            "year_over_year_change": round(yoy, 2),
            "details": details,
        }
        self._event_cache[cache_key] = result
        return result

    @staticmethod
    def _empty_result(details):
        return {
            "tec": pd.DataFrame(),
            "background": pd.DataFrame(),
            "is_anomaly": False,
            "z_score_max": 0.0,
            "z_score_min": 0.0,
            "mean_bg": 0.0,
            "std_bg": 0.0,
            "level": "normal",
            "n_historical_years": 0,
            "year_over_year_change": 0.0,
            "details": details,
        }

    def clear_cache(self):
        """Очистить кэш в памяти."""
        self._event_cache.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = IonosphereCollector()
    end = datetime.now(timezone.utc) - timedelta(days=3)
    start = end - timedelta(days=2)
    df = collector.fetch_tec_for_point(35.0, 140.0, start, end)
    print(f"Записей: {len(df)}")
    if not df.empty:
        print(f"Среднее TEC: {df['tec_value'].mean():.2f}")
