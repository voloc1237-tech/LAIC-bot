#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — ГЛАВНЫЙ ФАЙЛ LAIC-бота (исправленная версия)
Собирает → Анализирует → Прогнозирует → Отправляет
"""

import os
import sys
import logging
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ — ДОЛЖНО БЫТЬ ПЕРВЫМ!
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('LAIC')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════════
# ИМПОРТЫ МОДУЛЕЙ (ПОСЛЕ НАСТРОЙКИ ЛОГГЕРА)
# ═══════════════════════════════════════════════════════════════
from aftershock_filter import filter_aftershocks
from fault_zones import FaultZoneAnalyzer
#from dotenv import load_dotenv

#load_dotenv()  # ← ЭТО ЗАГРУЖАЕТ ПЕРЕМЕННЫЕ ИЗ .env

# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА СЛОВАРЯ РЕГИОНОВ
# ═══════════════════════════════════════════════════════════════
import pickle

def load_region_data():
    """Загружает словарь регионов с координатами."""
    data_dir = Path(__file__).parent / "data"
    pkl_path = data_dir / "region_mapping.pkl"

    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✅ Загружен словарь регионов: {len(data)} зон")
            return data
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки region_mapping.pkl: {e}")
            return {}
    else:
        logger.warning("⚠️ region_mapping.pkl не найден! Использую fallback.")
        return {}

REGION_DATA = load_region_data()

def get_zone_info(lat, lon, threshold=0.5):
    """
    Находит зону по координатам.
    Возвращает: (название_зоны, координаты) или (None, None)
    """
    if not REGION_DATA:
        return None, None

    lat = round(lat, 2)
    lon = round(lon, 2)

    for zone_id, info in REGION_DATA.items():
        z_lat, z_lon = info["coords"]
        if abs(z_lat - lat) <= threshold and abs(z_lon - lon) <= threshold:
            return info["name"], info["coords"]

    return None, None

# ═══════════════════════════════════════════════════════════════
# СОЗДАНИЕ ПАПОК
# ═══════════════════════════════════════════════════════════════
def ensure_directories():
    directories = [
        "data", "data/cache", "data/ionex_cache",
        "data/models", "data/models/lstm", "config",
        "data/space_weather_cache"  # ← новое для OMNI
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

ensure_directories()

# ═══════════════════════════════════════════════════════════════
# ОСТАЛЬНЫЕ ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════

# GEE
try:
    from ee_collector import GEEInitializer, get_all_gee_data
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("⚠️ Earth Engine модуль не найден")

# Визуализация
try:
    from visualizer import (
        plot_magnitude_series,
        create_interactive_report,
        create_anomaly_plot,
        create_iono_anomaly_plot,
        create_solar_activity_plot
    )
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    VISUALIZER_AVAILABLE = False
    logger.warning(f"⚠️ Модуль визуализации не найден: {e}")

# Телеграм
try:
    from telegram_sender import send_sync, alert_sync, send_photo, send_document
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    logger.warning("⚠️ Модуль Telegram не найден")

# Погода
try:
    from weather_collector import WeatherCollector
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False

# Ионосфера
try:
    from ionosphere_historical import EnhancedIonosphereCollector
    IONO_AVAILABLE = True
    logger.info("✅ Исторический анализатор ионосферы доступен (2000-2023)")
except ImportError:
    try:
        from ionosphere_collector import IonosphereCollector
        IONO_AVAILABLE = True
        logger.info("ℹ️ Legacy сборщик ионосферы")
    except ImportError:
        IONO_AVAILABLE = False
        logger.warning("⚠️ Сборщик ионосферы недоступен")

# Космическая погода
try:
    from space_weather import get_collector, get_data_quality
    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    logger.warning("⚠️ SpaceWeather недоступен")

# ИИ-анализ аномалий
try:
    from anomaly_detector import AnomalyDetector
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False

# Swarm
try:
    from swarm_collector import SwarmCollector
    SWARM_AVAILABLE = True
    logger.info("✅ SwarmCollector загружен")
except ImportError:
    SWARM_AVAILABLE = False
    logger.warning("⚠️ SwarmCollector не доступен")

# LSTM
try:
    from tensorflow.keras.models import load_model
    import pickle
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("⚠️ LSTM модуль недоступен (tensorflow не установлен)")

# LAIC прогнозирование (градиентный бустинг)
try:
    from earthquake_predictor import EarthquakePredictor as LAICPredictor
    LAIC_PREDICTOR_AVAILABLE = True
except ImportError:
    LAIC_PREDICTOR_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("🚀 LAIC EARTHQUAKE MONITOR")

    current_time = datetime.now(timezone.utc)
    logger.info(f"⏰ Текущая дата: {current_time.isoformat()}")
    logger.info("=" * 70)

    # Проверка токенов
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.error("❌ TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы!")
        sys.exit(1)
    logger.info("✅ Токены Telegram найдены")

    # Загрузка настроек
    import yaml
    try:
        with open('config/settings.yaml', 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("❌ config/settings.yaml не найден!")
        sys.exit(1)
    logger.info(f"✅ Настройки загружены: {len(settings.get('regions', {}))} регионов")

    # Подключение GEE
    ee_initialized = False
    if GEE_AVAILABLE:
        try:
            ee_initialized = GEEInitializer.init()
            if ee_initialized:
                logger.info("✅ Earth Engine подключен!")
        except Exception as e:
            logger.warning(f"⚠️ Earth Engine ошибка: {e}")

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1: СБОР ДАННЫХ USGS
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📥 ЭТАП 1: Сбор данных USGS")
    logger.info("=" * 70)

    from data_collector import collect_all_regions
    global_mode = settings.get('analysis', {}).get('global_mode', True)
    data = collect_all_regions(settings, global_mode=global_mode)

    total = sum(len(df) for df in data.values())
    logger.info(f"\n📊 Всего событий: {total}")

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1a: ФИЛЬТРАЦИЯ АФТЕРШОКОВ
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🔍 ЭТАП 1a: Фильтрация афтершоков")
    logger.info("=" * 70)

    data_filtered = {}
    aftershock_counts = {}

    for region_name, df in data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            data_filtered[region_name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            aftershock_counts[region_name] = 0
            continue

        df_filtered, df_aftershocks = filter_aftershocks(
            df, time_window_days=7, distance_km=50, mag_threshold=0.5
        )
        data_filtered[region_name] = df_filtered
        aftershock_counts[region_name] = len(df_aftershocks)
        logger.info(f"   📌 {region_name}: афтершоков {len(df_aftershocks)} из {len(df)}")

    data = data_filtered
    total_filtered = sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame))
    logger.info(f"\n📊 После фильтрации: {total_filtered} событий")

    # Определяем конец реальных данных
    data_end_time = None
    for df in data.values():
        if not df.empty and 'time' in df.columns:
            t = df['time'].max()
            if data_end_time is None or t > data_end_time:
                data_end_time = pd.to_datetime(t).tz_localize('UTC') if pd.to_datetime(t).tzinfo is None else pd.to_datetime(t)

    data_lag_days = (current_time - data_end_time).days if data_end_time else 0
    if data_lag_days > 2:
        logger.warning(f"⚠️ Данные USGS отстают на {data_lag_days} дней!")
        effective_time = data_end_time
    else:
        effective_time = current_time

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1b: СБОР СПУТНИКОВЫХ ДАННЫХ (GEE)
    # ═══════════════════════════════════════════════════════════════
    lst_cache = {}
    if ee_initialized:
        logger.info("\n" + "=" * 70)
        logger.info("🛰️ ЭТАП 1b: Сбор спутниковых данных (GEE)")
        logger.info("=" * 70)

        for region_name, region_data in data.items():
            if region_data.empty:
                continue

            last_event = region_data.iloc[0]
            lat = last_event['latitude']
            lon = last_event['longitude']

            end_date = effective_time.strftime('%Y-%m-%d')
            start_date = (effective_time - timedelta(days=7)).strftime('%Y-%m-%d')

            try:
                gee_result = get_all_gee_data(lat, lon, start_date, end_date)

                # Логируем что получили
                for key in ['lst', 'so2', 'co', 'ch4']:
                    val = gee_result.get(key, {})
                    if val and val.get(f'{key}_celsius' if key == 'lst' else key) is not None:
                        logger.info(f"✅ {region_name}: {key.upper()} получен")
                    else:
                        logger.warning(f"⚠️ {region_name}: {key.upper()} недоступен")

                lst_cache[region_name] = gee_result

            except Exception as e:
                logger.warning(f"⚠️ Ошибка GEE для {region_name}: {e}")
                lst_cache[region_name] = None


    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1c: СБОР ДОПОЛНИТЕЛЬНЫХ ДАННЫХ
    # ═══════════════════════════════════════════════════════════════
    min_mag_for_detail = settings.get('analysis', {}).get('detail_min_magnitude', 5.0)
    event_weather = {}
    event_iono = {}
    event_space = {}
    event_swarm = {}

    all_events = pd.concat(
        [df for df in data.values() if isinstance(df, pd.DataFrame) and not df.empty],
        ignore_index=True
    ) if data else pd.DataFrame()

    strong_events = all_events[all_events['magnitude'] >= min_mag_for_detail] if not all_events.empty else pd.DataFrame()

    if not strong_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info(f"📌 ЭТАП 1c: Детальный сбор для {len(strong_events)} событий M≥{min_mag_for_detail}")
        logger.info("=" * 70)

        weather = WeatherCollector() if WEATHER_AVAILABLE else None
        iono = EnhancedIonosphereCollector() if IONO_AVAILABLE else None
        space = SpaceWeatherCollector() if SPACE_AVAILABLE else None

        # Проверка качества космических данных
        if space:
            try:
                check_time = datetime.now(timezone.utc)
                test_start = check_time - timedelta(days=10)
                quality = get_data_quality(test_start, check_time)
                for param, info in quality.items():
                    if info['status'] == 'SYNTHETIC':
                        logger.error(f"🔴 {param}: полностью синтетические данные!")
                    elif info['status'] == 'PARTIAL':
                        logger.warning(f"⚠️ {param}: частично реальные ({info['real_pct']:.0f}%)")
                    else:
                        logger.info(f"✅ {param}: реальные данные ({info['real_pct']:.0f}%)")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка проверки качества: {e}")

        for idx, row in strong_events.iterrows():
            event_id = row['id']
            lat = row['latitude']
            lon = row['longitude']
            event_time = row['time']
            mag = row['magnitude']

            if row.get('is_aftershock', False):
                continue

            logger.info(f"🔍 Событие {event_id} M{mag:.1f} ({event_time.date()})")

            # Погода
            if weather:
                try:
                    wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    if not wdf.empty:
                        event_weather[event_id] = wdf
                        logger.info(f"   🌡️ Погода: {len(wdf)} дней")
                except Exception as e:
                    logger.debug(f"   Погода: {e}")

            # ===== ИОНОСФЕРА (TEC) — приоритет IONEX, резерв Swarm =====
            iono_data = None
            iono_anomaly = False

            # 1. Пробуем IONEX
            if IONO_AVAILABLE:
                try:
                    iono_data = iono.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    if iono_data and iono_data.get('tec') is not None and not iono_data['tec'].empty:
                        event_iono[event_id] = iono_data   # <-- ДОБАВИТЬ!
                        tec_mean = iono_data['tec']['tec_value'].mean()
                        logger.info(f"   🛰️ IONEX TEC: {tec_mean:.2f} TECU")
                        if 'z_score' in iono_data['tec'].columns:
                            z_max = iono_data['tec']['z_score'].max()
                            if z_max > 2.0:
                                logger.critical(f"   🔴🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ (IONEX): Z={z_max:.2f}σ")
                                iono_anomaly = True
                        # ✅ СОХРАНЯЕМ
                        event_iono[event_id] = iono_data
                except Exception as e:
                    logger.warning(f"   ⚠️ IONEX ошибка: {e}")

            # 2. Если IONEX не дал данных — пробуем Swarm
            if (iono_data is None or iono_data.get('tec') is None or iono_data['tec'].empty) and SWARM_AVAILABLE:
                logger.info("   🔄 IONEX нет данных, пробуем Swarm...")
                try:
                    swarm = SwarmCollector()
                    swarm_data = swarm.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)

                    tec_df = swarm_data.get('tec')
                    if tec_df is not None and not tec_df.empty:
                        tec_mean = swarm.get_tec_mean(tec_df)
                        logger.info(f"   🛰️ Swarm TEC: {tec_mean:.2f} TECU")
                        # ✅ СОХРАНЯЕМ
                        event_iono[event_id] = swarm_data

                    mag_df = swarm_data.get('magnetic')
                    if mag_df is not None and not mag_df.empty:
                        anomaly = swarm.detect_magnetic_anomaly(mag_df)
                        if anomaly['has_anomaly']:
                            logger.critical(f"   🔴🔴🔴 SWARM МАГНИТНАЯ АНОМАЛИЯ: Z={anomaly['max_z_score']:.2f}σ")
                            iono_anomaly = True
                            event_swarm[event_id] = swarm_data
                            break
                except Exception as e:
                    logger.warning(f"   ⚠️ Swarm ошибка: {e}")

            if not iono_anomaly:
                logger.info("   ✅ Ионосфера в норме")

            # ===== КОСМИЧЕСКАЯ ПОГОДА =====
            if space:
                try:
                    start = event_time - timedelta(days=7)
                    end = event_time + timedelta(days=3)
                    space_data = space.fetch_all_for_period(start, end)
                
                    # Проверяем, есть ли данные
                    has_data = False
                    for key, df in space_data.items():
                        if df is not None and not df.empty:
                            has_data = True
                            logger.info(f"   ☀️ {key}: {len(df)} записей")
                
                    if has_data:
                        event_space[event_id] = space_data
                        logger.info(f"   ☀️ Космическая погода: загружена")
                    else:
                        logger.warning(f"   ⚠️ Космическая погода: нет данных")
                except Exception as e:
                    logger.warning(f"   ⚠️ Космос ошибка: {e}")

        logger.info(f"✅ Собрано: погода {len(event_weather)}, ионосфера {len(event_iono)}, космос {len(event_space)}")
    else:
        logger.info("ℹ️ Нет сильных событий для детального сбора")
    
    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1d: ИИ-АНАЛИЗ АНОМАЛИЙ
    # ═══════════════════════════════════════════════════════════════
    if ANOMALY_AVAILABLE and not all_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info("🧠 ЭТАП 1d: ИИ-анализ аномалий")
        logger.info("=" * 70)

        events_for_ai = []
        for region_name, df in data.items():
            if df.empty:
                continue
            for _, row in df.iterrows():
                event_id = row.get('id', f'event_{row.name}')
                iono_info = event_iono.get(event_id, {})

                events_for_ai.append({
                    'id': event_id,
                    'region': region_name,
                    'magnitude': row.get('magnitude', 0),
                    'depth_km': row.get('depth_km', 0),
                    'time': row.get('time', None),
                    'iono_anomaly': iono_info.get('is_anomaly', False),
                    'z_score_max': iono_info.get('z_score_max', 0.0),
                    'z_score_min': iono_info.get('z_score_min', 0.0),
                    'iono_level': iono_info.get('level', 'normal'),
                    'historical_years': iono_info.get('n_historical_years', 0),
                    'yoy_change': iono_info.get('year_over_year_change', 0.0)
                })

        if len(events_for_ai) >= 10:
            # ИСПРАВЛЕНО: contamination не фиксирован, адаптивный
            n_strong = sum(1 for e in events_for_ai if e['magnitude'] >= 6.0)
            contamination = min(0.2, max(0.05, n_strong / len(events_for_ai)))

            detector = AnomalyDetector(contamination=contamination)
            anomaly_df = detector.analyze_events(events_for_ai)

            if not anomaly_df.empty:
                anomaly_counts = anomaly_df.groupby('region')['is_anomaly'].sum().to_dict()
                for region, count in anomaly_counts.items():
                    if count > 0:
                        logger.info(f"   🔴 {region}: {count} аномалий")

                if TG_AVAILABLE and chat_id:
                    for region_name in anomaly_df['region'].unique():
                        region_anomalies = anomaly_df[anomaly_df['region'] == region_name]
                        if len(region_anomalies) > 1:
                            try:
                                img_bytes = create_anomaly_plot(region_anomalies, region_name)
                                if img_bytes:
                                    send_photo(chat_id, img_bytes,
                                              caption=f"🧠 ИИ-анализ — {region_name}")
                            except Exception as e:
                                logger.error(f"❌ Ошибка отправки графика: {e}")
            else:
                logger.info("ℹ️ Аномалий не обнаружено")
        else:
            logger.info(f"ℹ️ Недостаточно данных ({len(events_for_ai)} < 10)")

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1e: LSTM-ПРОГНОЗИРОВАНИЕ (с учётом Swarm)
    # ═══════════════════════════════════════════════════════════════
    if LSTM_AVAILABLE and not all_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info("🔮 ЭТАП 1e: LSTM-прогнозирование (с Swarm)")
        logger.info("=" * 70)
    
        model_path = "models/lstm_full_model.keras"
        scaler_path = "models/lstm_full_scaler.pkl"
    
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning("⚠️ LSTM модель не найдена. Запустите train_full_lstm.py")
        else:
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("✅ LSTM модель загружена")
            
                # ===== ПРИЗНАКИ (СПИСОК) =====
                feature_names = ['magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                               'humidity_mean', 'tec_mean', 'kp_mean', 'dst_mean', 'f107_mean',
                               'swarm_anomaly']   # <-- ДОБАВЛЕНО
            
                # ===== СОБИРАЕМ ПРИЗНАКИ =====
                df_features = all_events[['id', 'magnitude', 'depth_km', 'time']].copy()
            
                for col in feature_names:
                    if col not in df_features.columns:
                        df_features[col] = 0.0
            
                for idx, row in df_features.iterrows():
                    event_id = row['id']
                
                    # LST
                    lst_data = lst_cache.get('global', {})
                    df_features.at[idx, 'lst_celsius'] = lst_data.get('lst_celsius', 0) if lst_data else 0
                
                    # Погода
                    weather_data = event_weather.get(event_id)
                    if weather_data is not None and hasattr(weather_data, 'empty') and not weather_data.empty:
                        df_features.at[idx, 'temp_mean'] = weather_data['temp_2m_mean'].mean()
                        df_features.at[idx, 'humidity_mean'] = weather_data['humidity_2m_mean'].mean()
                
                    # TEC
                    iono_data = event_iono.get(event_id, {})
                    tec_df = iono_data.get('tec', pd.DataFrame())
                    if tec_df is not None and hasattr(tec_df, 'empty') and not tec_df.empty:
                        df_features.at[idx, 'tec_mean'] = tec_df['tec_value'].mean()
                
                    # Космическая погода
                    space_data = event_space.get(event_id, {})
                    kp_df = space_data.get('kp', pd.DataFrame())
                    dst_df = space_data.get('dst', pd.DataFrame())
                    f107_df = space_data.get('f107', pd.DataFrame())
                    df_features.at[idx, 'kp_mean'] = kp_df['kp'].mean() if hasattr(kp_df, 'empty') and not kp_df.empty else 0
                    df_features.at[idx, 'dst_mean'] = dst_df['dst'].mean() if hasattr(dst_df, 'empty') and not dst_df.empty else 0
                    df_features.at[idx, 'f107_mean'] = f107_df['f107'].mean() if hasattr(f107_df, 'empty') and not f107_df.empty else 0
                
                    # ===== SWARM МАГНИТНАЯ АНОМАЛИЯ =====
                    swarm_info = event_swarm.get(event_id, {})
                    mag_df = swarm_info.get('magnetic', pd.DataFrame())
                    if mag_df is not None and hasattr(mag_df, 'empty') and not mag_df.empty:
                        # Используем SwarmCollector для обнаружения аномалии
                        from swarm_collector import SwarmCollector
                        swarm = SwarmCollector()
                        anomaly = swarm.detect_magnetic_anomaly(mag_df)
                        df_features.at[idx, 'swarm_anomaly'] = anomaly['max_z_score']
                    else:
                        df_features.at[idx, 'swarm_anomaly'] = 0.0
            
                # Удаляем строки с пропусками
                df_features = df_features.dropna(subset=feature_names)
            
                if len(df_features) >= 7:
                    last_events = df_features.tail(7)
                
                    # Проверяем наличие колонки
                    if 'swarm_anomaly' not in last_events.columns:
                        last_events['swarm_anomaly'] = 0.0
                
                    X = last_events[feature_names].values
                    X = X.reshape(1, 7, len(feature_names))
                    X_scaled = scaler.transform(X.reshape(-1, len(feature_names))).reshape(X.shape)
                
                    prob = model.predict(X_scaled, verbose=0)[0][0]
                
                    # Ограничиваем вероятность
                    if prob > 0.95:
                        prob = 0.85
                        logger.warning("⚠️ Вероятность >95% ограничена до 85%")
                
                    pred_mag = last_events['magnitude'].max() * (0.5 + prob * 0.5)
                
                    logger.info(f"🔮 LSTM Прогноз: вероятность M≥6.0 = {prob*100:.1f}%")
                    logger.info(f"   Прогнозируемая магнитуда: {pred_mag:.2f}")
                    logger.info(f"   Признаков: {len(feature_names)} (включая Swarm)")
                
                    # Отправка в Telegram
                    if TG_AVAILABLE and chat_id and prob > 0.2:
                        if prob > 0.5:
                            status = "⚠️ ПОВЫШЕННОЕ ВНИМАНИЕ! Возможно сильное землетрясение."
                        elif prob > 0.3:
                            status = "🔶 Умеренный риск. Рекомендуется следить за обновлениями."
                        else:
                            status = "✅ Ситуация стабильна. Риск низкий."
                    
                        forecast_msg = (
                            f"🧠 <b>LSTM ПРОГНОЗ (с Swarm)</b>\n"
                            f"{'─' * 30}\n"
                            f"Вероятность M≥6.0: <b>{prob*100:.1f}%</b>\n"
                            f"Прогнозируемая магнитуда: <b>{pred_mag:.2f}</b>\n"
                            f"Признаков: <b>{len(feature_names)}</b>\n"
                            f"{'─' * 30}\n"
                            f"{status}"
                        )
                        try:
                            url = f"https://api.telegram.org/bot{token}/sendMessage"
                            payload = {'chat_id': chat_id, 'text': forecast_msg, 'parse_mode': 'HTML'}
                            requests.post(url, data=payload, timeout=30)
                            logger.info("✅ LSTM прогноз отправлен в Telegram")
                        except Exception as e:
                            logger.error(f"❌ Ошибка отправки: {e}")
                else:
                    logger.info(f"ℹ️ Недостаточно данных для LSTM прогноза (нужно ≥7, есть {len(df_features)})")
                
            except Exception as e:
                logger.error(f"❌ Ошибка LSTM: {e}")
    else:
        logger.info("ℹ️ LSTM-прогнозирование пропущено")
    
                # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1e: LSTM-ПРОГНОЗИРОВАНИЕ (ОБНОВЛЕННАЯ ВЕРСИЯ)
    # ═══════════════════════════════════════════════════════════════
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔮 ЭТАП 1e: LSTM-прогнозирование")
    logger.info("=" * 70)

    if LSTM_AVAILABLE and not all_events.empty:
        # ИСПОЛЬЗУЕМ НОВУЮ ФИНАЛЬНУЮ МОДЕЛЬ
        model_path = "models/lstm_final_model.keras"
        scaler_path = "models/lstm_final_scaler.pkl"
        threshold_path = "models/lstm_final_threshold.pkl"  # опционально

        # Проверяем наличие новой модели, если нет - используем старую
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning("⚠️ Финальная LSTM модель не найдена. Пробую старую...")
            model_path = "models/lstm_model.keras"
            scaler_path = "models/lstm_scaler.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning("⚠️ LSTM модель не найдена. Запустите train_full_lstm_final.py")
            else:
                try:
                    # Загружаем старую модель
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    model = load_model(model_path)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    logger.info("✅ LSTM модель (старая) загружена")

                    # Используем старую логику прогноза...
                    # (код остаётся как был)
                except Exception as e:
                    logger.error(f"❌ Ошибка LSTM: {e}")
        else:
            try:
                # ЗАГРУЖАЕМ НОВУЮ ФИНАЛЬНУЮ МОДЕЛЬ
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                # Загружаем оптимальный порог (если есть)
                optimal_threshold = 0.5
                if os.path.exists(threshold_path):
                    with open(threshold_path, 'rb') as f:
                        optimal_threshold = pickle.load(f)
                    logger.info(f"📊 Оптимальный порог: {optimal_threshold:.3f}")

                logger.info("✅ Финальная LSTM модель загружена")

                # Получаем расширенный список признаков (как в обучении)
                feature_names_extended = [
                    'magnitude', 'depth_km', 'lst_celsius', 'temp_mean',
                    'humidity_mean',# 'tec_mean',# 'dst_mean',# 'f107_mean',
                    'magnitude_diff', 'depth_diff', 'temp_diff', 'humidity_diff',
                    'magnitude_rolling_mean', 'magnitude_rolling_std', 'kp_mean'
                ]

                # ===== СОБИРАЕМ ПРИЗНАКИ =====
                # Создаём DataFrame с признаками
                df_features = all_events[['id', 'magnitude', 'depth_km', 'time']].copy()

                # Инициализируем все колонки нулями
                for col in feature_names_extended:
                    if col not in df_features.columns:
                        df_features[col] = 0.0

                # Заполняем признаки для каждого события
                for idx, row in df_features.iterrows():
                    event_id = row['id']

                    # Базовые признаки уже есть (magnitude, depth_km)

                    # LST
                    lst_data = None
                    for region_name, lst_val in lst_cache.items():
                        if lst_val and isinstance(lst_val, dict):
                            lst_data = lst_val
                            break
                    if lst_data and isinstance(lst_data, dict):
                        df_features.at[idx, 'lst_celsius'] = lst_data.get('lst_celsius', 0)

                    # Погода
                    weather_data = event_weather.get(event_id)
                    if weather_data is not None and hasattr(weather_data, 'empty') and not weather_data.empty:
                        df_features.at[idx, 'temp_mean'] = weather_data['temp_2m_mean'].mean()
                        df_features.at[idx, 'humidity_mean'] = weather_data['humidity_2m_mean'].mean()

                    # TEC (ионосфера)
                    iono_data = event_iono.get(event_id, {})
                    if iono_data and isinstance(iono_data, dict):
                        tec_df = iono_data.get('tec', pd.DataFrame())
                        if tec_df is not None and hasattr(tec_df, 'empty') and not tec_df.empty:
                            df_features.at[idx, 'tec_mean'] = tec_df['tec_value'].mean()

                    # Космическая погода
                    space_data = event_space.get(event_id, {})
                    if space_data and isinstance(space_data, dict):
                        kp_df = space_data.get('kp', pd.DataFrame())
                        dst_df = space_data.get('dst', pd.DataFrame())
                        f107_df = space_data.get('f107', pd.DataFrame())
                        df_features.at[idx, 'kp_mean'] = kp_df['kp'].mean() if hasattr(kp_df, 'empty') and not kp_df.empty else 0
                        df_features.at[idx, 'dst_mean'] = dst_df['dst'].mean() if hasattr(dst_df, 'empty') and not dst_df.empty else 0
                        df_features.at[idx, 'f107_mean'] = f107_df['f107'].mean() if hasattr(f107_df, 'empty') and not f107_df.empty else 0

                # Вычисляем производные признаки
                df_features = df_features.sort_values('time').reset_index(drop=True)
                df_features['magnitude_diff'] = df_features['magnitude'].diff().fillna(0)
                df_features['depth_diff'] = df_features['depth_km'].diff().fillna(0)
                df_features['temp_diff'] = df_features['temp_mean'].diff().fillna(0)
                df_features['humidity_diff'] = df_features['humidity_mean'].diff().fillna(0)
                df_features['magnitude_rolling_mean'] = df_features['magnitude'].rolling(3, min_periods=1).mean().fillna(0)
                df_features['magnitude_rolling_std'] = df_features['magnitude'].rolling(3, min_periods=1).std().fillna(0)

                # Удаляем строки с пропусками
                df_features = df_features.dropna(subset=feature_names_extended)

                if len(df_features) >= 7:
                    # Берем последние 7 событий
                    last_events = df_features.tail(7)

                    # Диагностика
                    logger.info(f"📊 Входные данные для финальной LSTM:")
                    logger.info(f"   Количество событий: {len(last_events)}")
                    logger.info(f"   Магнитуды: {last_events['magnitude'].values}")
                    logger.info(f"   Средний Kp: {last_events['kp_mean'].mean():.2f}")
                    logger.info(f"   Средний TEC: {last_events['tec_mean'].mean():.2f}")

                    # Формируем последовательность
                    X = last_events[feature_names_extended].values
                    X = X.reshape(1, 7, len(feature_names_extended))
                    X_scaled = scaler.transform(X.reshape(-1, len(feature_names_extended))).reshape(X.shape)

                    # Прогноз
                    prob = model.predict(X_scaled, verbose=0)[0][0]

                    # Используем оптимальный порог
                    prediction = 1 if prob >= optimal_threshold else 0

                    # Прогнозируемая магнитуда (улучшенная формула)
                    max_mag_last_7 = last_events['magnitude'].max()
                    pred_mag = max_mag_last_7 * (0.5 + prob * 0.5)

                    # Диапазон погрешности
                    mag_error = 0.3 + (1 - prob) * 0.5

                    logger.info(f"🔮 Финальная LSTM Прогноз:")
                    logger.info(f"   Вероятность M≥6.0: {prob*100:.1f}% (порог: {optimal_threshold:.3f})")
                    logger.info(f"   Класс: {'ПОЗИТИВНЫЙ' if prediction else 'НЕГАТИВНЫЙ'}")
                    logger.info(f"   Прогнозируемая магнитуда: {pred_mag:.2f} ± {mag_error:.2f}")

                    # Отправка в Telegram при высоком риске
                    if TG_AVAILABLE and chat_id and prob > 0.3:
                        if prob > 0.6:
                            status = "🚨 ВЫСОКИЙ РИСК! Возможно сильное землетрясение в ближайшие дни."
                        elif prob > 0.4:
                            status = "⚠️ ПОВЫШЕННОЕ ВНИМАНИЕ! Умеренный риск."
                        else:
                            status = "🔶 Низкий риск. Ситуация стабильна."

                        forecast_msg = (
                            f"🧠 <b>ФИНАЛЬНАЯ LSTM ПРОГНОЗ</b>\n"
                            f"{'─' * 35}\n"
                            f"Вероятность M≥6.0: <b>{prob*100:.1f}%</b>\n"
                            f"Прогнозируемая магнитуда: <b>{pred_mag:.2f} ± {mag_error:.2f}</b>\n"
                            f"Оптимальный порог: {optimal_threshold:.3f}\n"
                            f"{'─' * 35}\n"
                            f"{status}\n"
                            f"{'─' * 35}\n"
                            f"📊 На основе: {len(last_events)} событий, 15 признаков"
                        )

                        try:
                            url = f"https://api.telegram.org/bot{token}/sendMessage"
                            payload = {'chat_id': chat_id, 'text': forecast_msg, 'parse_mode': 'HTML'}
                            requests.post(url, data=payload, timeout=30)
                            logger.info("✅ Финальный LSTM прогноз отправлен в Telegram")
                        except Exception as e:
                            logger.error(f"❌ Ошибка отправки: {e}")

                        # Отправляем график обучения (если есть)
                        plot_path = "models/lstm_final_training.png"
                        if os.path.exists(plot_path) and TG_AVAILABLE:
                            try:
                                with open(plot_path, 'rb') as f:
                                    send_photo(chat_id, f.read(), caption="📈 График обучения финальной LSTM")
                                logger.info("✅ График обучения отправлен")
                            except Exception as e:
                                logger.warning(f"⚠️ Не удалось отправить график: {e}")
                else:
                    logger.info(f"ℹ️ Недостаточно данных для LSTM прогноза (нужно ≥7, есть {len(df_features)})")

            except Exception as e:
                logger.error(f"❌ Ошибка финальной LSTM: {e}")
                # Резерв: пробуем старую модель
                logger.info("🔄 Пробуем старую LSTM модель...")
                try:
                    old_model_path = "models/lstm_model.keras"
                    old_scaler_path = "models/lstm_scaler.pkl"
                    if os.path.exists(old_model_path) and os.path.exists(old_scaler_path):
                        model = load_model(old_model_path)
                        with open(old_scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        logger.info("✅ Старая LSTM модель загружена")
                        # Здесь можно добавить логику старой модели
                except Exception as e2:
                    logger.error(f"❌ Старая модель тоже не сработала: {e2}")
                    # Пробуем гибридный прогноз
                    try:
                        from hybrid_predictor import HybridEarthquakePredictor
                        hybrid = HybridEarthquakePredictor(model_dir="data/models")
                        X, y_prob, y_mag = hybrid.prepare_tabular_features(
                            all_events, data, lst_cache, event_iono, event_space, event_weather
                        )
                        if len(X) >= 30:
                            hybrid.train(X, y_prob, y_mag)
                            forecast = hybrid.predict(
                                all_events, data, lst_cache, event_iono, event_space, event_weather
                            )
                            if forecast['probability'] > 0.2 and TG_AVAILABLE and chat_id:
                                msg = (
                                    f"🔮 <b>ГИБРИДНЫЙ ПРОГНОЗ (резерв)</b>\n"
                                    f"Вероятность M≥6.0: {forecast['probability']*100:.1f}%\n"
                                    f"Прогнозируемая магнитуда: {forecast['magnitude']:.1f}\n"
                                    f"Ожидаемое время: через {forecast['days']} дней"
                                )
                                url = f"https://api.telegram.org/bot{token}/sendMessage"
                                payload = {'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}
                                requests.post(url, data=payload, timeout=30)
                                logger.info("✅ Гибридный прогноз отправлен в Telegram")
                    except Exception as e3:
                        logger.error(f"❌ Гибридный прогноз тоже не сработал: {e3}")
    else:
        if not LSTM_AVAILABLE:
            logger.info("ℹ️ LSTM-прогнозирование пропущено (модуль недоступен)")
        else:
            logger.info("ℹ️ LSTM-прогнозирование пропущено (нет данных)")
    """
    # ═══════════════════════════════════════════════════════════════
    # 1f. АНАЛИЗ ПО СЕЙСМИЧЕСКИМ ЗОНАМ (с историей и тектоникой)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🌍 ЭТАП 1f: Анализ по сейсмическим зонам (с историей)")
    logger.info("=" * 70)

    zone_analyzer = FaultZoneAnalyzer()
    zone_alerts = {}        # для отправки алертов
    zone_risks = {}         # <-- ИНИЦИАЛИЗИРУЕМ СЛОВАРЬ ДЛЯ ВИЗУАЛИЗАЦИИ

    if zone_analyzer.gdf is not None:
        # Собираем аномальные события (например, из event_iono)
        anomaly_events = []
        for idx, row in strong_events.iterrows():
            event_id = row['id']
            if event_id in event_iono and event_iono[event_id].get('is_anomaly', False):
                anomaly_events.append({
                    'id': event_id,
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'mag': row['magnitude'],
                    'time': row['time'],
                    'z_score': event_iono[event_id].get('z_score_max', 0)
                })

        if anomaly_events:
            # Группируем по зонам
            zone_events = {}
            for evt in anomaly_events:
                zone_id, dist = zone_analyzer.get_nearest_zone(evt['lat'], evt['lon'], max_distance_km=150)
                if zone_id is None:
                    zone_id = "unknown"
                if zone_id not in zone_events:
                    zone_events[zone_id] = {'events': [], 'lat': evt['lat'], 'lon': evt['lon']}
                zone_events[zone_id]['events'].append(evt)

             # Рассчитываем риск для каждой зоны
            zone_reports = []
            for zone_id, zone_data in zone_events.items():   # <-- ПЕРЕИМЕНОВАНО zone_data
                if zone_id == "unknown":
                    continue
                lat = zone_data['lat']
                lon = zone_data['lon']
                current_anomalies = zone_data['events']
                risk = zone_analyzer.calculate_zone_risk(zone_id, lat, lon, current_anomalies)

                # Получаем название зоны из zone_analyzer (или используем zone_id)
                zone_name = zone_analyzer.get_zone_name(zone_id)
                if not zone_name:
                    zone_name = f"Зона {zone_id}"  # запасной вариант

                zone_reports.append({
                    'zone_id': zone_id,
                    'zone_name': zone_name,
                    'lat': lat,
                    'lon': lon,
                    'risk': risk,
                    'events': current_anomalies
                })
                if risk['level'] == 'critical':
                    zone_alerts[zone_id] = risk

            # Формируем отчёт
            report_lines = ["🌍 **Аномалии по сейсмическим зонам (с историей)**\n"]
            for z in zone_reports:
                r = z['risk']
                report_lines.append(
                    f"{r['emoji']} **{z['zone_name']}**\n"
                    f"   • Аномалий: {r['anomaly_count']}\n"
                    f"   • Макс магнитуда: {r['max_mag']:.1f}\n"
                    f"   • Макс Z-score: {r['max_z']:.2f}σ\n"
                    f"   • История: {r['hist_count']} событий за 10 лет, макс M={r['hist_max_mag']:.1f}\n"
                    f"   • Риск-скор: {r['risk_score']}/100\n"
                    f"   • Прогноз: {r['forecast']}\n"
                )
            zone_report = "\n".join(report_lines)

            # Отправляем в Telegram
            if TG_AVAILABLE and chat_id:
                try:
                    # Отправляем как текстовое сообщение
                    token = os.environ.get('TELEGRAM_BOT_TOKEN')
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {'chat_id': chat_id, 'text': zone_report, 'parse_mode': 'HTML'}
                    requests.post(url, data=payload, timeout=30)
                    logger.info("✅ Отчёт по зонам отправлен")
                except Exception as e:
                    logger.error(f"Ошибка отправки отчёта по зонам: {e}")

            # Отправляем алерты для критических зон
            for zone_id, risk in zone_alerts.items():
                alert_msg = (
                    f"🚨 **КРИТИЧЕСКИЙ ПРОГНОЗ ДЛЯ ЗОНЫ {zone_id}**\n"
                    f"Уровень риска: {risk['emoji']} {risk['level'].upper()}\n"
                    f"Прогноз: {risk['forecast']}\n"
                    f"Аномалий: {risk['anomaly_count']}, макс M: {risk['max_mag']:.1f}, Z: {risk['max_z']:.2f}σ"
                )
                try:
                    payload = {'chat_id': chat_id, 'text': alert_msg, 'parse_mode': 'HTML'}
                    requests.post(url, data=payload, timeout=30)
                    logger.info(f"🚨 Алерт для зоны {zone_id} отправлен")
                except Exception as e:
                    logger.error(f"Ошибка отправки алерта для {zone_id}: {e}")
        else:
            logger.info("ℹ️ Аномальных событий не найдено, анализ зон пропущен.")
    else:
        logger.info("ℹ️ Файл разломов не загружен, анализ зон отключён.")

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 2: LAIC-АНАЛИЗ
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🔬 ЭТАП 2: LAIC-анализ")
    logger.info("=" * 70)

    try:
        from laic_analyzer import LAICAnalyzer
        analyzer = LAICAnalyzer(settings)
        results = analyzer.analyze_all(data)

        # ============================================================
        # ОБОГАЩАЕМ РЕЗУЛЬТАТЫ НАЗВАНИЯМИ ЗОН ИЗ СЛОВАРЯ
        # ============================================================
        enriched_count = 0
        for result in results:
            if isinstance(result, dict):
                # Пробуем взять координаты из разных полей
                lat = result.get('latitude') or result.get('lat') or 0
                lon = result.get('longitude') or result.get('lon') or 0

                if lat and lon:
                    zone_name, zone_coords = get_zone_info(lat, lon)
                    if zone_name:
                        result['zone_name'] = zone_name
                        result['zone_coords'] = zone_coords
                        enriched_count += 1
                        logger.debug(f"   🗺️ {zone_name} ({lat:.2f}, {lon:.2f})")
                    else:
                        result['zone_name'] = f"Точка {lat:.2f}, {lon:.2f}"
                        result['zone_coords'] = [lat, lon]

        logger.info(f"✅ LAIC-анализ: {len(results)} регионов, обогащено {enriched_count} зон")

    except Exception as e:
        logger.error(f"❌ Ошибка LAIC-анализа: {e}")
        results = []

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 3: ОТПРАВКА В TELEGRAM
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📤 ЭТАП 3: Отправка отчётов")
    logger.info("=" * 70)

    alerts_sent = 0
    if TG_AVAILABLE and results:
        try:
            for r in results:
                if r.get('alert', False):
                    alert_sync(r)
                    alerts_sent += 1
            send_sync(results)
            logger.info(f"✅ Отправлено {alerts_sent} алертов")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки: {e}")


    # ═══════════════════════════════════════════════════════════════
    # 4. ВИЗУАЛИЗАЦИЯ (PNG + HTML)
    # ═══════════════════════════════════════════════════════════════
    if VISUALIZER_AVAILABLE and TG_AVAILABLE and chat_id:
        logger.info("\n" + "=" * 70)
        logger.info("🎨 ЭТАП 4: Визуализация")
        logger.info("=" * 70)

        period_days = settings.get('analysis', {}).get('history_days', 30)


        space_data = None
        iono_data = None

        for region_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            # 1. Временной ряд
            try:
                img = plot_magnitude_series(df, region_name)
                if img:
                    caption = f"📊 {region_name} – временной ряд магнитуд"
                    send_photo(chat_id, img, caption)
                    logger.info(f"✅ График {region_name}")
            except Exception as e:
                logger.error(f"❌ PNG {region_name}: {e}")

            # 2. График аномалий (если есть anomaly_df)
            if anomaly_df is not None and not anomaly_df.empty:
                region_anomalies = anomaly_df[anomaly_df['region'] == region_name]
                if not region_anomalies.empty:
                    try:
                        img_anom = create_anomaly_plot(region_anomalies, region_name)
                        if img_anom:
                            send_photo(chat_id, img_anom, f"🧠 Аномалии — {region_name}")
                    except Exception as e:
                        logger.error(f"❌ Аномалии {region_name}: {e}")

            # 3. График солнечной активности
            if event_space:
                space_data = None
                for eid in df['id'].tolist():
                    if eid in event_space:
                        space_data = event_space[eid]
                        break
                if space_data:
                    try:
                        img_solar = create_solar_activity_plot(space_data, region_name, df['magnitude'].max())
                        if img_solar:
                            send_photo(chat_id, img_solar, f"☀️ Солнечная активность — {region_name}")
                    except Exception as e:
                        logger.error(f"❌ Солнечная {region_name}: {e}")

            # 4. График ионосферы
            if event_iono:
                iono_data = None
                for eid in df['id'].tolist():
                    if eid in event_iono:
                        iono_data = event_iono[eid]
                        break
                if iono_data:
                    try:
                        img_iono = create_iono_anomaly_plot(iono_data, region_name)
                        if img_iono:
                            send_photo(chat_id, img_iono, f"🛰️ Ионосфера — {region_name}")
                    except Exception as e:
                        logger.error(f"❌ Ионосфера {region_name}: {e}")

            # 5. HTML-отчёт
            try:
                region_anomalies = anomaly_df[anomaly_df['region'] == region_name] if (anomaly_df is not None and not anomaly_df.empty) else None
                html_path = create_interactive_report(
                    region_name, df,
                    lst_data=lst_cache.get(region_name),
                    period_days=period_days,
                    iono_data=iono_data,
                    space_data=space_data,
                    anomaly_data=region_anomalies,
                    zone_risks=zone_risks   # <-- передаём словарь из этапа 1f
                )
                if html_path:
                    caption = f"📄 {region_name} – интерактивный отчёт"
                    send_document(chat_id, html_path, caption)
                    logger.info(f"✅ HTML {region_name}")
            except Exception as e:
                logger.error(f"❌ HTML {region_name}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # ИТОГ
    # ═══════════════════════════════════════════════════════════════
    critical = sum(1 for r in results if isinstance(r, dict) and r.get('risk_level') == 'critical')
    high = sum(1 for r in results if isinstance(r, dict) and r.get('risk_level') == 'high')

    logger.info("\n" + "=" * 70)
    logger.info("✅ АНАЛИЗ ЗАВЕРШЁН")
    logger.info(f"   Регионов: {len(results)}")
    logger.info(f"   🔴 Критических: {critical}")
    logger.info(f"   🟠 Высоких: {high}")
    logger.info(f"   🚨 Алертов: {alerts_sent}")
    if data_lag_days > 0:
        logger.info(f"   ⚠️ Задержка данных: {data_lag_days} дней")
    logger.info("=" * 70)

    sys.exit(0)


if __name__ == '__main__':
    main()
