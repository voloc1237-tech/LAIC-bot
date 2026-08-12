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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from aftershock_filter import filter_aftershocks
from fault_zones import FaultZoneAnalyzer

# ═══════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
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
# ИМПОРТЫ МОДУЛЕЙ
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

# Космическая погода — ИСПРАВЛЕНО: singleton
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

# LSTM прогнозирование — ИСПРАВЛЕНО: новый класс
try:
    from predictor import EarthquakeLSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("⚠️ LSTM predictor не доступен")

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
        space = get_collector() if SPACE_AVAILABLE else None  # ← singleton!

        # Проверка качества космических данных
        if space:
            try:
                test_start = effective_time - timedelta(days=10)
                quality = get_data_quality(test_start, effective_time)
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
                except Exception as e:
                    logger.debug(f"   Погода: {e}")

            # Ионосфера
            if iono:
                try:
                    iono_result = iono.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    
                    clim_years = iono_result.get('n_historical_years', 0)
                    if clim_years > 0:
                        logger.info(f"   📊 Климатология: {clim_years} лет")
                    
                    if iono_result.get('is_anomaly', False):
                        level = iono_result.get('level', 'unknown')
                        z_max = iono_result.get('z_score_max', 0)
                        z_min = iono_result.get('z_score_min', 0)
                        
                        if level == 'critical':
                            logger.critical(f"   🔴🔴🔴 КРИТИЧЕСКАЯ АНОМАЛИЯ: max={z_max:.2f}σ, min={z_min:.2f}σ")
                        elif level == 'high':
                            logger.warning(f"   🟠🟠 ВЫСОКАЯ АНОМАЛИЯ: max={z_max:.2f}σ, min={z_min:.2f}σ")
                        else:
                            logger.info(f"   🟡 Умеренная аномалия")
                    else:
                        logger.info(f"   ✅ Ионосфера в норме")
                    
                    event_iono[event_id] = iono_result
                    
                except Exception as e:
                    logger.debug(f"   Ионосфера: {e}")

            # Космическая погода — ИСПРАВЛЕНО: используем singleton
            if space:
                try:
                    start = event_time - timedelta(days=7)
                    end = event_time + timedelta(days=3)
                    space_data = space.fetch_all_for_period(start, end)
                    if any(not df.empty for df in space_data.values()):
                        event_space[event_id] = space_data
                except Exception as e:
                    logger.debug(f"   Космос: {e}")

        logger.info(f"✅ Собрано: погода {len(event_weather)}, ионосфера {len(event_iono)}, космос {len(event_space)}")

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
    # ЭТАП 1e: ГИБРИДНОЕ ПРОГНОЗИРОВАНИЕ
    # ═══════════════════════════════════════════════════════════════
    try:
        from hybrid_predictor import HybridEarthquakePredictor
        HYBRID_AVAILABLE = True
    except ImportError:
        HYBRID_AVAILABLE = False
        logger.warning("⚠️ Гибридный прогноз недоступен")

    if HYBRID_AVAILABLE and not all_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info("🔮 ЭТАП 1e: Гибридное прогнозирование (LSTM + CatBoost + XGBoost)")
        logger.info("=" * 70)
    
        hybrid = HybridEarthquakePredictor(model_dir="data/models")
    
        # Подготовка данных для обучения
        X, y_prob, y_mag = hybrid.prepare_tabular_features(
            all_events, data, lst_cache, event_iono, event_space, event_weather
        )
    
        if len(X) >= 30:
            # Обучение
            hybrid.train(X, y_prob, y_mag)
        
            # Прогноз на текущий момент
            forecast = hybrid.predict(
                all_events, data, lst_cache, event_iono, event_space, event_weather
            )
        
            if forecast['probability'] > 0.2:
                logger.info(f"🔮 Прогноз: вероятность M≥6.0 = {forecast['probability']*100:.1f}%")
                logger.info(f"   Магнитуда: {forecast['magnitude']:.1f}, метод: {forecast['method']}")
            
                # Отправка в Telegram
                if TG_AVAILABLE and chat_id:
                    msg = (
                        f"🔮 <b>ГИБРИДНЫЙ ПРОГНОЗ</b>\n"
                        f"Вероятность M≥6.0: {forecast['probability']*100:.1f}%\n"
                        f"Прогнозируемая магнитуда: {forecast['magnitude']:.1f}\n"
                        f"Ожидаемое время: через {forecast['days']} дней\n"
                        f"Модель: {forecast['method']}"
                    )
                    # Отправка
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}
                    requests.post(url, data=payload, timeout=30)
            else:
                logger.info("ℹ️ Риск низкий, прогноз не отправлен")
        else:
            logger.info(f"ℹ️ Недостаточно данных для гибридного обучения ({len(X)} точек)")
    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1e: LSTM ПРОГНОЗИРОВАНИЕ (ИСПРАВЛЕННОЕ)
    # ═══════════════════════════════════════════════════════════════
    #if LSTM_AVAILABLE and not all_events.empty:
        #logger.info("\n" + "=" * 70)
        #logger.info("🔮 ЭТАП 1e: LSTM прогнозирование")
        #logger.info("=" * 70)
        
        #lstm_predictor = EarthquakeLSTMPredictor(model_dir="data/models/lstm")
        
        # Подготовка ежедневных данных
        #daily_df = lstm_predictor.prepare_data(all_events)
        
        #if len(daily_df) >= lstm_predictor.sequence_length + 50:
            
            # Обучение если нужно
            #if not lstm_predictor.model:
                #success = lstm_predictor.train(daily_df, epochs=50)
                #if not success:
                    #logger.warning("⚠️ LSTM обучение не удалось")
            
            # Прогноз
            #if lstm_predictor.model:
                #forecast = lstm_predictor.predict(daily_df, current_time=effective_time)
                
                # Форматирование
                #windows_str = "\n".join(
                    #f"  • {k}: {v*100:.1f}%" 
                    #for k, v in forecast['windows'].items()
                #)
                
                #if forecast.get('primary_forecast'):
                    #pf = forecast['primary_forecast']
                    #main_msg = (
                        #f"🔮 <b>LSTM ПРОГНОЗ</b>\n"
                        #f"{'─' * 30}\n"
                        #f"📅 Прогноз на: {forecast['forecast_time'][:10]}\n"
                        #f"{'─' * 30}\n"
                        #f"<b>Окно риска:</b> {pf['window']}\n"
                        #f"<b>Вероятность M≥6.0:</b> {pf['probability']*100:.1f}%\n"
                        #f"<b>Через:</b> {pf['days_to_event_min']}-{pf['days_to_event_max']} дней\n"
                        #f"{'─' * 30}\n"
                        #f"<b>Все окна:</b>\n{windows_str}\n"
                    #)
                    
                    #if pf.get('is_significant'):
                        #main_msg += "\n🔴 <b>ВЫСОКИЙ РИСК!</b>"
                    #else:
                        #main_msg += "\n🟠 Повышенное внимание"
                #else:
                    #main_msg = (
                        #f"🔮 <b>LSTM ПРОГНОЗ</b>\n"
                        #f"{'─' * 30}\n"
                        #f"✅ Значимых паттернов не обнаружено\n"
                        #f"{'─' * 30}\n"
                        #f"{windows_str}"
                    #)
                
                # Предупреждение о свежести
                #if forecast.get('data_staleness_days', 0) > 2:
                    #main_msg += f"\n\n⚠️ Данные отстают на {forecast['data_staleness_days']} дней"
                
                #logger.info(f"🔮 LSTM:\n{main_msg.replace('<b>', '').replace('</b>', '')}")
                
                # Отправка
                #if TG_AVAILABLE and chat_id:
                    #try:
                        #url = f"https://api.telegram.org/bot{token}/sendMessage"
                        #payload = {'chat_id': chat_id, 'text': main_msg, 'parse_mode': 'HTML'}
                        #response = requests.post(url, data=payload, timeout=30)
                        #if response.status_code == 200:
                            #logger.info("✅ LSTM прогноз отправлен")
                        #else:
                            #logger.error(f"❌ Ошибка: {response.text}")
                    #except Exception as e:
                        #logger.error(f"❌ Ошибка отправки: {e}")
            #else:
                #logger.info("ℹ️ LSTM модель не загружена")
        #else:
            #logger.info(f"ℹ️ Недостаточно данных для LSTM ({len(daily_df)})")

    # ═══════════════════════════════════════════════════════════════
    # ЭТАП 1f: LAIC ПРОГНОЗИРОВАНИЕ (ГРАДИЕНТНЫЙ БУСТИНГ)
    # ═══════════════════════════════════════════════════════════════
    #if LAIC_PREDICTOR_AVAILABLE and len(all_events) >= 30:
        #logger.info("\n" + "=" * 70)
        #logger.info("🔮 ЭТАП 1f: LAIC прогнозирование (Gradient Boosting)")
        #logger.info("=" * 70)
        
        #from earthquake_predictor import create_risk_report
        
        #predictor = LAICPredictor(model_dir="data/models")
        
        #iono_cache = {eid: d for eid, d in event_iono.items()}
        #space_cache = {eid: d for eid, d in event_space.items()}
        #weather_cache = {eid: d for eid, d in event_weather.items()}
        
        # Обучение/загрузка
        #if not predictor.is_trained:
            #logger.info("🧠 Обучение LAIC моделей...")
            
            #X, y = predictor.prepare_training_data(
                #events_df=all_events,
                #region_data_dict=data,
                #lst_cache=lst_cache,
                #iono_cache=iono_cache,
                #space_cache=space_cache,
                #weather_cache=weather_cache
            #)
            
            #success = predictor.train(X, y)
            #if not success:
                #logger.warning("⚠️ LAIC обучение не удалось")
        
        #if predictor.is_trained:
            # Прогноз по сетке — ИСПРАВЛЕНО: используем predict_grid с horizon_days
            #anomalous_regions = []
            #for region_name, df in data.items():
                #if not df.empty:
                    #last = df.iloc[0]
                    #anomalous_regions.append({
                        #'name': region_name,
                        #'lat': last['latitude'],
                        #'lon': last['longitude']
                    #})
            
            #all_predictions = []
            
            #for region in anomalous_regions:
                #region_bounds = {
                    #'lat_min': max(-80, region['lat'] - 8),
                    #'lat_max': min(80, region['lat'] + 8),
                    #'lon_min': region['lon'] - 10,
                    #'lon_max': region['lon'] + 10
                #}
                
                # ИСПРАВЛЕНО: правильный вызов predict_grid
                #grid_pred = predictor.predict_grid(
                    #region_bounds=region_bounds,
                    #current_time=effective_time,
                    #region_data_dict=data,
                    #lst_cache=lst_cache,
                    #iono_cache=iono_cache,
                    #space_cache=space_cache,
                    #resolution=2.0,
                    #horizon_days=30
                #)
                
                #if not grid_pred.empty:
                    #all_predictions.append(grid_pred)
                    
                    # Логируем топ-3
                    #for i, (_, row) in enumerate(grid_pred.head(3).iterrows(), 1):
                        #emoji = "🔴" if row.get('is_alert') else "🟠"
                        #days = row.get('days_to_event', 'N/A')
                        #sign = "+" if isinstance(days, (int, float)) and days > 0 else ""
                        #logger.info(
                            #f"   {emoji} #{i}: "
                            #f"{row['lat']}°, {row['lon']}° | "
                            #f"M{row.get('predicted_magnitude', 0):.1f} | "
                            #f"{row.get('probability_m6', 0)*100:.0f}% | "
                            #f"{sign}{days:.0f}д"
                        #)
            
            # Объединение и отправка
            #if all_predictions:
                #combined = pd.concat(all_predictions, ignore_index=True)
                #combined = combined.sort_values('probability_m6', ascending=False)
                #combined = combined.drop_duplicates(subset=['lat', 'lon'], keep='first')
                
                #report_text = create_risk_report(combined, top_n=5)
                
                #if TG_AVAILABLE:
                    #try:
                        #alert_sync(report_text)
                        #logger.info("✅ LAIC прогноз отправлен")
                    #except Exception as e:
                        #logger.warning(f"⚠️ Ошибка отправки: {e}")
                
                # Сохранение
                #combined.to_csv(f"data/predictions_{effective_time.strftime('%Y%m%d')}.csv", index=False)
                #logger.info(f"💾 Прогноз сохранён: {len(combined)} зон")
                
                # Критические алерты
                #high_risk = combined[combined['is_alert'] == True] if 'is_alert' in combined.columns else pd.DataFrame()
                ##high_risk = combined[combined.get('is_alert', pd.Series([False]*len(combined)))] не включать!
                #if not high_risk.empty:
                    #logger.critical(f"🚨 {len(high_risk)} зон высокого риска!")
            #else:
                #logger.info("ℹ️ Значимых прогнозов нет")
        #else:
            #logger.warning("⚠️ LAIC модели не обучены")
    #else:
        #logger.info("ℹ️ LAIC прогнозирование пропущено")


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
                zone_reports.append({
                    'zone_id': zone_id,
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
                    f"{r['emoji']} **{z['zone_id']}**\n"
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
        logger.info(f"✅ LAIC-анализ: {len(results)} регионов")
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
