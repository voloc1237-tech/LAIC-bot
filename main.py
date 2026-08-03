#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — ГЛАВНЫЙ ФАЙЛ LAIC-бота
Собирает → Анализирует → Отправляет
"""

import os
import sys
import logging
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from aftershock_filter import filter_aftershocks

# ═══════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('LAIC')

# ═══════════════════════════════════════════════════════════════
# СОЗДАНИЕ НЕОБХОДИМЫХ ПАПОК
# ═══════════════════════════════════════════════════════════════
def ensure_directories():
    """Создаёт необходимые папки для работы бота."""
    directories = [
        "data",
        "data/cache",
        "data/ionex_cache",
        "data/models",
        "config"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"📁 Папка: {directory}")

# Создаём папки
ensure_directories()

# ═══════════════════════════════════════════════════════════════
# ИМПОРТЫ МОДУЛЕЙ (с проверкой наличия)
# ═══════════════════════════════════════════════════════════════

# GEE
try:
    from ee_collector import GEEInitializer, get_lst_data
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("⚠️ Earth Engine модуль не найден")

# Визуализация
try:
    from visualizer import plot_magnitude_series, create_interactive_report
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    logger.warning("⚠️ Модуль визуализации не найден (visualizer.py)")

# Телеграм (основной и дополнительные функции)
try:
    from telegram_sender import send_sync, alert_sync, send_photo, send_document
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    logger.warning("⚠️ Модуль Telegram не найден")

# Новые модули (погода, ионосфера, космическая погода)
try:
    from weather_collector import WeatherCollector
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False
    logger.warning("⚠️ WeatherCollector не доступен")

try:
    from ionosphere_collector import IonosphereCollector
    IONO_AVAILABLE = True
except ImportError:
    IONO_AVAILABLE = False
    logger.warning("⚠️ IonosphereCollector не доступен")

try:
    from space_weather import SpaceWeatherCollector
    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    logger.warning("⚠️ SpaceWeatherCollector не доступен")


def main():
    logger.info("=" * 70)
    logger.info("🚀 LAIC EARTHQUAKE MONITOR")
    logger.info(f"⏰ {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 70)

    # Проверка токенов Telegram
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.error("❌ TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы!")
        sys.exit(1)
    logger.info("✅ Токены Telegram найдены")

    # ═══════════════════════════════════════════════════════════════
    # ЗАГРУЗКА НАСТРОЕК
    # ═══════════════════════════════════════════════════════════════
    import yaml
    try:
        with open('config/settings.yaml', 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("❌ config/settings.yaml не найден!")
        sys.exit(1)
    logger.info(f"✅ Настройки загружены: {len(settings.get('regions', {}))} регионов")

    # ═══════════════════════════════════════════════════════════════
    # ПОДКЛЮЧЕНИЕ GOOGLE EARTH ENGINE
    # ═══════════════════════════════════════════════════════════════
    ee_initialized = False
    if GEE_AVAILABLE:
        try:
            ee_initialized = GEEInitializer.init()
            if ee_initialized:
                logger.info("✅ Earth Engine подключен!")
            else:
                logger.warning("⚠️ Earth Engine не инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Earth Engine ошибка: {e}")
    else:
        logger.info("ℹ️ Earth Engine не настроен")

    # ═══════════════════════════════════════════════════════════════
    # 1. СБОР ДАННЫХ USGS (глобально или по регионам)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📥 ЭТАП 1: Сбор данных USGS")
    logger.info("=" * 70)

    from data_collector import collect_all_regions

    # Определяем режим: глобальный или региональный
    global_mode = settings.get('analysis', {}).get('global_mode', True)
    data = collect_all_regions(settings, global_mode=global_mode)

    total = sum(len(df) for df in data.values())
    logger.info(f"\n📊 Всего событий: {total}")

    # ═══════════════════════════════════════════════════════════════
    # 1a. ФИЛЬТРАЦИЯ АФТЕРШОКОВ
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🔍 ЭТАП 1a: Фильтрация афтершоков")
    logger.info("=" * 70)
    
    data_filtered = {}
    aftershock_counts = {}
    
    for region_name, df in data.items():
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"⚠️ {region_name}: не DataFrame, пропускаем")
            data_filtered[region_name] = pd.DataFrame()
            aftershock_counts[region_name] = 0
            continue
        
        if df.empty:
            data_filtered[region_name] = df
            aftershock_counts[region_name] = 0
        else:
            df_filtered, df_aftershocks = filter_aftershocks(
                df,
                time_window_days=7,
                distance_km=50,
                mag_threshold=0.5
            )
            data_filtered[region_name] = df_filtered
            aftershock_counts[region_name] = len(df_aftershocks)
            logger.info(f"   📌 {region_name}: афтершоков {len(df_aftershocks)} из {len(df)} событий")
    
    data = data_filtered
    
    total_filtered = sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame))
    logger.info(f"\n📊 После фильтрации: {total_filtered} событий (удалено {sum(aftershock_counts.values())} афтершоков)")


    # ============================================================
    # 🔥 ОТЛАДКА: ПРИНУДИТЕЛЬНАЯ ДАТА ДЛЯ ПРОВЕРКИ РЕАЛЬНОГО СКАЧИВАНИЯ
    # Удали этот блок после проверки!
    # ============================================================
    TEST_MODE = True 
    
    if TEST_MODE and not strong_events.empty:
        # Берем первое событие и меняем дату на 15.01.2023 (UTC)
        test_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        logger.warning("🛠️ ОТЛАДОЧНЫЙ РЕЖИМ: Принудительно установлена дата события: 2023-01-15")
        logger.warning("🛠️ Цель: скачать реальные codg-файлы за 2023 и фон за 2022")
        
        # Обновляем дату в DataFrame
        strong_events.iloc[0]['time'] = test_date
        
        # Если у тебя события хранятся в словаре или списке отдельно, обнови и там
        # data['events'][0]['time'] = test_date 
    

    # ═══════════════════════════════════════════════════════════════
    # 1b. СБОР СПУТНИКОВЫХ ДАННЫХ (LST) — если доступен GEE
    # ═══════════════════════════════════════════════════════════════
    lst_cache = {}
    if ee_initialized:
        logger.info("\n" + "=" * 70)
        logger.info("🛰️ ЭТАП 1b: Сбор спутниковых данных (GEE)")
        logger.info("=" * 70)
    
        for region_name, region_data in data.items():
            if not region_data.empty:
                last_event = region_data.iloc[0]
                lat = last_event['latitude']
                lon = last_event['longitude']
                try:
                    end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
                    lst_data = get_lst_data(lat, lon, start_date, end_date)
                    
                    if lst_data.get('lst_celsius') is not None:
                        lst_cache[region_name] = lst_data
                        logger.info(f"🌡️ {region_name}: LST = {lst_data['lst_celsius']}°C (изобр: {lst_data['count']})")
                    else:
                        error_msg = lst_data.get('error', 'Неизвестная ошибка')
                        logger.warning(f"⚠️ Нет LST для {region_name}: {error_msg}")
                        lst_cache[region_name] = None
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка LST для {region_name}: {e}")
                    lst_cache[region_name] = None
    else:
        logger.info("ℹ️ Пропускаем спутниковые данные (GEE недоступен)")

    # ═══════════════════════════════════════════════════════════════
    # 1c. СБОР ДОПОЛНИТЕЛЬНЫХ ДАННЫХ (погода, ионосфера, космос)
    # ═══════════════════════════════════════════════════════════════
    min_mag_for_detail = settings.get('analysis', {}).get('detail_min_magnitude', 5.0)
    event_weather = {}
    event_iono = {}
    event_space = {}

    all_events = pd.concat(
        [df for df in data.values() if not df.empty],
        ignore_index=True
    ) if data else pd.DataFrame()

    strong_events = all_events[all_events['magnitude'] >= min_mag_for_detail] if not all_events.empty else pd.DataFrame()

    if not strong_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info(f"📌 ЭТАП 1c: Сбор дополнительных данных для {len(strong_events)} сильных событий (M≥{min_mag_for_detail})")
        logger.info("=" * 70)

        weather = WeatherCollector() if WEATHER_AVAILABLE else None
        iono = IonosphereCollector() if IONO_AVAILABLE else None
        space = SpaceWeatherCollector() if SPACE_AVAILABLE else None

        for idx, row in strong_events.iterrows():
            event_id = row['id']
            lat = row['latitude']
            lon = row['longitude']
            event_time = row['time']
            mag = row['magnitude']
            
            if row.get('is_aftershock', False):
                logger.info(f"⏭️ Афтершок {event_id}, пропускаем сбор данных")
                continue
            
            logger.info(f"🔍 Событие {event_id} M{mag:.1f} в {event_time}")

            # Погода
            if weather:
                try:
                    wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    if not wdf.empty:
                        event_weather[event_id] = wdf
                        logger.info(f"   🌡️ Погода: {len(wdf)} дней")
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка погоды: {e}")

            # Ионосфера (TEC, ROTI) + АНАЛИЗ АНОМАЛИЙ
            if iono:
                try:
                    iono_result = iono.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    
                    if iono_result.get('is_anomaly', False):
                        logger.critical(f"   🔴🔴🔴 ИОНОСФЕРНАЯ АНОМАЛИЯ: {iono_result.get('details', '')}")
                    else:
                        logger.info(f"   ✅ Ионосфера: {iono_result.get('details', 'В норме')}")
                    
                    # Сохраняем результат даже если нет данных
                    event_iono[event_id] = iono_result
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка ионосферы: {e}")

            # Космическая погода (Kp, Dst, F10.7)
            if space:
                try:
                    start = event_time - timedelta(days=7)
                    end = event_time + timedelta(days=3)
                    space_data = space.fetch_all_for_period(start, end)
                    if any(not df.empty for df in space_data.values()):
                        event_space[event_id] = space_data
                        logger.info(f"   ☀️ Космическая погода: загружена")
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка космической погоды: {e}")

        logger.info(f"✅ Собрано: погода для {len(event_weather)} событий, "
                    f"ионосфера для {len(event_iono)}, космос для {len(event_space)}")
    else:
        logger.info("ℹ️ Нет сильных событий для детального сбора данных.")

    # ═══════════════════════════════════════════════════════════════
    # 1d. ИИ-АНАЛИЗ АНОМАЛИЙ
    # ═══════════════════════════════════════════════════════════════
    try:
        from anomaly_detector import AnomalyDetector
        from visualizer import create_anomaly_plot
        ANOMALY_AVAILABLE = True
    except ImportError:
        ANOMALY_AVAILABLE = False
        logger.warning("⚠️ Модуль аномалий не найден")
    
    if ANOMALY_AVAILABLE and not all_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info("🧠 ЭТАП 1d: ИИ-анализ аномалий")
        logger.info("=" * 70)
        
        events_for_ai = []
        for region_name, df in data.items():
            if df.empty:
                continue
            for idx, row in df.iterrows():
                lst_data = lst_cache.get(region_name)
                lst_celsius = 0
                if lst_data and isinstance(lst_data, dict):
                    lst_celsius = lst_data.get('lst_celsius', 0)
                
                # Добавляем данные ионосферы, если есть
                event_id = row.get('id', f'event_{idx}')
                iono_info = event_iono.get(event_id, {})
                
                event_data = {
                    'id': event_id,
                    'region': region_name,
                    'magnitude': row.get('magnitude', 0),
                    'depth_km': row.get('depth_km', 0),
                    'time': row.get('time', None),
                    'lst_celsius': lst_celsius,
                    'weather': {
                        'temp_mean': row.get('temp_mean', 0),
                        'humidity_mean': row.get('humidity_mean', 0)
                    },
                    'kp': {
                        'mean': row.get('kp_mean', 0),
                        'max': row.get('kp_max', 0),
                        'std': row.get('kp_std', 0)
                    },
                    'iono_anomaly': iono_info.get('is_anomaly', False),
                    'z_score_max': iono_info.get('z_score_max', 0.0)
                }
                events_for_ai.append(event_data)
        
        if len(events_for_ai) >= 10:
            detector = AnomalyDetector(contamination=0.1)
            anomaly_df = detector.analyze_events(events_for_ai)
            
            if not anomaly_df.empty:
                anomaly_counts = anomaly_df.groupby('region')['is_anomaly'].sum().to_dict()
                for region, count in anomaly_counts.items():
                    if count > 0:
                        logger.info(f"   🔴 {region}: обнаружено {count} аномалий")
                
                if TG_AVAILABLE and chat_id:
                    for region_name in anomaly_df['region'].unique():
                        region_anomalies = anomaly_df[anomaly_df['region'] == region_name]
                        if len(region_anomalies) > 1:
                            img_bytes = create_anomaly_plot(region_anomalies, region_name)
                            if img_bytes:
                                send_photo(chat_id, img_bytes, 
                                          caption=f"🧠 ИИ-анализ аномалий — {region_name}")
                                logger.info(f"✅ График аномалий для {region_name} отправлен")
            else:
                logger.info("ℹ️ Недостаточно данных для ИИ-анализа")
        else:
            logger.info(f"ℹ️ ИИ-анализ пропущен (нужно ≥10 событий, собрано {len(events_for_ai)})")
    else:
        logger.info("ℹ️ ИИ-анализ пропущен (модуль недоступен или нет данных)")

    # ═══════════════════════════════════════════════════════════════
    # 1e. ПРОГНОЗИРОВАНИЕ (LSTM)
    # ═══════════════════════════════════════════════════════════════
    try:
        from predictor import EarthquakePredictor
        PREDICTOR_AVAILABLE = True
    except ImportError:
        PREDICTOR_AVAILABLE = False
        logger.warning("⚠️ Модуль прогнозирования не найден")
    
    if PREDICTOR_AVAILABLE and not all_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info("🔮 ЭТАП 1e: Прогнозирование (LSTM)")
        logger.info("=" * 70)
        
        predictor = EarthquakePredictor()
        
        training_data = []
        for _, row in all_events.iterrows():
            mag = row.get('magnitude', 0)
            energy = 10**(1.5 * mag + 4.8) / 4.184e12
            kp_mean = row.get('kp_mean', 0)
            temp_mean = row.get('temp_mean', 0)
            
            training_data.append({
                'magnitude': mag,
                'energy': energy,
                'kp_mean': kp_mean,
                'temp_mean': temp_mean
            })
        
        df_train = pd.DataFrame(training_data)
        
        if len(df_train) >= 20:
            predictor.train(df_train, epochs=30)
            
            prob, pred_mag = predictor.predict(df_train)
            if prob is not None:
                logger.info(f"🔮 Прогноз: вероятность M≥6.0 = {prob*100:.1f}%")
                logger.info(f"   Прогнозируемая магнитуда: {pred_mag:.2f}")
                
                if prob > 0.5:
                    status = "⚠️ ПОВЫШЕННОЕ ВНИМАНИЕ! Возможно сильное землетрясение."
                elif prob > 0.3:
                    status = "🔶 Умеренный риск. Рекомендуется следить за обновлениями."
                else:
                    status = "✅ Ситуация стабильна. Риск низкий."
                
                forecast_msg = (
                    f"🔮 <b>ПРОГНОЗ НА БЛИЖАЙШИЕ ДНИ</b>\n"
                    f"{'─' * 30}\n"
                    f"Вероятность M≥6.0: <b>{prob*100:.1f}%</b>\n"
                    f"Прогнозируемая магнитуда: <b>{pred_mag:.2f}</b>\n"
                    f"Количество событий в обучении: <b>{len(df_train)}</b>\n"
                    f"{'─' * 30}\n"
                    f"{status}"
                )
                
                try:
                    token = os.environ.get('TELEGRAM_BOT_TOKEN')
                    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
                    if token and chat_id:
                        url = f"https://api.telegram.org/bot{token}/sendMessage"
                        payload = {'chat_id': chat_id, 'text': forecast_msg, 'parse_mode': 'HTML'}
                        response = requests.post(url, data=payload, timeout=30)
                        if response.status_code == 200:
                            logger.info("✅ Прогноз отправлен в Telegram")
                        else:
                            logger.error(f"❌ Ошибка отправки прогноза: {response.text}")
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки прогноза: {e}")
        else:
            logger.info(f"ℹ️ Прогнозирование пропущено (нужно ≥20 событий, собрано {len(df_train)})")
    else:
        logger.info("ℹ️ Прогнозирование пропущено (модуль недоступен или нет данных)")

    # ═══════════════════════════════════════════════════════════════
    # 2. АНАЛИЗ LAIC
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🔬 ЭТАП 2: LAIC-анализ")
    logger.info("=" * 70)

    from laic_analyzer import LAICAnalyzer
    analyzer = LAICAnalyzer(settings)
    results = analyzer.analyze_all(data)

    # ═══════════════════════════════════════════════════════════════
    # 3. ОТПРАВКА ОСНОВНОГО ОТЧЁТА
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📤 ЭТАП 3: Отправка в Telegram")
    logger.info("=" * 70)

    if TG_AVAILABLE:
        try:
            success = send_sync(results)
            if success:
                logger.info("✅ Отчёт отправлен")
            else:
                logger.error("❌ Не удалось отправить отчёт")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки отчёта: {e}")

        alerts_sent = 0
        for r in results:
            if r.get('alert'):
                try:
                    alert_sync(r)
                    alerts_sent += 1
                except Exception as e:
                    logger.error(f"❌ Ошибка алерта {r.get('region', 'unknown')}: {e}")
    else:
        logger.warning("⚠️ Модуль Telegram не доступен, отчёт не отправлен")

    # ═══════════════════════════════════════════════════════════════
    # 4. ВИЗУАЛИЗАЦИЯ (PNG + HTML)
    # ═══════════════════════════════════════════════════════════════
    if VISUALIZER_AVAILABLE and TG_AVAILABLE and chat_id:
        logger.info("\n" + "=" * 70)
        logger.info("🎨 ЭТАП 4: Визуализация данных")
        logger.info("=" * 70)

        period_days = settings.get('analysis', {}).get('history_days', 30)

        for region_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            lst_info = lst_cache.get(region_name)

            try:
                img_bytes = plot_magnitude_series(df, region_name)
                if img_bytes:
                    caption = f"📊 {region_name} – временной ряд магнитуд (последние {period_days} дней)"
                    send_photo(chat_id, img_bytes, caption)
                    logger.info(f"✅ PNG-график для {region_name} отправлен")
            except Exception as e:
                logger.error(f"❌ Ошибка генерации PNG для {region_name}: {e}")

            try:
                html_path = create_interactive_report(region_name, df, lst_info, period_days=period_days)
                if html_path:
                    caption = f"📄 {region_name} – интерактивный отчёт"
                    send_document(chat_id, html_path, caption)
                    logger.info(f"✅ HTML-отчёт для {region_name} отправлен")
            except Exception as e:
                logger.error(f"❌ Ошибка генерации HTML для {region_name}: {e}")

    elif not VISUALIZER_AVAILABLE:
        logger.info("ℹ️ Визуализация пропущена (visualizer.py не найден)")
    elif not TG_AVAILABLE:
        logger.info("ℹ️ Визуализация пропущена (Telegram модуль не доступен)")

    # ═══════════════════════════════════════════════════════════════
    # ИТОГ
    # ═══════════════════════════════════════════════════════════════
    critical = sum(1 for r in results if r.get('risk_level') == 'critical')
    high = sum(1 for r in results if r.get('risk_level') == 'high')

    logger.info("\n" + "=" * 70)
    logger.info("✅ АНАЛИЗ ЗАВЕРШЁН")
    logger.info(f"   Регионов: {len(results)}")
    logger.info(f"   🔴 Критических: {critical}")
    logger.info(f"   🟠 Высоких: {high}")
    if TG_AVAILABLE:
        logger.info(f"   🚨 Алертов отправлено: {alerts_sent if 'alerts_sent' in locals() else 0}")
    logger.info("=" * 70)

    if critical > 0:
        sys.exit(0)


if __name__ == '__main__':
    main()
