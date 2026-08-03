#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — ГЛАВНЫЙ ФАЙЛ LAIC-бота
Собирает → Анализирует → Отправляет
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

# Логи в stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('LAIC')

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
                    lst_cache[region_name] = lst_data
                    logger.info(f"🌡️ {region_name}: LST = {lst_data['lst_celsius']}°C (изобр: {lst_data['count']})")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось получить LST для {region_name}: {e}")
                    lst_cache[region_name] = None
    else:
        logger.info("ℹ️ Пропускаем спутниковые данные (GEE недоступен)")

    # ═══════════════════════════════════════════════════════════════
    # 1c. СБОР ДОПОЛНИТЕЛЬНЫХ ДАННЫХ (погода, ионосфера, космос)
    #     только для сильных событий (M >= detail_min_magnitude)
    # ═══════════════════════════════════════════════════════════════
    min_mag_for_detail = settings.get('analysis', {}).get('detail_min_magnitude', 5.0)
    event_weather = {}
    event_iono = {}
    event_space = {}

    # Объединяем все данные в один DataFrame (если глобальный режим) или по регионам
    all_events = pd.concat(data.values(), ignore_index=True) if data else pd.DataFrame()
    strong_events = all_events[all_events['magnitude'] >= min_mag_for_detail] if not all_events.empty else pd.DataFrame()

    if not strong_events.empty:
        logger.info("\n" + "=" * 70)
        logger.info(f"📌 ЭТАП 1c: Сбор дополнительных данных для {len(strong_events)} сильных событий (M≥{min_mag_for_detail})")
        logger.info("=" * 70)

        # Инициализация сборщиков (если доступны)
        weather = WeatherCollector() if WEATHER_AVAILABLE else None
        iono = IonosphereCollector() if IONO_AVAILABLE else None
        space = SpaceWeatherCollector() if SPACE_AVAILABLE else None

        for idx, row in strong_events.iterrows():
            event_id = row['id']
            lat = row['latitude']
            lon = row['longitude']
            event_time = row['time']
            mag = row['magnitude']
            logger.info(f"🔍 Событие {event_id} M{mag:.1f} в {event_time}")

            # Погода (7 дней до, 3 после)
            if weather:
                try:
                    wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    if not wdf.empty:
                        event_weather[event_id] = wdf
                        logger.info(f"   🌡️ Погода: {len(wdf)} дней")
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка погоды: {e}")

            # Ионосфера (TEC, ROTI)
            if iono:
                try:
                    iono_data = iono.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
                    if iono_data['tec'] is not None and not iono_data['tec'].empty:
                        event_iono[event_id] = iono_data
                        logger.info(f"   🛰️ Ионосфера: TEC {len(iono_data['tec'])} записей")
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

        logger.info(f"✅ Собрано: погода для {len(event_weather)} событий, ионосфера для {len(event_iono)}, космос для {len(event_space)}")
    else:
        logger.info("ℹ️ Нет сильных событий для детального сбора данных.")

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
        # Основной отчёт
        try:
            success = send_sync(results)
            if success:
                logger.info("✅ Отчёт отправлен")
            else:
                logger.error("❌ Не удалось отправить отчёт")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки отчёта: {e}")

        # Алерты
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
            if df.empty:
                continue

            lst_info = lst_cache.get(region_name)

            # PNG-график
            try:
                img_bytes = plot_magnitude_series(df, region_name)
                if img_bytes:
                    caption = f"📊 {region_name} – временной ряд магнитуд (последние {period_days} дней)"
                    send_photo(chat_id, img_bytes, caption)
                    logger.info(f"✅ PNG-график для {region_name} отправлен")
            except Exception as e:
                logger.error(f"❌ Ошибка генерации PNG для {region_name}: {e}")

            # HTML-отчёт
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
        sys.exit(0)  # Не считать ошибкой


if __name__ == '__main__':
    main()
