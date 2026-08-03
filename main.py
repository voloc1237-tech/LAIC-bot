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

# Логи в stdout (GitHub Actions показывает)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('LAIC')

# ═══════════════════════════════════════════════════════════════
# GEE — опционально
# ═══════════════════════════════════════════════════════════════
try:
    from ee_collector import GEEInitializer, get_lst_data
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("⚠️ Earth Engine модуль не найден")

# ═══════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ — опционально
# ═══════════════════════════════════════════════════════════════
try:
    from visualizer import plot_magnitude_series, create_interactive_report
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    logger.warning("⚠️ Модуль визуализации не найден (visualizer.py)")

# ═══════════════════════════════════════════════════════════════
# TELEGRAM ОТПРАВКА (дополнительные функции)
# ═══════════════════════════════════════════════════════════════
from telegram_sender import send_sync, alert_sync, send_photo, send_document


def main():
    logger.info("=" * 70)
    logger.info("🚀 LAIC EARTHQUAKE MONITOR")
    logger.info(f"⏰ {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 70)

    # Проверка токенов
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        logger.error("❌ TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы!")
        logger.error("Добавьте в Settings → Secrets → Actions")
        sys.exit(1)

    logger.info("✅ Токены Telegram найдены")

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
                logger.warning("⚠️ Earth Engine не инициализирован — работаем без спутниковых данных")
        except Exception as e:
            logger.warning(f"⚠️ Earth Engine ошибка: {e}")
    else:
        logger.info("ℹ️ Earth Engine не настроен")

    # Загрузка настроек
    import yaml
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    logger.info(f"✅ Настройки загружены: {len(settings['regions'])} регионов")

    # ═══════════════════════════════════════════════════════════════
    # 1. СБОР ДАННЫХ USGS
    # ═══════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 70)
    logger.info("📥 ЭТАП 1: Сбор данных USGS")
    logger.info("=" * 70)

    from data_collector import collect_all_regions
    data = collect_all_regions(settings)

    total = sum(len(df) for df in data.values())
    logger.info(f"\n📊 Всего событий: {total}")

    # ═══════════════════════════════════════════════════════════════
    # 1b. СБОР СПУТНИКОВЫХ ДАННЫХ (GEE) — если доступен
    # ═══════════════════════════════════════════════════════════════

    lst_cache = {}  # словарь для хранения LST-данных по регионам

    if ee_initialized:
        logger.info("\n" + "=" * 70)
        logger.info("🛰️ ЭТАП 1b: Сбор спутниковых данных (GEE)")
        logger.info("=" * 70)

        for region_name, region_data in data.items():
            if not region_data.empty:
                # Берём последнее событие
                last_event = region_data.iloc[0]
                lat = last_event['latitude']
                lon = last_event['longitude']

                try:
                    # Температура за последние 7 дней
                    end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')

                    lst_data = get_lst_data(lat, lon, start_date, end_date)
                    lst_cache[region_name] = lst_data
                    logger.info(f"🌡️ {region_name}: LST = {lst_data['lst_celsius']}°C (изображений: {lst_data['count']})")

                except Exception as e:
                    logger.warning(f"⚠️ Не удалось получить LST для {region_name}: {e}")
                    lst_cache[region_name] = None
    else:
        logger.info("ℹ️ Пропускаем спутниковые данные (GEE недоступен)")

    
    # После этапа 1b, перед анализом

# ═══════════════════════════════════════════════════════════════
# 1c. СБОР ДОПОЛНИТЕЛЬНЫХ ДАННЫХ (погода, ионосфера, космическая погода)
# для сильных событий (M >= 5.0)
# ═══════════════════════════════════════════════════════════════

from weather_collector import WeatherCollector
from ionosphere_collector import IonosphereCollector
from space_weather import SpaceWeatherCollector

weather = WeatherCollector()
iono = IonosphereCollector()
space = SpaceWeatherCollector()

# Словари для хранения собранных данных по каждому событию
event_weather = {}
event_iono = {}
event_space = {}

# Определим порог магнитуды для детального анализа
min_mag_for_detail = settings.get('analysis', {}).get('detail_min_magnitude', 5.0)

if not data.empty:
    # Объединим все регионы в один DataFrame (если глобальный режим)
    if 'global' in data:
        all_events = data['global']
    else:
        all_events = pd.concat(data.values(), ignore_index=True)
    
    # Выберем сильные события
    strong_events = all_events[all_events['magnitude'] >= min_mag_for_detail]
    logger.info(f"📌 Сильных событий (M≥{min_mag_for_detail}): {len(strong_events)}")
    
    for idx, row in strong_events.iterrows():
        event_id = row['id']
        lat = row['latitude']
        lon = row['longitude']
        event_time = row['time']
        mag = row['magnitude']
        
        logger.info(f"🔍 Сбор данных для события {event_id} M{mag:.1f} в {event_time}")
        
        # Погода: 7 дней до, 3 дня после
        try:
            wdf = weather.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
            if not wdf.empty:
                event_weather[event_id] = wdf
                logger.info(f"   🌡️ Погода: {len(wdf)} дней")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка погоды: {e}")
        
        # Ионосфера: TEC, ROTI
        try:
            iono_data = iono.fetch_for_event(event_time, lat, lon, days_before=7, days_after=3)
            if iono_data['tec'] is not None and not iono_data['tec'].empty:
                event_iono[event_id] = iono_data
                logger.info(f"   🛰️ Ионосфера: TEC {len(iono_data['tec'])} записей")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка ионосферы: {e}")
        
        # Космическая погода: Kp, Dst, F10.7 за тот же период
        try:
            start = event_time - timedelta(days=7)
            end = event_time + timedelta(days=3)
            space_data = space.fetch_all_for_period(start, end)
            if any(not df.empty for df in space_data.values()):
                event_space[event_id] = space_data
                logger.info(f"   ☀️ Космическая погода: загружена")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка космической погоды: {e}")

# Сохраним собранные данные в кэш или в файл для последующего анализа
# (опционально: сохранить в JSON для отчетов)
    
    # ═══════════════════════════════════════════════════════════════
    # 2. АНАЛИЗ
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

    # Основной отчёт
    try:
        success = send_sync(results)
        if success:
            logger.info("✅ Отчёт отправлен")
        else:
            logger.error("❌ Не удалось отправить отчёт")
    except Exception as e:
        logger.error(f"❌ Ошибка отправки отчёта: {e}")

    # Алерты для критических
    alerts_sent = 0
    for r in results:
        if r['alert']:
            try:
                alert_sync(r)
                alerts_sent += 1
            except Exception as e:
                logger.error(f"❌ Ошибка алерта {r['region']}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 4. ВИЗУАЛИЗАЦИЯ (PNG + HTML) – если доступна
    # ═══════════════════════════════════════════════════════════════

    if VISUALIZER_AVAILABLE and chat_id:
        logger.info("\n" + "=" * 70)
        logger.info("🎨 ЭТАП 4: Визуализация данных")
        logger.info("=" * 70)

        for region_name, df in data.items():
            if df.empty:
                continue

            # Получаем LST-данные (если есть)
            lst_info = lst_cache.get(region_name)

            # 4a. Простой график (PNG)
            try:
                img_bytes = plot_magnitude_series(df, region_name)
                if img_bytes:
                    caption = f"📊 {region_name} – временной ряд магнитуд за последние 30 дней"
                    send_photo(chat_id, img_bytes, caption)
                    logger.info(f"✅ PNG-график для {region_name} отправлен")
                else:
                    logger.warning(f"⚠️ Не удалось сгенерировать график для {region_name}")
            except Exception as e:
                logger.error(f"❌ Ошибка генерации PNG для {region_name}: {e}")

            # 4b. Интерактивный HTML-отчёт
            try:
                html_path = create_interactive_report(region_name, df, lst_info, period_days=30)
                if html_path:
                    caption = f"📄 {region_name} – интерактивный отчёт (карта, график, таблица)"
                    send_document(chat_id, html_path, caption)
                    logger.info(f"✅ HTML-отчёт для {region_name} отправлен")
                else:
                    logger.warning(f"⚠️ Не удалось сгенерировать HTML для {region_name}")
            except Exception as e:
                logger.error(f"❌ Ошибка генерации HTML для {region_name}: {e}")

    elif not VISUALIZER_AVAILABLE:
        logger.info("ℹ️ Визуализация пропущена (модуль visualizer.py не найден)")
    elif not chat_id:
        logger.warning("⚠️ chat_id не задан – визуализация не отправлена")

    # ═══════════════════════════════════════════════════════════════
    # ИТОГ
    # ═══════════════════════════════════════════════════════════════

    critical = sum(1 for r in results if r['risk_level'] == 'critical')
    high = sum(1 for r in results if r['risk_level'] == 'high')

    logger.info("\n" + "=" * 70)
    logger.info("✅ АНАЛИЗ ЗАВЕРШЁН")
    logger.info(f"   Регионов: {len(results)}")
    logger.info(f"   🔴 Критических: {critical}")
    logger.info(f"   🟠 Высоких: {high}")
    logger.info(f"   🚨 Алертов отправлено: {alerts_sent}")
    logger.info("=" * 70)

    # Код выхода для GitHub Actions
    if critical > 0:
        sys.exit(0)  # Не считать ошибкой


if __name__ == '__main__':
    main()
