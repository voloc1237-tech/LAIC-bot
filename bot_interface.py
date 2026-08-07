#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_interface.py — интерактивный интерфейс управления через aiogram 3.x
Обрабатывает кнопки из главного отчета, выбор регионов, городов и расчет точек.
"""

import logging
import pandas as pd
from datetime import datetime, timezone
from aiogram import Dispatcher, F
from aiogram.types import CallbackQuery, Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from data_collector import USGSCollector
from laic_analyzer import LAICAnalyzer

logger = logging.getLogger(__name__)

# Состояния FSM для пошагового диалога
class ForecastStates(StatesGroup):
    waiting_for_region = State()
    waiting_for_point_choice = State()
    waiting_for_coordinates = State()


def get_settings():
    """Загрузчик настроек из settings.yaml (или передается из main)."""
    import yaml
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ═════════════════════════════════════════════════════════════════════════
# 1. ОБРАБОТКА ПЕРВЫХ КНОПОК ПОСЛЕ ГЛОБАЛЬНОГО ОТЧЁТА ("Да" / "Нет")
# ═════════════════════════════════════════════════════════════════════════

async def handle_regions_yes(callback: CallbackQuery, state: FSMContext):
    """Пользователь нажал «Да» — выводим список доступных регионов из конфига."""
    settings = get_settings()
    regions = settings.get('regions', {})
    
    if not regions:
        await callback.message.answer("⚠️ Список регионов в конфигурации не найден.")
        await callback.answer()
        return

    builder = InlineKeyboardBuilder()
    for key, cfg in regions.items():
        builder.button(text=cfg.get('name', key), callback_data=f"reg_{key}")
    
    # Кнопка для ручного ввода координат/города
    builder.button(text="📍 Ввести координаты / город", callback_data="reg_custom")
    builder.adjust(1)  пять в один столбец

    await callback.message.answer(
        "🗺 **Выберите регион для детального анализа** или укажите свою точку:",
        reply_markup=builder.as_markup(),
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_region)
    await callback.answer()


async def handle_regions_no(callback: CallbackQuery, state: FSMContext):
    """Пользователь нажал «Нет» — завершаем диалог."""
    await callback.message.answer("👍 Хорошо, глобальный мониторинг продолжается в обычном режиме.")
    await state.clear()
    await callback.answer()


# ═════════════════════════════════════════════════════════════════════════
# 2. ВЫБОР КОНКРЕТНОГО РЕГИОНА ИЗ СПИСКА
# ═════════════════════════════════════════════════════════════════════════

async def handle_region_selection(callback: CallbackQuery, state: FSMContext):
    """Обрабатывает выбор конкретного региона из инлайн-меню."""
    region_key = callback.data.replace("reg_", "")
    settings = get_settings()
    regions = settings.get('regions', {})
    
    if region_key not in regions:
        await callback.message.answer("❌ Регион не найден в конфигурации.")
        await callback.answer()
        return

    cfg = regions[region_key]
    await state.update_data(region_key=region_key, region_config=cfg)
    
    # Спрашиваем, нужен ли точечный анализ внутри региона
    builder = InlineKeyboardBuilder()
    builder.button(text="🎯 Да, уточнить город / точку", callback_data="point_yes")
    builder.button(text="📊 Нет, проанализировать весь регион", callback_data="point_no")
    builder.adjust(1)

    await callback.message.answer(
        f"Выбран регион: **{cfg.get('name')}**.\nХотите ввести конкретный город или координаты внутри этого региона?",
        reply_markup=builder.as_markup(),
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_point_choice)
    await callback.answer()


# ═════════════════════════════════════════════════════════════════════════
# 3. ТОЧЕЧНЫЙ АНАЛИЗ (ГОРОД ИЛИ КООРДИНАТЫ)
# ═════════════════════════════════════════════════════════════════════════

async def handle_point_choice_yes(callback: CallbackQuery, state: FSMContext):
    """Пользователь хочет ввести точку/город."""
    await callback.message.answer(
        "✍️ Введите название города (например, *Токио*) или координаты через запятую (например, `35.68, 139.65`):",
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_coordinates)
    await callback.answer()


async def handle_point_choice_no(callback: CallbackQuery, state: FSMContext):
    """Пользователь отказался от точки — запускаем расчет по всему региону."""
    data = await state.get_data()
    region_key = data.get('region_key')
    cfg = data.get('region_config')
    
    await callback.message.answer(f"⏳ Собираю данные и запускаю LAIC-анализ для региона **{cfg.get('name')}**...")
    
    settings = get_settings()
    collector = USGSCollector(
        cache_enabled=settings.get('cache', {}).get('enabled', True),
        cache_max_age_hours=settings.get('cache', {}).get('max_age_hours', 6)
    )
    analyzer = LAICAnalyzer(settings)
    
    # Сбор данных по региону
    days_back = settings.get('analysis', {}).get('history_days', 45)
    df = collector.fetch_for_region(cfg, days_back=days_back)
    
    # Расчет
    result = analyzer.analyze_region(df, cfg.get('name'), cfg)
    
    # Формируем отчет для пользователя
    text = (
        f"📊 **Результат анализа: {result['region']}** {result['emoji']}\n\n"
        f"• **Уровень риска:** {result['risk_level'].upper()} (Балл: {result['risk_score']})\n"
        f"• **Факторы:** {result['risk_factors']}\n"
        f"• **Макс. магнитуда (7д):** {result['stats']['max_mag_7d']}\n"
        f"• **Событий за 24ч:** {result['stats']['events_24h']}\n\n"
        f"💡 *Рекомендация:* {result['recommendation']}"
    )
    
    await callback.message.answer(text, parse_mode="Markdown")
    await state.clear()
    await callback.answer()


async def handle_custom_location_input(message: Message, state: FSMContext):
    """Обрабатывает введенный текст (город или координаты) и выполняет локальный радиусный анализ."""
    text_input = message.text.strip()
    settings = get_settings()
    
    lat, lon = None, None
    location_name = text_input

    # 1. Пробуем распарсить как координаты ("lat, lon")
    try:
        parts = text_input.replace(";", ",").split(",")
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
    except ValueError:
        pass

    # 2. Если не координаты, пробуем найти через простой словарь или геокодер
    if lat is None or lon is None:
        # Простой встроенный словарь популярных точек для примера
        known_cities = {
            "токио": (35.6762, 139.6503),
            "стамбул": (41.0082, 28.9784),
            "лос-анджелес": (34.0522, -118.2437),
            "петропавловск-камчатский": (53.0452, 158.6559)
        }
        city_key = text_input.lower()
        if city_key in known_cities:
            lat, lon = known_cities[city_key]
        else:
            # Попытка использовать geopy, если установлена
            try:
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="laic_bot_geo")
                loc = geolocator.geocode(text_input)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                    location_name = loc.address
            except Exception:
                pass

    if lat is None or lon is None:
        await message.answer("❌ Не удалось распознать координаты или найти город. Попробуйте ввести координаты в формате `35.68, 139.65`.")
        return

    await message.answer(f"🔍 Ищу события в радиусе 150 км от точки: **{location_name}** ({lat}, {lon})...")

    # Сбор глобальных данных за период и фильтрация по радиусу (например, ~1.5 градуса ≈ 150 км)
    collector = USGSCollector(
        cache_enabled=settings.get('cache', {}).get('enabled', True),
        cache_max_age_hours=settings.get('cache', {}).get('max_age_hours', 6)
    )
    analyzer = LAICAnalyzer(settings)
    
    days_back = settings.get('analysis', {}).get('history_days', 45)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days_back)
    
    # Забираем глобальные данные
    min_mag = settings.get('analysis', {}).get('global_min_magnitude', 4.0)
    df_global = collector.fetch_global(start, end, min_magnitude=min_mag)
    
    if df_global.empty:
        await message.answer("⚠️ Не удалось получить данные землетрясений для анализа точки.")
        await state.clear()
        return

    # Фильтрация по радиусу (приблизительно 1.5 градуса по широте/долготе)
    radius_deg = 1.5
    df_point = df_global[
        (df_global['latitude'] >= lat - radius_deg) & 
        (df_global['latitude'] <= lat + radius_deg) & 
        (df_global['longitude'] >= lon - radius_deg) & 
        (df_global['longitude'] <= lon + radius_deg)
    ].copy()

    # Запускаем LAIC-анализ для этой подвыборки
    result = analyzer.analyze_region(df_point, f"Точка: {location_name}", {})
    
    report_text = (
        f"🎯 **Локальный прогноз для точки:** {location_name} {result['emoji']}\n"
        f"📍 Координаты: `{lat:.2f}, {lon:.2f}`\n\n"
        f"• **Уровень риска:** {result['risk_level'].upper()} (Балл: {result['risk_score']})\n"
        f"• **Факторы:** {result['risk_factors']}\n"
        f"• **Событий в радиусе:** {result['stats']['total_events']}\n"
        f"• **Макс. магнитуда (7д):** {result['stats']['max_mag_7d']}\n\n"
        f"💡 *Рекомендация:* {result['recommendation']}"
    )

    await message.answer(report_text, parse_mode="Markdown")
    await state.clear()


# ═════════════════════════════════════════════════════════════════════════
- РЕГИСТРАЦИЯ ХЕНДЛЕРОВ В ДИСПЕТЧЕРЕ
# ═════════════════════════════════════════════════════════════════════════

def register_handlers(dp: Dispatcher):
    """Регистрирует все интерактивные колбэки и состояния в диспетчере aiogram."""
    # 1. Ответы на кнопки главного отчета
    dp.callback_query.register(handle_regions_yes, F.data == "regions_yes")
    dp.callback_query.register(handle_regions_no, F.data == "regions_no")
    
    # 2. Выбор региона
    dp.callback_query.register(handle_region_selection, F.data.startswith("reg_") & (F.data != "reg_custom"))
    
    # Если нажали "Ввести координаты" из общего списка регионов
    @dp.callback_query(F.data == "reg_custom")
    async def custom_reg_handler(callback: CallbackQuery, state: FSMContext):
        await callback.message.answer("✍️ Введите координаты (например, `35.68, 139.65`) или название города:")
        await state.set_state(ForecastStates.waiting_for_coordinates)
        await callback.answer()

    # 3. Выбор точечного анализа внутри региона
    dp.callback_query.register(handle_point_choice_yes, F.data == "point_yes")
    dp.callback_query.register(handle_point_choice_no, F.data == "point_no")
    
    # 4. Текстовый ввод координат или города (FSM)
    dp.message.register(handle_custom_location_input, ForecastStates.waiting_for_coordinates)
    
    logger.info("✅ Обработчики bot_interface успешно зарегистрированы.")
