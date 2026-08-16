#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_interface.py — interactive management interface via aiogram 3.x
Handles buttons from the main report, region selection, cities, and point calculations.
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

# FSM states for step-by-step dialog
class ForecastStates(StatesGroup):
    waiting_for_region = State()
    waiting_for_point_choice = State()
    waiting_for_coordinates = State()


def get_settings():
    """Load settings from settings.yaml."""
    import yaml
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ============================================================
# 1. HANDLE FIRST BUTTONS AFTER GLOBAL REPORT ("Yes" / "No")
# ============================================================

async def handle_regions_yes(callback: CallbackQuery, state: FSMContext):
    """User clicked 'Yes' — show list of available regions from config."""
    settings = get_settings()
    regions = settings.get('regions', {})
    
    if not regions:
        await callback.message.answer("⚠️ Regions list not found in configuration.")
        await callback.answer()
        return

    builder = InlineKeyboardBuilder()
    for key, cfg in regions.items():
        builder.button(text=cfg.get('name', key), callback_data=f"reg_{key}")
    
    # Button for manual coordinates/city input
    builder.button(text="📍 Enter coordinates / city", callback_data="reg_custom")
    builder.adjust(1)  # one column

    await callback.message.answer(
        "🗺 **Select a region for detailed analysis** or specify your location:",
        reply_markup=builder.as_markup(),
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_region)
    await callback.answer()


async def handle_regions_no(callback: CallbackQuery, state: FSMContext):
    """User clicked 'No' — end the dialog."""
    await callback.message.answer("👍 Okay, global monitoring continues in normal mode.")
    await state.clear()
    await callback.answer()


# ============================================================
# 2. HANDLE SPECIFIC REGION SELECTION
# ============================================================

async def handle_region_selection(callback: CallbackQuery, state: FSMContext):
    """Handle selection of a specific region from the inline menu."""
    region_key = callback.data.replace("reg_", "")
    settings = get_settings()
    regions = settings.get('regions', {})
    
    if region_key not in regions:
        await callback.message.answer("❌ Region not found in configuration.")
        await callback.answer()
        return

    cfg = regions[region_key]
    await state.update_data(region_key=region_key, region_config=cfg)
    
    # Ask if user wants point analysis inside the region
    builder = InlineKeyboardBuilder()
    builder.button(text="🎯 Yes, specify city / point", callback_data="point_yes")
    builder.button(text="📊 No, analyze the whole region", callback_data="point_no")
    builder.adjust(1)

    await callback.message.answer(
        f"Selected region: **{cfg.get('name')}**.\nDo you want to enter a specific city or coordinates within this region?",
        reply_markup=builder.as_markup(),
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_point_choice)
    await callback.answer()


# ============================================================
# 3. POINT ANALYSIS (CITY OR COORDINATES)
# ============================================================

async def handle_point_choice_yes(callback: CallbackQuery, state: FSMContext):
    """User wants to enter a point/city."""
    await callback.message.answer(
        "✍️ Enter a city name (e.g., *Tokyo*) or coordinates separated by comma (e.g., `35.68, 139.65`):",
        parse_mode="Markdown"
    )
    await state.set_state(ForecastStates.waiting_for_coordinates)
    await callback.answer()


async def handle_point_choice_no(callback: CallbackQuery, state: FSMContext):
    """User declined point input — run analysis for the whole region."""
    data = await state.get_data()
    region_key = data.get('region_key')
    cfg = data.get('region_config')
    
    await callback.message.answer(f"⏳ Collecting data and running LAIC analysis for **{cfg.get('name')}**...")
    
    settings = get_settings()
    collector = USGSCollector(
        cache_enabled=settings.get('cache', {}).get('enabled', True),
        cache_max_age_hours=settings.get('cache', {}).get('max_age_hours', 6)
    )
    analyzer = LAICAnalyzer(settings)
    
    # Collect data for region
    days_back = settings.get('analysis', {}).get('history_days', 45)
    df = collector.fetch_for_region(cfg, days_back=days_back)
    
    # Calculate
    result = analyzer.analyze_region(df, cfg.get('name'), cfg)
    
    # Format report for user
    text = (
        f"📊 **Analysis result: {result['region']}** {result['emoji']}\n\n"
        f"• **Risk level:** {result['risk_level'].upper()} (Score: {result['risk_score']})\n"
        f"• **Factors:** {result['risk_factors']}\n"
        f"• **Max magnitude (7d):** {result['stats']['max_mag_7d']}\n"
        f"• **Events in 24h:** {result['stats']['events_24h']}\n\n"
        f"💡 *Recommendation:* {result['recommendation']}"
    )
    
    await callback.message.answer(text, parse_mode="Markdown")
    await state.clear()
    await callback.answer()


async def handle_custom_location_input(message: Message, state: FSMContext):
    """Handle user input (city or coordinates) and perform local radius analysis."""
    text_input = message.text.strip()
    settings = get_settings()
    
    lat, lon = None, None
    location_name = text_input

    # Try to parse as coordinates ("lat, lon")
    try:
        parts = text_input.replace(";", ",").split(",")
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
    except ValueError:
        pass

    # If not coordinates, try to find via simple dictionary or geocoder
    if lat is None or lon is None:
        known_cities = {
            "tokyo": (35.6762, 139.6503),
            "istanbul": (41.0082, 28.9784),
            "los angeles": (34.0522, -118.2437),
            "petropavlovsk-kamchatsky": (53.0452, 158.6559)
        }
        city_key = text_input.lower()
        if city_key in known_cities:
            lat, lon = known_cities[city_key]
        else:
            # Try using geopy if installed
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
        await message.answer("❌ Could not recognize coordinates or find city. Please try `35.68, 139.65`.")
        return

    await message.answer(f"🔍 Searching for events within 150 km radius of: **{location_name}** ({lat}, {lon})...")

    # Collect global data and filter by radius (~1.5 degrees ≈ 150 km)
    collector = USGSCollector(
        cache_enabled=settings.get('cache', {}).get('enabled', True),
        cache_max_age_hours=settings.get('cache', {}).get('max_age_hours', 6)
    )
    analyzer = LAICAnalyzer(settings)
    
    days_back = settings.get('analysis', {}).get('history_days', 45)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days_back)
    
    min_mag = settings.get('analysis', {}).get('global_min_magnitude', 4.0)
    df_global = collector.fetch_global(start, end, min_magnitude=min_mag)
    
    if df_global.empty:
        await message.answer("⚠️ Could not get earthquake data for point analysis.")
        await state.clear()
        return

    # Filter by radius (~1.5 degrees in lat/lon)
    radius_deg = 1.5
    df_point = df_global[
        (df_global['latitude'] >= lat - radius_deg) & 
        (df_global['latitude'] <= lat + radius_deg) & 
        (df_global['longitude'] >= lon - radius_deg) & 
        (df_global['longitude'] <= lon + radius_deg)
    ].copy()

    # Run LAIC analysis on this subset
    result = analyzer.analyze_region(df_point, f"Point: {location_name}", {})
    
    report_text = (
        f"🎯 **Local forecast for point:** {location_name} {result['emoji']}\n"
        f"📍 Coordinates: `{lat:.2f}, {lon:.2f}`\n\n"
        f"• **Risk level:** {result['risk_level'].upper()} (Score: {result['risk_score']})\n"
        f"• **Factors:** {result['risk_factors']}\n"
        f"• **Events in radius:** {result['stats']['total_events']}\n"
        f"• **Max magnitude (7d):** {result['stats']['max_mag_7d']}\n\n"
        f"💡 *Recommendation:* {result['recommendation']}"
    )

    await message.answer(report_text, parse_mode="Markdown")
    await state.clear()


# ============================================================
# REGISTER HANDLERS IN DISPATCHER
# ============================================================

def register_handlers(dp: Dispatcher):
    """Register all interactive callbacks and states in the aiogram dispatcher."""
    # 1. Responses to main report buttons
    dp.callback_query.register(handle_regions_yes, F.data == "regions_yes")
    dp.callback_query.register(handle_regions_no, F.data == "regions_no")
    
    # 2. Region selection
    dp.callback_query.register(handle_region_selection, F.data.startswith("reg_") & (F.data != "reg_custom"))
    
    # If user clicked "Enter coordinates" from the region list
    @dp.callback_query(F.data == "reg_custom")
    async def custom_reg_handler(callback: CallbackQuery, state: FSMContext):
        await callback.message.answer("✍️ Enter coordinates (e.g., `35.68, 139.65`) or city name:")
        await state.set_state(ForecastStates.waiting_for_coordinates)
        await callback.answer()

    # 3. Point analysis choice inside region
    dp.callback_query.register(handle_point_choice_yes, F.data == "point_yes")
    dp.callback_query.register(handle_point_choice_no, F.data == "point_no")
    
    # 4. Text input for coordinates or city (FSM)
    dp.message.register(handle_custom_location_input, ForecastStates.waiting_for_coordinates)
    
    logger.info("✅ bot_interface handlers registered successfully.")
