#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_sender.py — отправка отчётов в Telegram
Поддержка прокси (SOCKS5/HTTP) через переменную PROXY_URL
"""

import os
import requests
import asyncio
import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.constants import ParseMode
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.request import HTTPXRequest

logger = logging.getLogger(__name__)

# ==========================================
# ПОДДЕРЖКА ПРОКСИ
# ==========================================
PROXY_URL = os.environ.get('PROXY_URL', '')

def get_proxy_dict():
    """Возвращает словарь прокси для requests, если PROXY_URL задан."""
    if not PROXY_URL:
        return {}
    return {
        'http': PROXY_URL,
        'https': PROXY_URL
    }

def get_bot_with_proxy(token):
    """Создаёт Bot с поддержкой прокси."""
    if PROXY_URL:
        try:
            request = HTTPXRequest(proxy=PROXY_URL, connection_pool_size=8)
            return Bot(token=token, request=request)
        except Exception as e:
            logger.warning(f"⚠️ Не удалось настроить прокси для Bot: {e}. Использую без прокси.")
            return Bot(token=token)
    return Bot(token=token)

# ==========================================
# ФУНКЦИИ ОТПРАВКИ
# ==========================================

def send_photo(chat_id: str, photo_bytes: bytes, caption: str = "") -> bool:
    """Отправляет изображение (PNG) в Telegram."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        return False

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('image.png', photo_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption}

    try:
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=90,  # увеличен таймаут
            proxies=get_proxy_dict()
        )
        if response.status_code != 200:
            logger.error(f"Ошибка отправки фото: {response.text}")
            return False
        return True
    except requests.exceptions.Timeout:
        logger.error("❌ Таймаут при отправке фото (соединение слишком долгое)")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ошибка соединения при отправке фото (возможно, блокировка)")
        return False
    except Exception as e:
        logger.error(f"❌ Исключение при отправке фото: {e}")
        return False


def send_document(chat_id: str, file_path: str, caption: str = "") -> bool:
    """Отправляет файл (HTML) в Telegram и удаляет его после отправки."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        return False

    url = f"https://api.telegram.org/bot{token}/sendDocument"

    try:
        with open(file_path, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': chat_id, 'caption': caption}
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=90,
                proxies=get_proxy_dict()
            )

        # Удаляем файл после отправки (даже если ошибка)
        try:
            os.unlink(file_path)
        except:
            pass

        if response.status_code != 200:
            logger.error(f"Ошибка отправки документа: {response.text}")
            return False
        return True

    except requests.exceptions.Timeout:
        logger.error("❌ Таймаут при отправке документа")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ошибка соединения при отправке документа")
        return False
    except Exception as e:
        logger.error(f"❌ Исключение при отправке документа: {e}")
        return False


# Токены из переменных окружения
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


class TelegramSender:
    """Отправляет отчёты и алерты в Telegram."""

    def __init__(self):
        if not BOT_TOKEN or not CHAT_ID:
            raise ValueError("TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть заданы!")

        # Используем бот с поддержкой прокси
        self.bot = get_bot_with_proxy(BOT_TOKEN)
        self.chat_id = CHAT_ID
        logger.info(f"✅ TelegramSender инициализирован (прокси: {'да' if PROXY_URL else 'нет'})")

    def _format_report(self, results):
        """Форматирует расширенный отчёт."""
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        region_results = [r for r in results if r.get('type') != 'stats']
        if not region_results:
            return "⚠️ Нет данных для отображения"

        lines = [
            "🌍 <b>LAIC МОНИТОРИНГ ЗЕМЛЕТРЯСЕНИЙ</b>",
            f"📅 {now}",
            "─" * 38,
            ""
        ]

        # Общая статистика
        total_events = sum(r.get('stats', {}).get('total_events', 0) for r in region_results)
        max_mag = max((r.get('stats', {}).get('max_mag_7d', 0) for r in region_results), default=0)
        lines.append(f"📊 Всего событий: {total_events}")
        lines.append(f"🔥 Макс. магнитуда: {max_mag:.1f}")
        lines.append("")

        # По регионам (с приоритетом zone_name)
        for r in region_results:
            region = r.get('zone_name') or r.get('region', 'Unknown')
            risk_score = r.get('risk_score', 0)
            emoji = r.get('emoji', '')
            stats = r.get('stats', {})

            # Координаты (если есть)
            coords = r.get('zone_coords', [0, 0])
            coord_str = f" ({coords[0]:.2f}, {coords[1]:.2f})" if coords != [0, 0] else ""

            lines.append(f"{emoji} <b>{region}</b>{coord_str} – <code>{risk_score:.0f}/100</code>")
            if stats.get('max_mag_7d', 0) > 0:
                lines.append(f"  • Mmax7д = {stats['max_mag_7d']:.1f}")
            if stats.get('events_24h', 0) > 0:
                lines.append(f"  • N24ч = {stats['events_24h']}")
            if stats.get('swarm_detected', False):
                lines.append(f"  • ⚡ Рой!")
            if stats.get('trend_detected', False):
                lines.append(f"  • 📈 Тренд↑")
            lines.append("")

        # Аномалии
        anomaly_count = sum(1 for r in region_results if r.get('stats', {}).get('iono_anomaly', False))
        if anomaly_count > 0:
            lines.append(f"🧠 Обнаружено аномалий: {anomaly_count}")
            lines.append("")

        lines.append("─" * 38)
        lines.append("📎 Интерактивный отчёт приложен файлом.")
        lines.append("🌐 Источник: USGS, NOAA, NASA OMNIWeb")

        return "\n".join(lines)

    def _format_alert(self, result):
        """Форматирует срочное оповещение с названием зоны."""

        # Получаем название зоны (приоритет: zone_name > region)
        zone_name = result.get('zone_name') or result.get('region', 'Unknown')
        coords = result.get('zone_coords', [0, 0])

        stats = result.get('stats', {})
        if not stats:
            stats = {
                'max_mag_7d': result.get('max_mag_7d', 0),
                'events_24h': result.get('events_24h', 0),
                'energy_7d': result.get('energy_7d', 0)
            }

        lines = [
            f"🚨 <b>ВНИМАНИЕ! ВЫСОКИЙ РИСК ЗЕМЛЕТРЯСЕНИЯ</b>",
            f"{'─' * 35}",
            f"📍 <b>Зона:</b> {zone_name}",
            f"   Координаты: {coords[0]:.2f}, {coords[1]:.2f}",
            f"",
            f"Уровень: <b>{result.get('risk_level', 'unknown').upper()}</b>",
            f"Индекс риска: <code>{result.get('risk_score', 0):.0f}/100</code>",
            f"",
            f"📊 Данные:",
            f"  • Макс M за 7д: {stats.get('max_mag_7d', 0):.1f}",
            f"  • Событий за 24ч: {stats.get('events_24h', 0)}",
            f"  • Энергия за 7д: {stats.get('energy_7d', 0):.2f} кт TNT",
        ]

        if result.get('swarm', {}).get('is_swarm', False):
            lines.append(f"  • ⚡ Обнаружен РОЙ землетрясений!")

        if result.get('magnitude_trend', {}).get('trend_detected', False):
            lines.append(f"  • 📈 Нарастание магнитуд!")

        lines.extend([
            f"",
            f"💡 {result.get('recommendation', 'Будьте внимательны')[:150]}...",
            f"",
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        ])

        return '\n'.join(lines)
    

    async def send_report(self, results):
        """Отправляет ежедневный отчёт."""
        message = self._format_report(results)

        # Создаем кнопки
        keyboard = [
            [
                InlineKeyboardButton("Да", callback_data="regions_yes"),
                InlineKeyboardButton("Нет", callback_data="regions_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if len(message) > 4000:
            message = message[:3950] + "\n\n... (сообщение обрезано)"

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
                disable_web_page_preview=True
            )
            logger.info("✅ Отчёт отправлен в Telegram")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки отчёта: {e}")
            return False

    async def send_alert(self, result):
        """Отправляет срочное оповещение."""
        message = self._format_alert(result)

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
            logger.info(f"🚨 АЛЕРТ отправлен: {result.get('region', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки алерта: {e}")
            return False


# ==========================================
# СИНХРОННЫЕ ОБЁРТКИ
# ==========================================

def send_sync(results):
    """Синхронная обёртка для отправки отчёта."""
    try:
        sender = TelegramSender()
        return asyncio.run(sender.send_report(results))
    except Exception as e:
        logger.error(f"❌ Ошибка в send_sync: {e}")
        return False


def alert_sync(result):
    """Синхронная обёртка для отправки алерта."""
    try:
        sender = TelegramSender()
        return asyncio.run(sender.send_alert(result))
    except Exception as e:
        logger.error(f"❌ Ошибка в alert_sync: {e}")
        return False
