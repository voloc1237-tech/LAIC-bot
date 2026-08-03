#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_sender.py — отправка отчётов в Telegram
"""

import os
import requests
import asyncio
import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)

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
        response = requests.post(url, files=files, data=data, timeout=30)
        if response.status_code != 200:
            logger.error(f"Ошибка отправки фото: {response.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"Исключение при отправке фото: {e}")
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
            response = requests.post(url, files=files, data=data, timeout=30)
        os.unlink(file_path)  # удаляем временный файл
        if response.status_code != 200:
            logger.error(f"Ошибка отправки документа: {response.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"Исключение при отправке документа: {e}")
        return False

# Токены из переменных окружения
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


class TelegramSender:
    """Отправляет отчёты и алерты в Telegram."""
    
    def __init__(self):
        if not BOT_TOKEN or not CHAT_ID:
            raise ValueError("TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть заданы!")
        
        self.bot = Bot(token=BOT_TOKEN)
        self.chat_id = CHAT_ID
    
    def _format_report(self, results):
        """Форматирует полный ежедневный отчёт."""
        
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        
        # Заголовок
        lines = [
            f"🌍 <b>LAIC МОНИТОРИНГ ЗЕМЛЕТРЯСЕНИЙ</b>",
            f"📅 {now}",
            f"{'─' * 38}",
            ""
        ]
        
        # Фильтруем только регионы (исключаем элементы с type='stats')
        region_results = [r for r in results if r.get('type') != 'stats']
        
        if not region_results:
            lines.append("⚠️ Нет данных для отображения")
            return '\n'.join(lines)
        
        # Максимальный риск
        max_risk = max(region_results, key=lambda x: x.get('risk_score', 0))
        
        if max_risk.get('risk_score', 0) > 0:
            lines.extend([
                f"<b>🔥 Наибольший риск:</b> {max_risk.get('region', 'Unknown')}",
                f"{max_risk.get('emoji', '')} {max_risk.get('risk_level', 'unknown').upper()} "
                f"(<code>{max_risk.get('risk_score', 0):.0f}/100</code>)",
                ""
            ])
        
        # По регионам
        lines.append("<b>📍 РЕГИОНЫ:</b>\n")
        
        for r in region_results:
            # Основная строка
            region_name = r.get('region', 'Unknown')
            risk_score = r.get('risk_score', 0)
            emoji = r.get('emoji', '')
            risk_level = r.get('risk_level', 'unknown')
            
            lines.append(
                f"{emoji} <b>{region_name}</b> "
                f"<code>{risk_score:.0f}/100</code>"
            )

            
            # После блока "По регионам", перед итоговой статистикой:

            # Добавляем блок с аномалиями (если есть)
            anomaly_results = [r for r in results if r.get('type') == 'anomaly']
            if anomaly_results:
                lines.append("🧠 <b>ОБНАРУЖЕННЫЕ АНОМАЛИИ:</b>")
                for anomaly in anomaly_results:
                    region = anomaly.get('region', 'Unknown')
                    count = anomaly.get('anomaly_count', 0)
                    if count > 0:
                        lines.append(f"  🔴 {region}: {count} аномалий")
                    else:
                        lines.append(f"  ✅ {region}: аномалий не обнаружено")
                lines.append("")
            
            # Детали — безопасное получение stats
            stats = r.get('stats', {})
            if not stats:
                # Если stats нет, используем другие доступные поля
                stats = {
                    'max_mag_7d': r.get('max_mag_7d', 0),
                    'events_24h': r.get('events_24h', 0),
                    'energy_7d': r.get('energy_7d', 0)
                }
            
            # Детали — улучшенные
            details = []
            stats = r.get('stats', {})
            
            if stats.get('max_mag_7d', 0) > 0:
                details.append(f"  Mmax7д={stats['max_mag_7d']:.1f}")
            if stats.get('events_24h', 0) > 0:
                details.append(f"  N24ч={stats['events_24h']}")
            if stats.get('energy_anomaly', False):
                details.append(f"  ⚡ Аномалия энергии!")
            if stats.get('swarm_detected', False):
                details.append(f"  ⚡ РОЙ!")
            if stats.get('trend_detected', False):
                details.append(f"  📈 Тренд↑")
            if stats.get('silence_anomaly', False):
                details.append(f"  🔇 Тишина")
            
            # Факторы риска
            risk_factors = r.get('risk_factors', '')
            if risk_factors and risk_factors != 'Нет явных факторов':
                details.append(f"  📌 {risk_factors}")
            
            # Рекомендация
            recommendation = r.get('recommendation', '')
            if recommendation:
                details.append(f"  💡 {recommendation}")
            
            # Значимые события
            significant_events = r.get('significant_events', [])
            if significant_events:
                lines.append("  🎯 Последние:")
                for evt in significant_events[-2:]:  # Последние 2
                    lines.append(
                        f"    {evt.get('mag_formatted', '')} | {evt.get('time_ago', '')} | "
                        f"{evt.get('place', '')[:35]}"
                    )
            
            lines.append("")  # Пустая строка
        
        # Итоговая статистика
        critical = sum(1 for r in region_results if r.get('risk_level') == 'critical')
        high = sum(1 for r in region_results if r.get('risk_level') == 'high')
        moderate = sum(1 for r in region_results if r.get('risk_level') == 'moderate')
        low = len(region_results) - critical - high - moderate
        
        lines.extend([
            f"{'─' * 38}",
            f"📊 Итого: 🔴×{critical} 🟠×{high} 🟡×{moderate} 🟢×{low}",
            f"⏰ Следующий отчёт: через 12 часов",
            f"🌐 Источник: USGS | Метод: LAIC"
        ])
        
        return '\n'.join(lines)
    
    def _format_alert(self, result):
        """Форматирует срочное оповещение."""
        
        # Безопасное получение stats
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
            f"{result.get('emoji', '')} <b>{result.get('region', 'Unknown')}</b>",
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
        
        # Обрезаем если слишком длинное
        if len(message) > 4000:
            message = message[:3950] + "\n\n... (сообщение обрезано)"
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
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


# Синхронные обёртки
def send_sync(results):
    sender = TelegramSender()
    return asyncio.run(sender.send_report(results))


def alert_sync(result):
    sender = TelegramSender()
    return asyncio.run(sender.send_alert(result))
