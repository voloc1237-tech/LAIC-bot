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
        
        # Максимальный риск
        max_risk = max(results, key=lambda x: x['risk_score'])
        
        if max_risk['risk_score'] > 0:
            lines.extend([
                f"<b>🔥 Наибольший риск:</b> {max_risk['region']}",
                f"{max_risk['emoji']} {max_risk['risk_level'].upper()} "
                f"(<code>{max_risk['risk_score']:.0f}/100</code>)",
                ""
            ])
        
        # По регионам
        lines.append("<b>📍 РЕГИОНЫ:</b>\n")
        
        for r in results:
            # Основная строка
            lines.append(
                f"{r['emoji']} <b>{r['region']}</b> "
                f"<code>{r['risk_score']:.0f}/100</code>"
            )
            
            # Детали
            stats = r['stats']
            details = []
            
            if stats['max_mag_7d'] > 0:
                details.append(f"  Mmax7д={stats['max_mag_7d']:.1f}")
            if stats['events_24h'] > 0:
                details.append(f"  N24ч={stats['events_24h']}")
            if r['swarm']['is_swarm']:
                details.append(f"  ⚡ РОЙ!")
            if r['magnitude_trend'].get('trend_detected'):
                details.append(f"  📈 тренд↑")
            
            if details:
                lines.append(' |'.join(details))
            
            # Значимые события
            if r['significant_events']:
                lines.append("  🎯 Последние:")
                for evt in r['significant_events'][-2:]:  # Последние 2
                    lines.append(
                        f"    {evt['mag_formatted']} | {evt['time_ago']} | "
                        f"{evt['place'][:35]}"
                    )
            
            lines.append("")  # Пустая строка
        
        # Итоговая статистика
        critical = sum(1 for r in results if r['risk_level'] == 'critical')
        high = sum(1 for r in results if r['risk_level'] == 'high')
        moderate = sum(1 for r in results if r['risk_level'] == 'moderate')
        
        lines.extend([
            f"{'─' * 38}",
            f"📊 Итого: 🔴×{critical} 🟠×{high} 🟡×{moderate} 🟢×{len(results)-critical-high-moderate}",
            f"⏰ Следующий отчёт: через 12 часов",
            f"🌐 Источник: USGS | Метод: LAIC"
        ])
        
        return '\n'.join(lines)
    
    def _format_alert(self, result):
        """Форматирует срочное оповещение."""
        
        lines = [
            f"🚨 <b>ВНИМАНИЕ! ВЫСОКИЙ РИСК ЗЕМЛЕТРЯСЕНИЯ</b>",
            f"{'─' * 35}",
            f"{result['emoji']} <b>{result['region']}</b>",
            f"",
            f"Уровень: <b>{result['risk_level'].upper()}</b>",
            f"Индекс риска: <code>{result['risk_score']:.0f}/100</code>",
            f"",
            f"📊 Данные:",
            f"  • Макс M за 7д: {result['stats']['max_mag_7d']:.1f}",
            f"  • Событий за 24ч: {result['stats']['events_24h']}",
            f"  • Энергия за 7д: {result['energy']['7d']['total_energy_kt']:.2f} кт TNT",
        ]
        
        if result['swarm']['is_swarm']:
            lines.append(f"  • ⚡ Обнаружен РОЙ землетрясений!")
        
        if result['magnitude_trend'].get('trend_detected'):
            lines.append(f"  • 📈 Нарастание магнитуд!")
        
        lines.extend([
            f"",
            f"💡 {result['recommendation'][:150]}...",
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
            logger.info(f"🚨 АЛЕРТ отправлен: {result['region']}")
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
      
