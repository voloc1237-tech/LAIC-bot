#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — ГЛАВНЫЙ ФАЙЛ LAIC-бота

Собирает → Анализирует → Отправляет
"""

import os
import sys
import logging
from datetime import datetime, timezone

# Логи в stdout (GitHub Actions показывает)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('LAIC')


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
    
    # Загрузка настроек
    import yaml
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    logger.info(f"✅ Настройки загружены: {len(settings['regions'])} регионов")
    
    # ═══════════════════════════════════════════════════════════════
    # 1. СБОР ДАННЫХ
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 70)
    logger.info("📥 ЭТАП 1: Сбор данных USGS")
    logger.info("=" * 70)
    
    from data_collector import collect_all_regions
    data = collect_all_regions(settings)
    
    total = sum(len(df) for df in data.values())
    logger.info(f"\n📊 Всего событий: {total}")
    
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
    # 3. ОТПРАВКА
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 70)
    logger.info("📤 ЭТАП 3: Отправка в Telegram")
    logger.info("=" * 70)
    
    from telegram_sender import send_sync, alert_sync
    
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
  
