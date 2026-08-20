#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_swarm.py — тест интеграции Swarm
"""

import os
import logging
from datetime import datetime, timedelta, timezone

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_swarm():
    """Тестирует подключение к Swarm и получение данных."""
    
    logger.info("=" * 60)
    logger.info("🧪 ТЕСТИРОВАНИЕ SWARM COLLECTOR")
    logger.info("=" * 60)
    
    # 1. Проверяем наличие токена
    token = os.environ.get('VIRES_ACCESS_TOKEN')
    if not token:
        logger.error("❌ Токен VIRES_ACCESS_TOKEN не найден в окружении")
        logger.info("   Установите: export VIRES_ACCESS_TOKEN='ваш_токен'")
        return False
    
    logger.info(f"✅ Токен найден (длина: {len(token)})")
    
    # 2. Импортируем и инициализируем коллектор
    try:
        from swarm_collector import SwarmCollector
        logger.info("✅ SwarmCollector импортирован")
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта SwarmCollector: {e}")
        return False
    
    collector = SwarmCollector()
    status = collector.get_status()
    logger.info(f"📊 Статус: {status}")
    
    if not status['request_ready']:
        logger.error("❌ Соединение с Swarm не установлено")
        return False
    
    # 3. Тестовый запрос
    # Берём событие за последние 3 дня (чтобы были данные)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=3)
    lat, lon = 35.0, 140.0  # Япония
    
    logger.info(f"📅 Запрос данных: {start_time.date()} — {end_time.date()}")
    logger.info(f"📍 Координаты: ({lat}, {lon})")
    
    try:
        # Магнитные данные (FAST)
        logger.info("🛰️ Запрос магнитных данных (FAST)...")
        mag_df = collector.fetch_magnetic_data(lat, lon, start_time, end_time, use_fast=True)
        
        if mag_df.empty:
            logger.warning("⚠️ Магнитные данные не получены (FAST)")
            logger.info("🔄 Пробуем OPER...")
            mag_df = collector.fetch_magnetic_data(lat, lon, start_time, end_time, use_fast=False)
        
        if not mag_df.empty:
            logger.info(f"✅ Получено {len(mag_df)} записей магнитных данных")
            logger.info(f"   Колонки: {mag_df.columns.tolist()[:10]}...")
            
            # Проверяем аномалии
            anomaly = collector.detect_anomaly(mag_df)
            logger.info(f"   Аномалия: {anomaly['has_anomaly']}")
            if anomaly['has_anomaly']:
                logger.info(f"   Z-max: {anomaly['max_z_score']:.2f}σ")
                logger.info(f"   Кол-во аномалий: {anomaly['anomaly_count']}")
        else:
            logger.warning("⚠️ Магнитные данные не получены")
        
        # TEC данные
        logger.info("🛰️ Запрос TEC данных...")
        tec_df = collector.fetch_tec_data(lat, lon, start_time, end_time, use_fast=True)
        
        if tec_df.empty:
            logger.warning("⚠️ TEC данные не получены (FAST)")
            logger.info("🔄 Пробуем OPER...")
            tec_df = collector.fetch_tec_data(lat, lon, start_time, end_time, use_fast=False)
        
        if not tec_df.empty:
            logger.info(f"✅ Получено {len(tec_df)} записей TEC данных")
        else:
            logger.warning("⚠️ TEC данные не получены")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запросе данных: {e}")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ Тест Swarm завершён")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = test_swarm()
    exit(0 if success else 1)
