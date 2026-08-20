#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_swarm.py — тест с явной передачей токена
"""

import os
import logging
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_swarm():
    logger.info("=" * 60)
    logger.info("🧪 ТЕСТИРОВАНИЕ SWARM")
    logger.info("=" * 60)
    
    token = os.environ.get('VIRES_ACCESS_TOKEN')
    if not token:
        logger.error("❌ Токен не найден")
        return False
    
    logger.info(f"✅ Токен найден (длина: {len(token)})")
    
    try:
        from swarm_collector import SwarmCollector
        collector = SwarmCollector()
        
        # Проверяем статус
        status = collector.get_status()
        logger.info(f"📊 Статус: {status}")
        
        if not status['available']:
            logger.error("❌ viresclient не доступен")
            return False
        
        if not status['token_set']:
            logger.error("❌ Токен не установлен")
            return False
        
        # Тестовый запрос
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)
        lat, lon = 35.0, 140.0  # Япония
        
        logger.info(f"📅 Запрос: {start.date()} — {end.date()}")
        logger.info(f"📍 Координаты: ({lat}, {lon})")
        
        # Пробуем получить магнитные данные
        mag_df = collector.fetch_magnetic_data(lat, lon, start, end, spacecraft='A', use_fast=False)
        
        if not mag_df.empty:
            logger.info(f"✅ Получено {len(mag_df)} записей")
            anomaly = collector.detect_anomaly(mag_df)
            logger.info(f"   Аномалия: {anomaly['has_anomaly']}")
            if anomaly['has_anomaly']:
                logger.info(f"   Z-max: {anomaly['max_z_score']:.2f}σ")
        else:
            logger.warning("⚠️ Данные не получены")
            
            # Пробуем B и C
            for sc in ['B', 'C']:
                df = collector.fetch_magnetic_data(lat, lon, start, end, spacecraft=sc, use_fast=False)
                if not df.empty:
                    logger.info(f"✅ Спутник {sc}: {len(df)} записей")
                    break
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_swarm()
    exit(0 if success else 1)
