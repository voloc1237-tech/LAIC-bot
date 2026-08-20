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
        
        if not collector.get_status()['available']:
            logger.error("❌ viresclient не доступен")
            return False
        
        # ===== ТЕСТ 1: ДАННЫЕ ЗА 2024 ГОД (ГАРАНТИРОВАННО ЕСТЬ) =====
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        lat, lon = 35.0, 140.0
        
        logger.info(f"📅 Тест 1: {start.date()} — {end.date()}")
        logger.info(f"📍 Координаты: ({lat}, {lon})")
        
        # Пробуем спутник Alpha (A) за этот период
        mag_df = collector.fetch_magnetic_data(lat, lon, start, end, spacecraft='A', use_fast=False)
        
        if not mag_df.empty:
            logger.info(f"✅ Alpha: получено {len(mag_df)} записей")
            logger.info(f"   Колонки: {mag_df.columns.tolist()}")
            logger.info(f"   Первые 3 записи:\n{mag_df.head(3)}")
            anomaly = collector.detect_anomaly(mag_df)
            logger.info(f"   Аномалия: {anomaly['has_anomaly']}")
            if anomaly['has_anomaly']:
                logger.info(f"   Z-max: {anomaly['max_z_score']:.2f}σ")
        else:
            logger.warning("⚠️ Alpha: данных нет")
            
            # Пробуем Bravo (B)
            mag_df = collector.fetch_magnetic_data(lat, lon, start, end, spacecraft='B', use_fast=False)
            if not mag_df.empty:
                logger.info(f"✅ Bravo: получено {len(mag_df)} записей")
            else:
                logger.warning("⚠️ Bravo: данных нет")
                
                # Пробуем Charlie (C)
                mag_df = collector.fetch_magnetic_data(lat, lon, start, end, spacecraft='C', use_fast=False)
                if not mag_df.empty:
                    logger.info(f"✅ Charlie: получено {len(mag_df)} записей")
                else:
                    logger.warning("⚠️ Данных нет ни с одного спутника за этот период")
                    logger.info("   Попробуйте другой период или координаты")
                    return False
        
        # ===== ТЕСТ 2: ДАННЫЕ ЗА ПОСЛЕДНИЕ 3 ДНЯ =====
        logger.info("\n" + "=" * 60)
        logger.info("🔄 Тест 2: Последние 3 дня")
        
        end_now = datetime.now(timezone.utc)
        start_now = end_now - timedelta(days=3)
        
        mag_df_now = collector.fetch_magnetic_data(lat, lon, start_now, end_now, spacecraft='A', use_fast=False)
        if not mag_df_now.empty:
            logger.info(f"✅ Alpha (последние 3 дня): {len(mag_df_now)} записей")
        else:
            logger.info("ℹ️ Alpha (последние 3 дня): данных нет (нет пролётов)")

        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_swarm()
    exit(0 if success else 1)
