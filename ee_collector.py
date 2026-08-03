"""
ee_collector.py — Google Earth Engine инициализация и сбор данных

Использует Service Account (через base64 ключ) или OAuth2 токен.
"""

import ee
import os
import base64
import tempfile
import json
import logging

logger = logging.getLogger(__name__)


class GEEInitializer:
    """Универсальная инициализация Google Earth Engine."""
    
    @staticmethod
    def init():
        """
        Пробует все способы инициализации GEE.
        Возвращает True если успешно, иначе False.
        """
        
        # ═══════════════════════════════════════════════
        # Способ 1: Уже аутентифицировано (локально)
        # ═══════════════════════════════════════════════
        try:
            ee.Initialize(project=os.environ.get('EARTH_ENGINE_PROJECT'))
            logger.info("✅ EE инициализирован (существующая аутентификация)")
            return True
        except Exception:
            pass
        
        # ═══════════════════════════════════════════════
        # Способ 2: Service Account из файла
        # ═══════════════════════════════════════════════
        key_path = os.environ.get('EARTH_ENGINE_KEY_PATH')
        if key_path and os.path.exists(key_path):
            try:
                credentials = ee.ServiceAccountCredentials('', key_path)
                ee.Initialize(
                    credentials,
                    project=os.environ.get('EARTH_ENGINE_PROJECT')
                )
                logger.info("✅ EE инициализирован (Service Account, файл)")
                return True
            except Exception as e:
                logger.warning(f"Service Account файл не сработал: {e}")
        
        # ═══════════════════════════════════════════════
        # Способ 3: Service Account из Base64
        # ═══════════════════════════════════════════════
        key_b64 = os.environ.get('EARTH_ENGINE_KEY_BASE64')
        if key_b64:
            try:
                key_json = base64.b64decode(key_b64).decode('utf-8')
                
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False
                ) as f:
                    f.write(key_json)
                    temp_path = f.name
                
                try:
                    credentials = ee.ServiceAccountCredentials('', temp_path)
                    ee.Initialize(
                        credentials,
                        project=os.environ.get('EARTH_ENGINE_PROJECT')
                    )
                    logger.info("✅ EE инициализирован (Service Account, base64)")
                    return True
                finally:
                    os.unlink(temp_path)
                    
            except Exception as e:
                logger.warning(f"Service Account base64 не сработал: {e}")
        
        # ═══════════════════════════════════════════════
        # Способ 4: OAuth2 токен
        # ═══════════════════════════════════════════════
        token_json = os.environ.get('EARTHENGINE_TOKEN')
        if token_json:
            try:
                import google.oauth2.credentials
                stored = json.loads(token_json)
                credentials = google.oauth2.credentials.Credentials(
                    None,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=stored["client_id"],
                    client_secret=stored["client_secret"],
                    refresh_token=stored["refresh_token"],
                    quota_project_id=stored.get("project"),
                )
                ee.Initialize(credentials=credentials)
                logger.info("✅ EE инициализирован (OAuth2)")
                return True
            except Exception as e:
                logger.warning(f"OAuth2 не сработал: {e}")
        
        # ═══════════════════════════════════════════════
        # Ничего не сработало
        # ═══════════════════════════════════════════════
        logger.error("❌ Не удалось инициализировать Earth Engine!")
        logger.error("Задайте один из:")
        logger.error("  • EARTH_ENGINE_KEY_BASE64 (рекомендуется)")
        logger.error("  • EARTH_ENGINE_KEY_PATH")
        logger.error("  • EARTHENGINE_TOKEN")
        return False


def get_lst_data(lat: float, lon: float, date_start: str, date_end: str):
    """
    Получить температуру поверхности (LST) из MODIS.
    
    Args:
        lat, lon: координаты
        date_start, date_end: даты в формате 'YYYY-MM-DD'
    
    Returns:
        dict: информация об изображении
    """
    # Инициализируем GEE
    if not GEEInitializer.init():
        raise RuntimeError("Earth Engine не инициализирован!")
    
    # Точка интереса
    point = ee.Geometry.Point([lon, lat])
    
    # Коллекция MODIS LST
    collection = ee.ImageCollection("MODIS/061/MOD11A1") \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .select('LST_Day_1km')
    
    # Берём среднее за период
    mean_lst = collection.mean()
    
    # Получаем значение в точке
    sample = mean_lst.sample(point, 1000).first()
    
    # Переводим в градусы Цельсия (MODIS хранит в Кельвинах × 0.02)
    lst_raw = sample.get('LST_Day_1km').getInfo()
    lst_celsius = (lst_raw * 0.02) - 273.15 if lst_raw else None
    
    return {
        'lst_celsius': round(lst_celsius, 2) if lst_celsius else None,
        'count': collection.size().getInfo(),
        'period': f"{date_start} to {date_end}"
    }


# ═══════════════════════════════════════════════════════
# ТЕСТ (можно запустить локально)
# ═════════════════════════════════════════════════════==

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тест: Токио
    try:
        result = get_lst_data(
            lat=35.6895,
            lon=139.6917,
            date_start='2024-01-01',
            date_end='2024-01-10'
        )
        print(f"✅ Тест пройден! LST: {result['lst_celsius']}°C")
    except Exception as e:
        print(f"❌ Тест не пройден: {e}")
      
