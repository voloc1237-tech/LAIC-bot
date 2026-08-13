# fault_zones.py — расширенная версия
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging
import os
import requests

logger = logging.getLogger(__name__)

class FaultZoneAnalyzer:
    def __init__(self, geojson_path="data/gem_active_faults.geojson", cache_dir="data/cache"):
        self.geojson_path = geojson_path
        self.cache_dir = cache_dir
        self.gdf = None
        self.history_cache = {}
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.geojson_path):
            logger.warning(f"Файл {self.geojson_path} не найден.")
            self.gdf = None
            return
        try:
            self.gdf = gpd.read_file(self.geojson_path)
            if self.gdf.crs is None or self.gdf.crs.to_epsg() != 4326:
                self.gdf = self.gdf.to_crs(epsg=4326)
            logger.info(f"✅ Загружено {len(self.gdf)} объектов разломов")
        except Exception as e:
            logger.error(f"Ошибка загрузки разломов: {e}")
            self.gdf = None

    def get_nearest_zone(self, lat, lon, max_distance_km=150):
        """Возвращает ID зоны и расстояние до ближайшего разлома."""
        if self.gdf is None:
            return None, None
        point = Point(lon, lat)
        min_dist_km = float('inf')
        best_zone = None
        for idx, row in self.gdf.iterrows():
            dist_deg = row.geometry.distance(point)
            dist_km = dist_deg * 111.0  # приблизительно
            if dist_km < min_dist_km:
                min_dist_km = dist_km
                # Используем поле 'id' или 'zone_name'
                # Берём название разлома, если есть
                zone_name = row.get('name')
                if not zone_name:
                    # Если названия нет, используем ID
                    zone_name = row.get('id', f'zone_{idx}')
                best_zone = zone_name
                
        if min_dist_km <= max_distance_km:
            return best_zone, min_dist_km
        return None, None

    def get_zone_history(self, zone_id, lat, lon, years_back=10):
        """
        Загружает исторические события USGS для зоны за указанное количество лет.
        Возвращает DataFrame с событиями.
        """
        cache_key = f"history_{zone_id}_{years_back}"
        if cache_key in self.history_cache:
            return self.history_cache[cache_key]

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365 * years_back)

        # Пытаемся загрузить из кэша (файл)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['time'])
            self.history_cache[cache_key] = df
            logger.info(f"📂 История зоны {zone_id} загружена из кэша ({len(df)} событий)")
            return df

        # Если кэша нет — запрашиваем USGS
        # Используем bounding box вокруг точки (радиус ~100 км)
        # Приблизительно: 1° ~ 111 км, значит 1° ~ 100 км
        lat_min = lat - 0.9
        lat_max = lat + 0.9
        lon_min = lon - 0.9
        lon_max = lon + 0.9

        params = {
            'format': 'geojson',
            'starttime': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'minlatitude': lat_min,
            'maxlatitude': lat_max,
            'minlongitude': lon_min,
            'maxlongitude': lon_max,
            'minmagnitude': 4.0,
            'orderby': 'time',
            'limit': 20000
        }
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            features = data.get('features', [])
            records = []
            for f in features:
                props = f['properties']
                coords = f['geometry']['coordinates']
                records.append({
                    'time': pd.to_datetime(props['time'], unit='ms', utc=True),
                    'mag': props['mag'],
                    'place': props.get('place', ''),
                    'lat': coords[1],
                    'lon': coords[0]
                })
            df = pd.DataFrame(records)
            if not df.empty:
                df.sort_values('time', inplace=True)
                # Сохраняем в кэш
                os.makedirs(self.cache_dir, exist_ok=True)
                df.to_csv(cache_file, index=False)
            else:
                df = pd.DataFrame()
            self.history_cache[cache_key] = df
            logger.info(f"📥 История зоны {zone_id} загружена из USGS ({len(df)} событий)")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки истории для зоны {zone_id}: {e}")
            return pd.DataFrame()

    def calculate_zone_risk(self, zone_id, lat, lon, current_anomalies):
        """
        Рассчитывает итоговый риск для зоны на основе:
        - текущих аномалий (количество, магнитуда, Z-score)
        - исторической активности за 10 лет
        - тектонического фона (упрощённо: частота событий за 10 лет)
        """
        # 1. Базовый риск от текущих аномалий
        anomaly_count = len(current_anomalies)
        max_mag = max([a['mag'] for a in current_anomalies], default=0)
        max_z = max([a['z_score'] for a in current_anomalies], default=0)

        # 2. Исторический фон
        hist_df = self.get_zone_history(zone_id, lat, lon, years_back=10)
        hist_count = len(hist_df)
        hist_max_mag = hist_df['mag'].max() if not hist_df.empty else 0

        # 3. Вычисляем риск
        risk_score = 0

        # Текущие аномалии
        if anomaly_count >= 3 or max_z > 3.5:
            risk_score += 40
        elif anomaly_count >= 2 or max_z > 2.5:
            risk_score += 25
        elif anomaly_count >= 1 or max_z > 2.0:
            risk_score += 10

        # Историческая активность (если были сильные события M>6 за 10 лет)
        if hist_max_mag >= 6.0:
            risk_score += 20
        elif hist_max_mag >= 5.5:
            risk_score += 10

        # Частота событий (тектоническая активность)
        if hist_count > 50:  # много событий -> активная зона
            risk_score += 10
        elif hist_count > 20:
            risk_score += 5

        # Определяем уровень
        if risk_score >= 60:
            level = 'critical'
            emoji = '🔴'
            forecast = 'Высокая вероятность сильного события (M≥6.0) в ближайшие 3-5 дней'
        elif risk_score >= 40:
            level = 'high'
            emoji = '🟠'
            forecast = 'Повышенный риск, рекомендуется усилить мониторинг'
        elif risk_score >= 20:
            level = 'moderate'
            emoji = '🟡'
            forecast = 'Умеренный риск, следите за обновлениями'
        else:
            level = 'low'
            emoji = '🟢'
            forecast = 'Риск низкий'

        return {
            'level': level,
            'emoji': emoji,
            'risk_score': risk_score,
            'forecast': forecast,
            'anomaly_count': anomaly_count,
            'max_mag': max_mag,
            'max_z': max_z,
            'hist_count': hist_count,
            'hist_max_mag': hist_max_mag
        }
