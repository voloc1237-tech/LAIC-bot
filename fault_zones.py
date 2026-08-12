# fault_zones.py
import geopandas as gpd
from shapely.geometry import Point
import logging
import os

logger = logging.getLogger(__name__)

class FaultZoneAnalyzer:
    def __init__(self, geojson_path="data/gem_active_faults.geojson"):
        self.geojson_path = geojson_path
        self.gdf = None
        self.load_data()

    def load_data(self):
        """Загружает GeoJSON с разломами."""
        if not os.path.exists(self.geojson_path):
            logger.warning(f"Файл {self.geojson_path} не найден. Анализ зон отключён.")
            self.gdf = None
            return
        try:
            self.gdf = gpd.read_file(self.geojson_path)
            # Если данные в географических координатах (EPSG:4326), оставляем как есть
            if self.gdf.crs is None or self.gdf.crs.to_epsg() != 4326:
                self.gdf = self.gdf.to_crs(epsg=4326)
            logger.info(f"✅ Загружено {len(self.gdf)} объектов разломов")
        except Exception as e:
            logger.error(f"Ошибка загрузки разломов: {e}")
            self.gdf = None

    def get_nearest_zone(self, lat, lon, max_distance_km=100):
        """
        Возвращает название зоны (или ID) и расстояние до ближайшего разлома.
        Если разлом не найден в радиусе max_distance_km, возвращает None.
        """
        if self.gdf is None:
            return None, None

        point = Point(lon, lat)
        # Вычисляем расстояние до каждого разлома (в градусах)
        # Для более точного расчёта в километрах нужно перепроецировать в метрическую систему
        # Используем приближение: 1 градус ~ 111 км (на экваторе), но для точности лучше использовать pyproj.
        # Упрощённо: вычисляем минимальное расстояние в градусах и переводим в км.
        # Более точно: используем геодезическое расстояние.
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        min_dist_km = float('inf')
        best_zone = None

        # Для каждого разлома (линия или полигон) вычисляем расстояние до точки
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            # Расстояние до геометрии (в градусах, примерно)
            # Более точно: вычисляем длину кратчайшей линии между точкой и геометрией
            # shapely даёт расстояние в единицах CRS (градусы для EPSG:4326)
            dist_deg = geom.distance(point)
            # Переводим градусы в километры (приблизительно, на экваторе 1° ~ 111 км)
            dist_km = dist_deg * 111.0  # грубо, для средних широт
            if dist_km < min_dist_km:
                min_dist_km = dist_km
                best_zone = row.get('zone_name', row.get('id', f'zone_{idx}'))  # подберите поле с названием

        if min_dist_km <= max_distance_km:
            return best_zone, min_dist_km
        return None, None
