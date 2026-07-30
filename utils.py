#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — вспомогательные функции
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def ensure_dir(path):
    """Создаёт папку если нет."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_cache_path(key, cache_dir='data/cache'):
    """Возвращает путь к файлу кэша."""
    ensure_dir(cache_dir)
    # Хэш ключа → имя файла
    filename = hashlib.md5(key.encode()).hexdigest() + '.json'
    return os.path.join(cache_dir, filename)


def load_cache(key, max_age_hours=6, cache_dir='data/cache'):
    """
    Загружает кэш если он свежий.
    Возвращает данные или None.
    """
    path = get_cache_path(key, cache_dir)
    
    if not os.path.exists(path):
        return None
    
    # Проверяем возраст
    mtime = os.path.getmtime(path)
    age_hours = (datetime.now().timestamp() - mtime) / 3600
    
    if age_hours > max_age_hours:
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_cache(key, data, cache_dir='data/cache'):
    """Сохраняет данные в кэш."""
    path = get_cache_path(key, cache_dir)
    ensure_dir(cache_dir)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def format_magnitude(mag):
    """Форматирует магнитуду с цветовым индикатором."""
    if mag >= 7.0:
        return f"🔴 M{mag:.1f}"
    elif mag >= 6.0:
        return f"🟠 M{mag:.1f}"
    elif mag >= 5.0:
        return f"🟡 M{mag:.1f}"
    else:
        return f"🟢 M{mag:.1f}"


def format_depth(depth_km):
    """Форматирует глубину."""
    if depth_km < 10:
        return f"⚠️ {depth_km:.1f}км (очень мелко!)"
    elif depth_km < 35:
        return f"🔺 {depth_km:.1f}км (мелко)"
    elif depth_km < 70:
        return f"⬜ {depth_km:.1f}км (средне)"
    else:
        return f"🔷 {depth_km:.1f}км (глубоко)"


def seismic_energy(magnitude):
    """
    Расчёт сейсмической энергии в джоулях.
    Формула: log10(E) = 1.5*M + 4.8
    """
    return 10 ** (1.5 * magnitude + 4.8)


def energy_in_kilotons(magnitude):
    """Энергия в килотоннах TNT (1 кт = 4.184e12 Дж)."""
    joules = seismic_energy(magnitude)
    return joules / 4.184e12


def b_value_calculation(magnitudes):
    """
    Расчёт b-значения (отношение малых/больших землетрясений).
    b < 1: преобладают крупные (напряжение накапливается)
    b > 1: преобладают малые
    """
    if len(magnitudes) < 10:
        return None
    
    import numpy as np
    from scipy import stats
    
    # Метод максимального правдоподобия
    mags = np.array(magnitudes)
    m_min = mags.min()
    
    # b = 1 / (mean(M - Mmin) * ln(10))
    mean_diff = np.mean(mags - m_min)
    if mean_diff > 0:
        b_value = 1.0 / (mean_diff * np.log(10))
        return b_value
    
    return None
  
