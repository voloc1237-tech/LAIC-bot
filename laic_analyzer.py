#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laic_analyzer.py — улучшенный LAIC-анализ сейсмической активности
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class LAICAnalyzer:
    """
    Улучшенный LAIC-анализ с учётом:
    - Накопленной сейсмической энергии
    - Трендов магнитуд
    - Аномалий в "тишине" (отсутствие фоновых событий)
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.risk_levels = settings.get('analysis', {}).get('risk_levels', {
            'low': 25,
            'moderate': 45,
            'high': 65,
            'critical': 80
        })
    
    def _calculate_energy(self, magnitudes):
        """Расчёт сейсмической энергии в кт TNT (по формуле Гутенберга-Рихтера)."""
        if not magnitudes:
            return 0
        # log10(E) = 1.5*M + 4.8 (энергия в джоулях)
        # Переводим в килотонны TNT (1 кт = 4.184e12 Дж)
        return sum(10**(1.5 * m + 4.8) / 4.184e12 for m in magnitudes if m > 0)
    
    def _detect_swarm(self, df, hours=72):
        """Обнаружение роя землетрясений."""
        if len(df) < 3:
            return False, 0
        
        # Группируем по времени (окна по hours часов)
        df_sorted = df.sort_values('time')
        time_diffs = df_sorted['time'].diff().dt.total_seconds() / 3600
        
        # Если много событий за короткое время
        recent_events = time_diffs[time_diffs <= hours]
        if len(recent_events) >= 3:
            return True, len(recent_events) + 1
        
        return False, 0
    
    def _detect_energy_anomaly(self, df, window_days=14):
        """
        Обнаружение аномалии в накопленной энергии.
        Сравнивает энергию в текущем окне со средним за период.
        """
        if len(df) < 5:
            return False, 0
        
        df_sorted = df.sort_values('time')
        now = df_sorted['time'].max()
        window_start = now - timedelta(days=window_days)
        
        # Энергия в текущем окне
        df_window = df_sorted[df_sorted['time'] >= window_start]
        current_energy = self._calculate_energy(df_window['magnitude'].tolist())
        
        # Средняя энергия за весь период (исключая текущее окно)
        df_rest = df_sorted[df_sorted['time'] < window_start]
        if len(df_rest) < 5:
            return False, 0
        
        avg_energy = self._calculate_energy(df_rest['magnitude'].tolist()) / (len(df_rest) / len(df_window))
        
        # Аномалия, если энергия превышает среднюю более чем в 3 раза
        if avg_energy > 0 and current_energy / avg_energy > 3:
            return True, current_energy / avg_energy
        
        return False, 0
    
    def _detect_magnitude_trend(self, df, min_events=5):
        """Обнаружение тренда магнитуд (простой линейный тренд)."""
        if len(df) < min_events:
            return False, 0
        
        df_sorted = df.sort_values('time').tail(min_events * 2)
        magnitudes = df_sorted['magnitude'].values
        indices = np.arange(len(magnitudes))
        
        # Простая линейная регрессия
        if len(magnitudes) >= 3:
            slope = np.polyfit(indices, magnitudes, 1)[0]
            if slope > 0.1:  # Положительный тренд > 0.1 за период
                return True, slope
        
        return False, 0
    
    def _detect_silence_anomaly(self, df, lookback_days=30, silence_threshold_hours=72):
        """
        Обнаружение аномалии в "тишине" — отсутствие событий определённой магнитуды.
        """
        if len(df) < 10:
            return False, 0
        
        now = df['time'].max()
        lookback_start = now - timedelta(days=lookback_days)
        df_lookback = df[df['time'] >= lookback_start]
        
        if len(df_lookback) < 5:
            return False, 0
        
        # Ищем события с магнитудой > 5.0
        strong_events = df_lookback[df_lookback['magnitude'] >= 5.0]
        if len(strong_events) < 2:
            return False, 0
        
        # Проверяем, было ли событие в последние silence_threshold_hours часов
        last_strong = strong_events['time'].max()
        hours_since = (now - last_strong).total_seconds() / 3600
        
        if hours_since > silence_threshold_hours:
            return True, hours_since
        
        return False, 0
    
    def analyze_region(self, df, region_name, config):
        """Улучшенный анализ региона."""
        if df.empty:
            return {
                'region': region_name,
                'risk_score': 0,
                'risk_level': 'low',
                'emoji': '🟢',
                'stats': {'max_mag_7d': 0, 'events_24h': 0},
                'significant_events': [],
                'alert': False,
                'recommendation': 'Данных недостаточно'
            }
        
        now = pd.Timestamp.now(tz='UTC')
        
        # Данные за последние 7 дней
        week_ago = now - timedelta(days=7)
        df_week = df[df['time'] >= week_ago]
        
        # Данные за последние 24 часа
        day_ago = now - timedelta(days=1)
        df_day = df[df['time'] >= day_ago]
        
        # Данные за последние 30 дней (для трендов)
        month_ago = now - timedelta(days=30)
        df_month = df[df['time'] >= month_ago]
        
        # Статистика
        max_mag_7d = df_week['magnitude'].max() if not df_week.empty else 0
        events_24h = len(df_day)
        total_events = len(df)
        
        # Расчёт риска (улучшенная версия)
        risk_score = 0
        risk_factors = []
        
        # 1. Магнитуда (макс за 7 дней)
        if max_mag_7d >= 6.5:
            risk_score += 30
            risk_factors.append(f"Сильная магнитуда ({max_mag_7d:.1f})")
        elif max_mag_7d >= 6.0:
            risk_score += 20
            risk_factors.append(f"Магнитуда {max_mag_7d:.1f}")
        elif max_mag_7d >= 5.5:
            risk_score += 10
        
        # 2. Количество событий за 24 часа (рой)
        swarm_detected, swarm_count = self._detect_swarm(df_day, hours=24)
        if swarm_detected:
            risk_score += 25
            risk_factors.append(f"Рой из {swarm_count} событий за 24ч")
        elif events_24h >= 3:
            risk_score += 10
            risk_factors.append(f"{events_24h} событий за 24ч")
        elif events_24h >= 1:
            risk_score += 5
        
        # 3. Аномалия в накопленной энергии
        energy_anomaly, energy_ratio = self._detect_energy_anomaly(df, window_days=14)
        if energy_anomaly:
            risk_score += 15
            risk_factors.append(f"Аномалия энергии (×{energy_ratio:.1f})")
        
        # 4. Тренд магнитуд
        trend_detected, trend_slope = self._detect_magnitude_trend(df_month, min_events=5)
        if trend_detected:
            risk_score += 10
            risk_factors.append(f"Тренд магнитуд (+{trend_slope:.2f})")
        
        # 5. Аномалия "тишины"
        silence_anomaly, hours_since = self._detect_silence_anomaly(df, lookback_days=30)
        if silence_anomaly:
            risk_score += 10
            risk_factors.append(f"Тишина ({hours_since:.0f}ч без M≥5.0)")
        
        # Определяем уровень риска
        if risk_score >= self.risk_levels['critical']:
            risk_level = 'critical'
            emoji = '🔴'
            alert = True
            recommendation = 'ВЫСОКИЙ РИСК! Возможно сильное землетрясение в ближайшие 1-3 дня.'
        elif risk_score >= self.risk_levels['high']:
            risk_level = 'high'
            emoji = '🟠'
            alert = True
            recommendation = 'Повышенный риск. Рекомендуется усилить мониторинг.'
        elif risk_score >= self.risk_levels['moderate']:
            risk_level = 'moderate'
            emoji = '🟡'
            alert = False
            recommendation = 'Умеренный риск. Следите за обновлениями.'
        else:
            risk_level = 'low'
            emoji = '🟢'
            alert = False
            recommendation = 'Риск низкий. Мониторинг продолжается.'
        
        # Значимые события
        significant_events = []
        for _, row in df_week.head(5).iterrows():
            significant_events.append({
                'time': row['time'].strftime('%Y-%m-%d %H:%M'),
                'place': row.get('place', 'Unknown'),
                'mag': row['magnitude'],
                'mag_formatted': f"M{row['magnitude']:.1f}",
                'time_ago': f"{int((now - row['time']).total_seconds() / 3600)}ч"
            })
        
        # Вычисляем энергию за 7 дней
        energy_7d = self._calculate_energy(df_week['magnitude'].tolist()) if not df_week.empty else 0
        
        # Формируем список факторов риска для отчёта
        risk_factors_str = ', '.join(risk_factors) if risk_factors else 'Нет явных факторов'
        
        return {
            'region': region_name,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'emoji': emoji,
            'alert': alert,
            'recommendation': recommendation,
            'risk_factors': risk_factors_str,
            'stats': {
                'max_mag_7d': max_mag_7d,
                'events_24h': events_24h,
                'total_events': total_events,
                'energy_7d': energy_7d,
                'energy_anomaly': energy_anomaly,
                'swarm_detected': swarm_detected,
                'trend_detected': trend_detected,
                'silence_anomaly': silence_anomaly
            },
            'significant_events': significant_events,
            'magnitude_trend': {'trend_detected': trend_detected, 'slope': trend_slope if trend_detected else 0},
            'swarm': {'is_swarm': swarm_detected, 'count': swarm_count}
        }
    
    
    def analyze_all(self, data):
        """Анализирует все регионы."""
        results = []
        
        for region_name, df in data.items():
            # Проверяем, что df — DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"⚠️ {region_name}: данные не являются DataFrame, пропускаем")
                continue
            
            config = self.settings.get('regions', {}).get(region_name, {})
            result = self.analyze_region(df, region_name, config)
            results.append(result)
        
        logger.info(f"✅ LAIC-анализ завершён: {len(results)} регионов")
        return results
   
