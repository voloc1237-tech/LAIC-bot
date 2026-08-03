#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laic_analyzer.py — LAIC-анализ сейсмической активности
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class LAICAnalyzer:
    """LAIC-анализ данных."""
    
    def __init__(self, settings):
        self.settings = settings
        self.risk_levels = settings.get('analysis', {}).get('risk_levels', {
            'low': 25,
            'moderate': 45,
            'high': 65,
            'critical': 80
        })
    
    def analyze_region(self, df, region_name, config):
        """Анализирует один регион."""
        if df.empty:
            return {
                'region': region_name,
                'risk_score': 0,
                'risk_level': 'low',
                'emoji': '🟢',
                'stats': {'max_mag_7d': 0, 'events_24h': 0},
                'significant_events': [],
                'alert': False
            }
        
        now = pd.Timestamp.now(tz='UTC')
        
        # Данные за последние 7 дней
        week_ago = now - timedelta(days=7)
        df_week = df[df['time'] >= week_ago]
        
        # Данные за последние 24 часа
        day_ago = now - timedelta(days=1)
        df_day = df[df['time'] >= day_ago]
        
        # Статистика
        max_mag_7d = df_week['magnitude'].max() if not df_week.empty else 0
        events_24h = len(df_day)
        total_events = len(df)
        
        # Расчёт риска (упрощённая версия)
        risk_score = 0
        
        # 1. Магнитуда (макс за 7 дней)
        if max_mag_7d >= 6.5:
            risk_score += 40
        elif max_mag_7d >= 6.0:
            risk_score += 30
        elif max_mag_7d >= 5.5:
            risk_score += 20
        elif max_mag_7d >= 5.0:
            risk_score += 10
        
        # 2. Количество событий за 24 часа
        if events_24h >= 5:
            risk_score += 20
        elif events_24h >= 3:
            risk_score += 10
        elif events_24h >= 1:
            risk_score += 5
        
        # 3. Общее количество событий
        if total_events > 50:
            risk_score += 15
        elif total_events > 30:
            risk_score += 10
        elif total_events > 10:
            risk_score += 5
        
        # Определяем уровень риска
        if risk_score >= self.risk_levels['critical']:
            risk_level = 'critical'
            emoji = '🔴'
            alert = True
        elif risk_score >= self.risk_levels['high']:
            risk_level = 'high'
            emoji = '🟠'
            alert = True
        elif risk_score >= self.risk_levels['moderate']:
            risk_level = 'moderate'
            emoji = '🟡'
            alert = False
        else:
            risk_level = 'low'
            emoji = '🟢'
            alert = False
        
        # Последние события
        significant_events = []
        for _, row in df_week.head(5).iterrows():
            significant_events.append({
                'time': row['time'].strftime('%Y-%m-%d %H:%M'),
                'place': row.get('place', 'Unknown'),
                'mag': row['magnitude'],
                'mag_formatted': f"M{row['magnitude']:.1f}",
                'time_ago': f"{int((now - row['time']).total_seconds() / 3600)}ч"
            })
        
        return {
            'region': region_name,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'emoji': emoji,
            'alert': alert,
            'stats': {
                'max_mag_7d': max_mag_7d,
                'events_24h': events_24h,
                'total_events': total_events,
                'energy_7d': sum(10**(1.5 * row['magnitude']) for _, row in df_week.iterrows()) if not df_week.empty else 0
            },
            'significant_events': significant_events,
            'magnitude_trend': {'trend_detected': False},  # упрощённо
            'swarm': {'is_swarm': events_24h > 10}
        }
    
    def analyze_all(self, data):
        """Анализирует все регионы."""
        results = []
        
        for region_name, df in data.items():
            config = self.settings.get('regions', {}).get(region_name, {})
            result = self.analyze_region(df, region_name, config)
            results.append(result)
        
        logger.info(f"✅ LAIC-анализ завершён: {len(results)} регионов")
        return results
