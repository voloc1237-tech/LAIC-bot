#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laic_analyzer.py — полноценный анализ LAIC-предвестников
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy import stats
import logging

from utils import (
    seismic_energy, energy_in_kilotons, b_value_calculation,
    format_magnitude, format_depth
)

logger = logging.getLogger(__name__)


class LAICAnalyzer:
    """
    Полноценный анализатор пресейсмических аномалий.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.thresholds = settings['analysis']['risk_levels']
        self.anomaly_params = settings['analysis']['anomaly']
        self.swarm_hours = settings['analysis']['swarm_detection_hours']
    
    # ═══════════════════════════════════════════════════════════════
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════
    
    def _get_time_windows(self, df, now):
        """Создаёт временные окна для анализа."""
        windows = {
            '24h': df[df['time'] > now - timedelta(hours=24)],
            '72h': df[df['time'] > now - timedelta(hours=72)],
            '7d': df[df['time'] > now - timedelta(days=7)],
            '14d': df[df['time'] > now - timedelta(days=14)],
            '30d': df[df['time'] > now - timedelta(days=30)],
        }
        return windows
    
    def _calculate_energy_stats(self, df_subset):
        """Расчёт энергетических статистик."""
        if len(df_subset) == 0:
            return {
                'total_energy_j': 0,
                'total_energy_kt': 0,
                'mean_energy_j': 0,
                'max_energy_j': 0
            }
        
        energies = df_subset['magnitude'].apply(seismic_energy)
        
        return {
            'total_energy_j': float(energies.sum()),
            'total_energy_kt': float(energies.sum() / 4.184e12),
            'mean_energy_j': float(energies.mean()),
            'max_energy_j': float(energies.max()),
            'energy_count': len(energies)
        }
    
    def _detect_swarm(self, df, hours=72):
        """
        Обнаружение землетрясения-роя (swarm).
        Много событий в короткий срок = возможная подготовка большого.
        """
        if len(df) < 3:
            return {
                'is_swarm': False,
                'total_events': len(df),
                'peak_hourly_rate': 0,
                'swarm_score': 0
            }
        
        now = datetime.now(timezone.utc)
        recent = df[df['time'] > now - timedelta(hours=hours)].copy()
        
        if len(recent) < 3:
            return {
                'is_swarm': False,
                'total_events': len(recent),
                'peak_hourly_rate': 0,
                'swarm_score': 0
            }
        
        # Группировка по часам
        recent['hour'] = recent['time'].dt.floor('H')
        hourly = recent.groupby('hour').size()
        
        peak_rate = hourly.max()
        mean_rate = hourly.mean() if len(hourly) > 0 else 0
        
        # Swarm score: насколько пик превышает среднее
        swarm_score = peak_rate / max(mean_rate, 1)
        
        # Критерии роя
        is_swarm = (
            len(recent) >= 10 and peak_rate >= 3
        ) or (
            swarm_score >= 3 and len(recent) >= 5
        )
        
        return {
            'is_swarm': is_swarm,
            'total_events': len(recent),
            'peak_hourly_rate': int(peak_rate),
            'mean_hourly_rate': float(mean_rate),
            'swarm_score': float(swarm_score),
            'hours_analyzed': hours
        }
    
    def _magnitude_trend(self, df, window_days=7):
        """
        Анализ тренда магнитуд (нарастание = плохой знак).
        """
        if len(df) < self.anomaly_params['min_events_for_trend']:
            return {
                'trend_detected': False,
                'slope': 0,
                'p_value': 1.0,
                'r_squared': 0
            }
        
        now = datetime.now(timezone.utc)
        recent = df[df['time'] > now - timedelta(days=window_days)].copy()
        
        if len(recent) < 5:
            return {
                'trend_detected': False,
                'slope': 0,
                'p_value': 1.0,
                'r_squared': 0
            }
        
        # Время в днях от начала окна
        recent['days_from_start'] = (
            recent['time'] - recent['time'].min()
        ).dt.total_seconds() / 86400
        
        # Линейная регрессия
        x = recent['days_from_start'].values
        y = recent['magnitude'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Тренд значим если:
        # - положительный наклон (магнитуды растут)
        # - p-value < 0.05
        # - достаточно точек
        trend_detected = (
            slope > self.anomaly_params['magnitude_acceleration_threshold'] and
            p_value < self.anomaly_params['trend_significance'] and
            len(recent) >= self.anomaly_params['min_events_for_trend']
        )
        
        return {
            'trend_detected': trend_detected,
            'slope': float(slope),
            'p_value': float(p_value),
            'r_squared': float(r_value ** 2),
            'events_count': len(recent),
            'window_days': window_days
        }
    
    def _foreshock_probability(self, df, alert_threshold):
        """
        Оценка вероятности, что событие — форшок (предвестник).
        """
        if len(df) < 2:
            return {'probability': 0, 'indicators': []}
        
        now = datetime.now(timezone.utc)
        last_7d = df[df['time'] > now - timedelta(days=7)]
        
        indicators = []
        
        # Индикатор 1: последнее событие близко к порогу тревоги
        if len(last_7d) > 0:
            max_recent = last_7d['magnitude'].max()
            if max_recent >= alert_threshold - 0.5:
                indicators.append('near_threshold')
        
        # Индикатор 2: учащение событий
        daily_counts = df[df['time'] > now - timedelta(days=14)].copy()
        if len(daily_counts) >= 10:
            daily_counts['date'] = daily_counts['time'].dt.date
            daily = daily_counts.groupby('date').size()
            if len(daily) >= 7:
                first_week = daily.iloc[:7].mean()
                last_week = daily.iloc[-7:].mean()
                if last_week > first_week * 2:
                    indicators.append('acceleration')
        
        # Индикатор 3: мелкие события перед крупным
        shallow_events = df[
            (df['depth_km'] < 20) & 
            (df['magnitude'] >= alert_threshold - 1.5)
        ]
        if len(shallow_events) > 0:
            last_shallow = shallow_events['time'].max()
            if (now - last_shallow).total_seconds() < 86400 * 3:
                indicators.append('shallow_recent')
        
        # Вероятность форшока
        probability = min(len(indicators) / 3 * 0.7, 0.95)
        
        return {
            'probability': probability,
            'indicators': indicators,
            'indicators_count': len(indicators)
        }
    
    def _b_value_analysis(self, df):
        """
        Анализ b-значения (Gutenberg-Richter).
        b < 1: доминируют крупные события = накопление напряжения.
        """
        if len(df) < 20:
            return {'b_value': None, 'reliable': False}
        
        mags = df['magnitude'].dropna()
        b_val = b_value_calculation(mags.tolist())
        
        if b_val is None:
            return {'b_value': None, 'reliable': False}
        
        # Интерпретация
        stress_state = 'unknown'
        if b_val < 0.8:
            stress_state = 'high_stress'  # Опасно!
        elif b_val < 1.0:
            stress_state = 'elevated'
        elif b_val < 1.5:
            stress_state = 'normal'
        else:
            stress_state = 'low_stress'
        
        return {
            'b_value': round(b_val, 3),
            'reliable': len(mags) >= 50,
            'stress_state': stress_state,
            'interpretation': {
                'high_stress': '⚠️ Накопление напряжения (b<0.8)',
                'elevated': '⚡ Повышенное напряжение',
                'normal': '✅ Норма',
                'low_stress': 'ℹ️ Низкое напряжение'
            }[stress_state]
        }
    
    # ═══════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ МЕТОД: ПОЛНЫЙ АНАЛИЗ РЕГИОНА
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_region(self, region_key, region_config, df):
        """
        ПОЛНЫЙ анализ одного региона.
        Возвращает словарь со всеми метриками.
        """
        now = datetime.now(timezone.utc)
        name = region_config['name']
        alert_threshold = region_config.get('alert_threshold', 6.0)
        
        # Базовый результат (если нет данных)
        empty_result = {
            'region': name,
            'region_key': region_key,
            'risk_level': 'low',
            'risk_score': 0,
            'emoji': '🟢',
            'alert': False,
            'summary': f'{name}: нет данных',
            'details': {},
            'recommendation': 'Недостаточно данных для анализа.'
        }
        
        if df is None or df.empty:
            return empty_result
        
        # Временные окна
        windows = self._get_time_windows(df, now)
        
        # ═══════════════════════════════════════════════════════════
        # 1. БАЗОВЫЕ СТАТИСТИКИ
        # ═══════════════════════════════════════════════════════════
        
        stats_basic = {
            'total_events_30d': len(windows['30d']),
            'events_7d': len(windows['7d']),
            'events_24h': len(windows['24h']),
            'events_72h': len(windows['72h']),
            'max_mag_30d': windows['30d']['magnitude'].max() if len(windows['30d']) > 0 else 0,
            'max_mag_7d': windows['7d']['magnitude'].max() if len(windows['7d']) > 0 else 0,
            'max_mag_24h': windows['24h']['magnitude'].max() if len(windows['24h']) > 0 else 0,
            'mean_mag_7d': windows['7d']['magnitude'].mean() if len(windows['7d']) > 0 else 0,
        }
        
        # ═══════════════════════════════════════════════════════════
        # 2. ЭНЕРГЕТИЧЕСКИЙ АНАЛИЗ
        # ═══════════════════════════════════════════════════════════
        
        energy_7d = self._calculate_energy_stats(windows['7d'])
        energy_30d = self._calculate_energy_stats(windows['30d'])
        
        # Доля энергии за последнюю неделю
        energy_ratio = (
            energy_7d['total_energy_j'] / energy_30d['total_energy_j']
            if energy_30d['total_energy_j'] > 0 else 0
        )
        
        # ═══════════════════════════════════════════════════════════
        # 3. ОБНАРУЖЕНИЕ РОЯ (SWARM)
        # ═══════════════════════════════════════════════════════════
        
        swarm = self._detect_swarm(df, hours=self.swarm_hours)
        
        # ═══════════════════════════════════════════════════════════
        # 4. ТРЕНД МАГНИТУД
        # ═══════════════════════════════════════════════════════════
        
        mag_trend = self._magnitude_trend(df, window_days=7)
        
        # ═══════════════════════════════════════════════════════════
        # 5. ВЕРОЯТНОСТЬ ФОРШОКА
        # ═══════════════════════════════════════════════════════════
        
        foreshock = self._foreshock_probability(df, alert_threshold)
        
        # ═══════════════════════════════════════════════════════════
        # 6. B-ЗНАЧЕНИЕ
        # ═══════════════════════════════════════════════════════════
        
        b_analysis = self._b_value_analysis(windows['30d'])
        
        # ═══════════════════════════════════════════════════════════
        # 7. РАСЧЁТ ИНДЕКСА РИСКА (0-100)
        # ═══════════════════════════════════════════════════════════
        
        score = 0.0
        score_breakdown = {}
        
        # --- Фактор 1: Магнитуда за неделю (0-30) ---
        mag_score = 0
        if stats_basic['max_mag_7d'] >= alert_threshold + 1.0:
            mag_score = 30
        elif stats_basic['max_mag_7d'] >= alert_threshold + 0.5:
            mag_score = 25
        elif stats_basic['max_mag_7d'] >= alert_threshold:
            mag_score = 20
        elif stats_basic['max_mag_7d'] >= alert_threshold - 0.5:
            mag_score = 12
        elif stats_basic['max_mag_7d'] >= alert_threshold - 1.0:
            mag_score = 5
        score += mag_score
        score_breakdown['magnitude'] = mag_score
        
        # --- Фактор 2: Магнитуда за сутки (0-20) ---
        mag24_score = 0
        if stats_basic['max_mag_24h'] >= alert_threshold:
            mag24_score = 20
        elif stats_basic['max_mag_24h'] >= alert_threshold - 0.5:
            mag24_score = 12
        elif stats_basic['max_mag_24h'] >= 4.5:
            mag24_score = 4
        score += mag24_score
        score_breakdown['magnitude_24h'] = mag24_score
        
        # --- Фактор 3: Активность (0-18) ---
        activity_score = 0
        if stats_basic['events_24h'] >= 15:
            activity_score = 18
        elif stats_basic['events_24h'] >= 8:
            activity_score = 12
        elif stats_basic['events_24h'] >= 4:
            activity_score = 6
        elif stats_basic['events_24h'] >= 2:
            activity_score = 2
        score += activity_score
        score_breakdown['activity'] = activity_score
        
        # --- Фактор 4: Рой (0-15) ---
        swarm_score = 0
        if swarm['is_swarm']:
            swarm_score = min(15, int(swarm['swarm_score'] * 3))
        score += swarm_score
        score_breakdown['swarm'] = swarm_score
        
        # --- Фактор 5: Тренд магнитуд (0-10) ---
        trend_score = 0
        if mag_trend['trend_detected']:
            trend_score = 10
        score += trend_score
        score_breakdown['trend'] = trend_score
        
        # --- Фактор 6: Энергетический баланс (0-5) ---
        energy_score = 0
        if energy_ratio > 0.6:
            energy_score = 5
        elif energy_ratio > 0.4:
            energy_score = 3
        score += energy_score
        score_breakdown['energy'] = energy_score
        
        # --- Фактор 7: Форшок (0-5) ---
        foreshock_score = 0
        if foreshock['probability'] > 0.5:
            foreshock_score = 5
        elif foreshock['probability'] > 0.3:
            foreshock_score = 3
        score += foreshock_score
        score_breakdown['foreshock'] = foreshock_score
        
        # Ограничиваем
        score = min(score, 100)
        
        # ═══════════════════════════════════════════════════════════
        # 8. ОПРЕДЕЛЕНИЕ УРОВНЯ РИСКА
        # ═══════════════════════════════════════════════════════════
        
        if score >= self.thresholds['critical']:
            level = 'critical'
            emoji = '🔴'
            alert = True
            recommendation = (
                f"🚨 КРИТИЧЕСКИЙ УРОВЕНЬ! Высокая вероятность сильного "
                f"землетрясения M≥{alert_threshold} в ближайшие дни. "
                f"Рекомендуется усиленный мониторинг."
            )
        elif score >= self.thresholds['high']:
            level = 'high'
            emoji = '🟠'
            alert = True
            recommendation = (
                f"⚠️ ВЫСОКИЙ РИСК. Обнаружены значимые предвестниковые "
                f"признаки. Возможно землетрясение M≥{alert_threshold} "
                f"в течение 1-2 недель."
            )
        elif score >= self.thresholds['moderate']:
            level = 'moderate'
            emoji = '🟡'
            alert = False
            recommendation = (
                f"⚡ УМЕРЕННЫЙ РИСК. Повышенная сейсмическая активность. "
                f"Внимательное наблюдение рекомендуется."
            )
        else:
            level = 'low'
            emoji = '🟢'
            alert = False
            recommendation = (
                f"✅ НОРМА. Сейсмическая обстановка в пределах нормы. "
                f"Регулярный мониторинг продолжается."
            )
        
        # ═══════════════════════════════════════════════════════════
        # 9. ЗНАЧИМЫЕ СОБЫТИЯ (для отчёта)
        # ═══════════════════════════════════════════════════════════
        
        significant = df[
            df['magnitude'] >= alert_threshold - 0.5
        ].tail(5)
        
        sig_list = []
        for _, row in significant.iterrows():
            time_ago = now - row['time']
            hours_ago = time_ago.total_seconds() / 3600
            
            if hours_ago < 1:
                time_str = "только что"
            elif hours_ago < 24:
                time_str = f"{int(hours_ago)}ч назад"
            else:
                days = int(hours_ago / 24)
                time_str = f"{days}д назад"
            
            sig_list.append({
                'time': row['time'].strftime('%m-%d %H:%M UTC'),
                'time_ago': time_str,
                'magnitude': row['magnitude'],
                'mag_formatted': format_magnitude(row['magnitude']),
                'place': row['place'][:45],
                'depth_km': row['depth_km'],
                'depth_formatted': format_depth(row['depth_km']),
                'url': row.get('url', '')
            })
        
        # ═══════════════════════════════════════════════════════════
        # 10. СБОРКА РЕЗУЛЬТАТА
        # ═══════════════════════════════════════════════════════════
        
        result = {
            'region': name,
            'region_key': region_key,
            'risk_level': level,
            'risk_score': round(score, 1),
            'emoji': emoji,
            'alert': alert,
            'alert_threshold': alert_threshold,
            
            # Статистики
            'stats': stats_basic,
            
            # Энергия
            'energy': {
                '7d': energy_7d,
                '30d': energy_30d,
                'ratio': round(energy_ratio, 3)
            },
            
            # Рой
            'swarm': swarm,
            
            # Тренд
            'magnitude_trend': mag_trend,
            
            # Форшок
            'foreshock': foreshock,
            
            # B-значение
            'b_value': b_analysis,
            
            # Значимые события
            'significant_events': sig_list,
            
            # Разбор счёта
            'score_breakdown': score_breakdown,
            
            # Рекомендация
            'recommendation': recommendation,
            
            # Время анализа
            'analysis_time': now.isoformat()
        }
        
        # Короткое summary для быстрой проверки
        result['summary'] = (
            f"{emoji} {name}: {score:.0f}/100 | "
            f"Mmax7d={stats_basic['max_mag_7d']:.1f} | "
            f"N24h={stats_basic['events_24h']} | "
            f"Swarm={'ДА' if swarm['is_swarm'] else 'нет'}"
        )
        
        logger.info(f"  {result['summary']}")
        
        return result
    
    def analyze_all(self, data_by_region):
        """Анализ ВСЕХ регионов."""
        results = []
        
        for key, config in self.settings['regions'].items():
            logger.info(f"\n🔬 Анализ: {config['name']}")
            df = data_by_region.get(key, pd.DataFrame())
            result = self.analyze_region(key, config, df)
            results.append(result)
        
        # Сортируем по убыванию риска
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return results
      
