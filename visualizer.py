#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer.py — супер-красивая визуализация данных LAIC
Включает: 2D-карту, 3D-карту, мультипанельные графики, ионосферу, солнечную активность
"""

import os
import io
import tempfile
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Устанавливаем уровень логов для matplotlib, чтобы скрыть предупреждения
plt.set_loglevel('warning')
logger = logging.getLogger(__name__)

try:
    from htmlmin import minify
    HAS_HTMLMIN = True
except ImportError:
    HAS_HTMLMIN = False


# ═══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def _get_mag_column(df):
    for col in ['mag', 'magnitude', 'M', 'Magnitude']:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        logger.warning(f"Колонка магнитуды не найдена. Использую '{numeric_cols[0]}'.")
        return numeric_cols[0]
    raise KeyError("Не найдена колонка с магнитудой")


def _get_depth_column(df):
    for col in ['depth', 'depth_km', 'Depth', 'd']:
        if col in df.columns:
            return col
    return None


# ═══════════════════════════════════════════════════════════════
# 1. PNG-ГРАФИКИ (для Telegram)
# ═══════════════════════════════════════════════════════════════

def plot_magnitude_series(df: pd.DataFrame, region_name: str) -> bytes:
    """Временной ряд магнитуд с гистограммой и скользящими средними."""
    if df.empty:
        return None
    mag_col = _get_mag_column(df)
    if 'time' not in df.columns:
        logger.error("Колонка 'time' не найдена")
        return None

    df_sorted = df.sort_values('time')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # ---- Верхний график: магнитуды + скользящие средние ----
    ax1.plot(df_sorted['time'], df_sorted[mag_col], 'o-', color='royalblue',
             markersize=3, linewidth=0.8, alpha=0.6, label='Магнитуда')
    
    # Скользящее среднее за 7 дней
    rolling_7d = df_sorted[mag_col].rolling(7, min_periods=1).mean()
    ax1.plot(df_sorted['time'], rolling_7d, 'r-', linewidth=2, alpha=0.8, label='Скользящее среднее (7д)')
    
    # Скользящее среднее за 30 дней
    rolling_30d = df_sorted[mag_col].rolling(30, min_periods=1).mean()
    ax1.plot(df_sorted['time'], rolling_30d, 'g--', linewidth=2, alpha=0.7, label='Скользящее среднее (30д)')
    
    # Пороговые линии
    ax1.axhline(y=6.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='M=6.0')
    ax1.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='M=5.0')
    
    ax1.set_ylabel('Магнитуда (M)', fontsize=12)
    ax1.set_title(f'[Chart] {region_name} – Временной ряд магнитуд (сглаженный)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ---- Нижний график: гистограмма ----
    bins = np.arange(0, 9, 0.5)
    ax2.hist(df_sorted[mag_col], bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=6.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='M=6.0')
    ax2.axvline(x=5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='M=5.0')
    ax2.set_xlabel('Дата', fontsize=12)
    ax2.set_ylabel('Количество', fontsize=12)
    ax2.set_title('[Chart] Распределение магнитуд', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ---- Улучшенные метки времени ----
    total_days = (df_sorted['time'].max() - df_sorted['time'].min()).days
    if total_days > 60:
        interval = 7
    elif total_days > 30:
        interval = 3
    else:
        interval = 1

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    # Ограничиваем количество меток на оси X до 10
    import matplotlib.ticker as mticker
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.xticks(rotation=30)
    #ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    #ax2.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    #plt.xticks(rotation=30)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def create_anomaly_plot(df_anomalies, region_name):
    """График с выделением аномалий."""
    if df_anomalies.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    
    normal = df_anomalies[~df_anomalies['is_anomaly']]
    anomalies = df_anomalies[df_anomalies['is_anomaly']]

    ax.scatter(normal.index, normal['magnitude'],
               color='#2ecc71', label='Normal', alpha=0.6, s=40)
    ax.scatter(anomalies.index, anomalies['magnitude'],
               color='#e74c3c', label='[Anomaly]', s=80, marker='X', edgecolor='black', linewidth=1.5)

    ax.axhline(y=6.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold M=6.0')
    ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold M=5.0')
    
    ax.set_xlabel('Event (#)', fontsize=12)
    ax.set_ylabel('Magnitude (M)', fontsize=12)
    ax.set_title(f'[AI] Anomaly detection — {region_name}', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def create_solar_activity_plot(space_data, region_name, mag=0):
    """График солнечной активности (Kp, F10.7, Dst)."""
    if not space_data:
        return None
    
    kp_df = space_data.get('kp', pd.DataFrame())
    f107_df = space_data.get('f107', pd.DataFrame())
    dst_df = space_data.get('dst', pd.DataFrame())
    
    if kp_df.empty and f107_df.empty and dst_df.empty:
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Kp
    if not kp_df.empty:
        axes[0].plot(kp_df.index, kp_df['kp'], 'b-', linewidth=1.5, alpha=0.8, label='Kp')
        axes[0].fill_between(kp_df.index, 0, kp_df['kp'], alpha=0.15, color='blue')
        axes[0].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Threshold Kp=5')
        axes[0].axhline(y=7, color='red', linestyle='--', alpha=0.7, label='Threshold Kp=7')
        axes[0].set_ylabel('Kp index', fontsize=11)
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Kp data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_ylabel('Kp')
    
    # F10.7
    if not f107_df.empty:
        axes[1].plot(f107_df.index, f107_df['f107'], 'r-', linewidth=2, alpha=0.8, label='F10.7')
        axes[1].fill_between(f107_df.index, 0, f107_df['f107'], alpha=0.15, color='red')
        axes[1].axhline(y=150, color='orange', linestyle='--', alpha=0.7, label='Threshold 150')
        axes[1].set_ylabel('F10.7 (sfu)', fontsize=11)
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No F10.7 data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_ylabel('F10.7')
    
    # Dst
    if not dst_df.empty:
        axes[2].plot(dst_df.index, dst_df['dst'], 'g-', linewidth=1.5, alpha=0.8, label='Dst')
        # Заливка только где Dst < -50
        axes[2].fill_between(dst_df.index, -100, dst_df['dst'], alpha=0.2, color='green', where=(dst_df['dst'] < -50))
        axes[2].axhline(y=-50, color='orange', linestyle='--', alpha=0.7, label='Threshold -50')
        axes[2].axhline(y=-100, color='red', linestyle='--', alpha=0.7, label='Threshold -100')
        axes[2].set_ylabel('Dst (nT)', fontsize=11)
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No Dst data', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_ylabel('Dst')
        axes[2].set_xlabel('Date', fontsize=11)
    
    # Вертикальная линия события, если есть event_time
    if 'event_time' in space_data:
        event_time = space_data['event_time']
        for ax in axes:
            ax.axvline(x=event_time, color='black', linestyle='-', linewidth=2, alpha=0.7, label='[Event]')
    
    plt.suptitle(f'[Solar] Solar activity — {region_name} (M{mag:.1f})', fontsize=14)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def create_iono_anomaly_plot(iono_result, region_name):
    """График ионосферной аномалии (TEC с Z-score)."""
    if not iono_result:
        return None
    
    tec_df = iono_result.get('tec', pd.DataFrame())
    bg_df = iono_result.get('background', pd.DataFrame())
    
    if tec_df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # TEC
    ax1.plot(tec_df.index, tec_df['tec_value'], 'b-', linewidth=2, label='TEC (event)')
    if not bg_df.empty:
        ax1.plot(bg_df.index, bg_df['tec_value'], 'g--', linewidth=1.5, alpha=0.7, label='TEC (background, previous year)')
    ax1.set_ylabel('TEC (TECU)', fontsize=11)
    ax1.set_title(f'[Iono] Ionospheric anomaly — {region_name}', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Z-score
    if 'z_score' in tec_df.columns:
        ax2.plot(tec_df.index, tec_df['z_score'], 'r-', linewidth=2, label='Z-score')
        ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold +2σ')
        ax2.axhline(y=-2.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold -2σ')
        ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold +3σ')
        ax2.axhline(y=-3.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold -3σ')
        ax2.fill_between(tec_df.index, -2, 2, alpha=0.1, color='green')
        ax2.set_ylabel('Z-score (σ)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Z-score not calculated', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_ylabel('Z-score')
        ax2.set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# 2. 3D-КАРТА (Plotly)
# ═══════════════════════════════════════════════════════════════

def create_3d_map(df: pd.DataFrame, region_name: str) -> str:
    """
    Создаёт 3D-карту землетрясений с осями: широта, долгота, глубина.
    Размер и цвет точек — по магнитуде.
    """
    if df.empty:
        return "<p>No data for 3D map</p>"

    mag_col = _get_mag_column(df)
    depth_col = _get_depth_column(df)
    
    df_plot = df.copy()
    
    # Глубина (положительная для отображения вниз)
    if depth_col:
        df_plot['depth_display'] = df_plot[depth_col].abs()
    else:
        df_plot['depth_display'] = 5  # default value
    
    # Текст при наведении
    hover_texts = []
    for _, row in df_plot.iterrows():
        text = f"<b>{row['place']}</b><br>"
        text += f"Magnitude: {row[mag_col]}<br>"
        if depth_col:
            text += f"Depth: {row[depth_col]} km<br>"
        text += f"Time: {row['time']}"
        hover_texts.append(text)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_plot['longitude'],
        y=df_plot['latitude'],
        z=-df_plot['depth_display'],
        mode='markers',
        marker=dict(
            size=df_plot[mag_col] * 2 + 3,
            color=df_plot[mag_col],               # числовое значение
            colorscale='Viridis',                 # цветовая шкала
            opacity=0.85,
            colorbar=dict(title='Magnitude'),
            line=dict(width=1, color='darkgrey')
        ),
        text=hover_texts,
        hoverinfo='text'
    )])

    fig.update_layout(
        title=dict(
            text=f'[Globe] 3D earthquake map — {region_name}',
            font=dict(size=18, color='#1a237e')
        ),
        scene=dict(
            xaxis=dict(
                title='Longitude',
                backgroundcolor='#f5f5f5',
                gridcolor='#e0e0e0',
                showbackground=True
            ),
            yaxis=dict(
                title='Latitude',
                backgroundcolor='#f5f5f5',
                gridcolor='#e0e0e0',
                showbackground=True
            ),
            zaxis=dict(
                title='Depth (km)',
                backgroundcolor='#f5f5f5',
                gridcolor='#e0e0e0',
                showbackground=True,
                autorange='reversed'
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.2, z=0.6)
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template='plotly_white',
        height=650
    )

    return plot(fig, output_type='div', include_plotlyjs='cdn')


# ═══════════════════════════════════════════════════════════════
# 3. ИНТЕРАКТИВНЫЙ HTML-ОТЧЁТ
# ═══════════════════════════════════════════════════════════════

def create_interactive_report(
    region_name: str,
    df: pd.DataFrame,
    lst_data: dict = None,
    period_days: int = 30,
    iono_data: dict = None,
    space_data: dict = None,
    anomaly_data: pd.DataFrame = None
    zone_risks: dict = None   # <-- добавить сюда
) -> str:
    """
    Создаёт самодостаточный HTML-отчёт с картами, графиками и таблицей.
    """
    if df.empty:
        return None

    mag_col = _get_mag_column(df)
    depth_col = _get_depth_column(df)

    if 'time' not in df.columns or 'place' not in df.columns:
        logger.error("Колонки 'time' или 'place' не найдены")
        return None
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.error("Колонки 'latitude' / 'longitude' не найдены")
        return None

    df_sorted = df.sort_values('time')
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # ========== 1. 2D-КАРТА (Folium) ==========
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='CartoDB positron'
    )
    
    # Тепловая карта
    heat_data = [[row['latitude'], row['longitude'], row[mag_col]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=20, blur=15, max_zoom=10,
            gradient={0.4: 'green', 0.65: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)
    
    # Маркеры с кластеризацией
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in df.iterrows():
        mag = row[mag_col]
        if mag >= 6:
            color = 'red'
        elif mag >= 5:
            color = 'orange'
        elif mag >= 4:
            color = 'yellow'
        else:
            color = 'green'
        
        popup_text = f"""
        <b>{row['place']}</b><br>
        <b>Magnitude:</b> {mag}<br>
        <b>Depth:</b> {row[depth_col] if depth_col else 'N/A'} km<br>
        <b>Time:</b> {row['time']}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=mag * 1.5,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=2
        ).add_to(marker_cluster)

    # Легенда
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 13px;">
        <b>Magnitude</b><br>
        <span style="color:red;">●</span> ≥ 6.0<br>
        <span style="color:orange;">●</span> 5.0 – 5.9<br>
        <span style="color:yellow;">●</span> 4.0 – 4.9<br>
        <span style="color:green;">●</span> < 4.0
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    # ========== ЗОНЫ РИСКА НА КАРТЕ ==========
    if zone_risks:
        # Добавляем разломы из GeoJSON
        try:
            import json
            with open('data/gem_active_faults.geojson', 'r') as f:
                faults = json.load(f)
            folium.GeoJson(
                faults,
                name='Активные разломы',
                style_function=lambda x: {'color': 'gray', 'weight': 1, 'opacity': 0.5}
            ).add_to(m)
        except Exception as e:
            logger.warning(f"Не удалось добавить разломы на карту: {e}")

        # Для каждой зоны с риском добавляем маркер или круг
        for zone_id, zone_data in zone_risks.items():
            risk = zone_data.get('risk', {})
            if risk.get('level') in ['critical', 'high']:
                color = 'red' if risk['level'] == 'critical' else 'orange'
                lat = zone_data.get('lat')
                lon = zone_data.get('lon')
                if lat is not None and lon is not None:
                    folium.Circle(
                        location=[lat, lon],
                        radius=50000,  # 50 км
                        color=color,
                        fill=True,
                        fill_opacity=0.2,
                        popup=f"Зона {zone_id}: {risk['level'].upper()}"
                    ).add_to(m)
    map_html = m._repr_html_()

    # ========== 2. 3D-КАРТА ==========
    if len(df) >= 3:
        plotly_3d_div = create_3d_map(df, region_name)
    else:
        plotly_3d_div = "<p>Not enough data for 3D visualization (min 3 events)</p>"

    # ========== 3. МУЛЬТИПАНЕЛЬНЫЕ ГРАФИКИ (Plotly) ==========
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Magnitudes (time)', 'Histogram of magnitudes',
            'Cumulative energy', 'Depths (if available)',
            'Kp index', 'TEC Z-score'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1) Магнитуды (время)
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=df_sorted[mag_col],
                   mode='lines+markers', name='Magnitude',
                   marker=dict(size=5, color='red')),
        row=1, col=1
    )
    for threshold in [4, 5, 6]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 2) Гистограмма магнитуд
    bins = np.arange(0, 9, 0.5)
    hist_vals, _ = np.histogram(df_sorted[mag_col], bins=bins)
    fig.add_trace(
        go.Bar(x=bins[:-1], y=hist_vals, name='Distribution', marker_color='steelblue'),
        row=1, col=2
    )
    
    # 3) Накопленная энергия
    energy = df_sorted[mag_col].apply(lambda m: 10**(1.5*m + 4.8) / 4.184e12)
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=energy.cumsum(),
                   mode='lines', name='Cumulative energy', line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Energy (kt TNT)", row=2, col=1)
    
    # 4) Глубины
    if depth_col:
        fig.add_trace(
            go.Scatter(x=df_sorted['time'], y=df_sorted[depth_col],
                       mode='markers', name='Depth',
                       marker=dict(size=6, color='purple')),
            row=2, col=2
        )
        fig.update_yaxes(title_text="Depth (km)", row=2, col=2)
    else:
        fig.add_annotation(text="No depth data", row=2, col=2, showarrow=False)
    
    # 5) Kp (если есть space_data)
    if space_data and 'kp' in space_data and not space_data['kp'].empty:
        kp_df = space_data['kp']
        fig.add_trace(
            go.Scatter(x=kp_df.index, y=kp_df['kp'],
                       mode='lines', name='Kp', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=5, line_dash="dash", line_color="orange", row=3, col=1)
        fig.add_hline(y=7, line_dash="dash", line_color="red", row=3, col=1)
    fig.update_yaxes(title_text="Kp", row=3, col=1)
    
    # 6) TEC Z-score (если есть iono_data)
    if iono_data and 'tec' in iono_data and 'z_score' in iono_data['tec'].columns:
        tec_df = iono_data['tec']
        fig.add_trace(
            go.Scatter(x=tec_df.index, y=tec_df['z_score'],
                       mode='lines', name='Z-score', line=dict(color='red', width=2)),
            row=3, col=2
        )
        fig.add_hline(y=2, line_dash="dash", line_color="orange", row=3, col=2)
        fig.add_hline(y=-2, line_dash="dash", line_color="orange", row=3, col=2)
        fig.update_yaxes(title_text="Z-score (σ)", row=3, col=2)
    
    fig.update_layout(height=900, width=1100, template='plotly_white',
                      title=f'[Chart] {region_name} – Comprehensive analysis')
    plotly_div = plot(fig, output_type='div', include_plotlyjs='cdn')

    # ========== 4. Дополнительный график солнечной активности (Plotly) ==========
    if space_data:
        solar_div = create_solar_plotly(space_data)
    else:
        solar_div = "<p>No solar activity data</p>"

    # ========== 5. ТАБЛИЦА сильных событий ==========
    strong_events = df_sorted[df_sorted[mag_col] >= 5.0].head(10)
    if not strong_events.empty:
        table_cols = ['time', 'place', mag_col]
        if depth_col:
            table_cols.append(depth_col)
        table_data = strong_events[table_cols].copy()
        rename_map = {mag_col: 'mag'}
        if depth_col:
            rename_map[depth_col] = 'depth'
        table_data.rename(columns=rename_map, inplace=True)
        table_html = table_data.to_html(classes='table table-striped table-bordered', 
                                        index=False, table_id='strong-events')
    else:
        table_html = "<p>No events M≥5.0</p>"

    # ========== 6. СТАТИСТИКА ==========
    max_mag_val = df[mag_col].max()
    mean_mag_val = df[mag_col].mean()
    lst_str = ''
    if lst_data and lst_data.get('lst_celsius') is not None:
        lst_str = f'<div class="stat-item"><span class="stat-label">LST</span><br><span class="stat-value">{lst_data["lst_celsius"]:.2f}°C</span></div>'

    stats_html = f'''
    <div class="stats-grid">
        <div class="stat-item"><span class="stat-label">Period</span><br><span class="stat-value">{df['time'].min().strftime('%Y-%m-%d')} — {df['time'].max().strftime('%Y-%m-%d')}</span></div>
        <div class="stat-item"><span class="stat-label">Events</span><br><span class="stat-value">{len(df)}</span></div>
        <div class="stat-item"><span class="stat-label">Max Magnitude</span><br><span class="stat-value">{max_mag_val:.1f}</span></div>
        <div class="stat-item"><span class="stat-label">Mean Magnitude</span><br><span class="stat-value">{mean_mag_val:.2f}</span></div>
        {lst_str}
    </div>
    '''

    # ========== 7. СБОРКА HTML ==========
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LAIC report – {region_name}</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
            body {{ padding: 20px; background: #f8f9fa; }}
            .container {{ max-width: 1300px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h3 {{ color: #2c3e50; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stat-item {{ text-align: center; }}
            .stat-label {{ font-size: 12px; color: #6c757d; }}
            .stat-value {{ font-size: 18px; font-weight: bold; color: #1a237e; }}
            .row {{ margin-top: 25px; }}
            .footer {{ margin-top: 30px; text-align: center; color: #6c757d; font-size: 12px; }}
            #strong-events_wrapper {{ margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 {region_name} – Seismic Report</h1>
            {stats_html}
            <div class="row"><div class="col-12"><h3>Interactive Map</h3><div style="height: 500px;">{map_html}</div></div></div>
            <div class="row"><div class="col-12"><h3>3D Visualization</h3>{plotly_3d_div}</div></div>
            <div class="row"><div class="col-12"><h3>Comprehensive Analysis</h3>{plotly_div}</div></div>
            <div class="row"><div class="col-12"><h3>Solar Activity</h3>{solar_div}</div></div>
            <div class="row"><div class="col-12"><h3>Strong Events (M≥5.0)</h3>{table_html}</div></div>
            <div class="footer">Generated automatically by LAIC bot • Data: USGS, NOAA, NASA OMNIWeb</div>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
        <script>$(document).ready(function(){{ $('#strong-events').DataTable({{ "order": [[0, "desc"]], "pageLength": 10, "lengthMenu": [[5,10,25,-1],[5,10,25,"All"]] }}); }});</script>
    </body>
    </html>
    '''

    if HAS_HTMLMIN:
        try:
            html_minified = minify(html_template, remove_empty_space=True, remove_comments=True)
        except Exception:
            html_minified = html_template
    else:
        html_minified = html_template

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_minified)
        return f.name


# ═══════════════════════════════════════════════════════════════
# 4. ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ДЛЯ HTML
# ═══════════════════════════════════════════════════════════════

def create_solar_plotly(space_data: dict) -> str:
    """
    Создаёт интерактивный график солнечной активности (Kp, F10.7, Dst) с помощью Plotly.
    """
    if not space_data:
        return "<p>No solar activity data</p>"
    
    kp_df = space_data.get('kp', pd.DataFrame())
    f107_df = space_data.get('f107', pd.DataFrame())
    dst_df = space_data.get('dst', pd.DataFrame())
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Kp index', 'F10.7 (solar flux)', 'Dst index')
    )
    
    # Kp
    if not kp_df.empty:
        fig.add_trace(
            go.Scatter(x=kp_df.index, y=kp_df['kp'],
                       mode='lines', name='Kp',
                       line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=5, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=7, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_yaxes(title_text="Kp", row=1, col=1)
    
    # F10.7
    if not f107_df.empty:
        fig.add_trace(
            go.Scatter(x=f107_df.index, y=f107_df['f107'],
                       mode='lines', name='F10.7',
                       line=dict(color='#ff7f0e', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=150, line_dash="dash", line_color="orange", row=2, col=1)
        fig.update_yaxes(title_text="F10.7 (sfu)", row=2, col=1)
    
    # Dst
    if not dst_df.empty:
        fig.add_trace(
            go.Scatter(x=dst_df.index, y=dst_df['dst'],
                       mode='lines', name='Dst',
                       line=dict(color='#2ca02c', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=-50, line_dash="dash", line_color="orange", row=3, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="red", row=3, col=1)
        fig.update_yaxes(title_text="Dst (nT)", row=3, col=1)
    
    fig.update_layout(
        height=500,
        title=dict(text='Solar activity over period', font=dict(size=14)),
        template='plotly_white',
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=40)
    )
    
    return plot(fig, output_type='div', include_plotlyjs='cdn')
