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
from datetime import datetime, timedelta

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
        logger.warning(f"⚠️ Колонка магнитуды не найдена. Использую '{numeric_cols[0]}'.")
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
    """Временной ряд магнитуд с гистограммой."""
    if df.empty:
        return None
    mag_col = _get_mag_column(df)
    if 'time' not in df.columns:
        logger.error("Колонка 'time' не найдена")
        return None

    df_sorted = df.sort_values('time')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Верхний график: магнитуды
    ax1.plot(df_sorted['time'], df_sorted[mag_col], 'o-', color='royalblue', 
             markersize=5, linewidth=1.5, alpha=0.8)
    ax1.axhline(y=6.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='M=6.0')
    ax1.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='M=5.0')
    ax1.set_ylabel('Магнитуда (M)', fontsize=11)
    ax1.set_title(f'📊 {region_name} – Временной ряд магнитуд', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Нижний график: гистограмма
    bins = np.arange(0, 9, 0.5)
    ax2.hist(df_sorted[mag_col], bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=6.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='M=6.0')
    ax2.axvline(x=5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='M=5.0')
    ax2.set_xlabel('Дата', fontsize=11)
    ax2.set_ylabel('Количество', fontsize=11)
    ax2.set_title('📊 Распределение магнитуд', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df_sorted)//10)))
    plt.xticks(rotation=20)
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
               color='#2ecc71', label='Норма', alpha=0.6, s=40)
    ax.scatter(anomalies.index, anomalies['magnitude'],
               color='#e74c3c', label='🔴 Аномалия', s=80, marker='X', edgecolor='black', linewidth=1.5)

    ax.axhline(y=6.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Порог M=6.0')
    ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Порог M=5.0')
    
    ax.set_xlabel('Событие (#)', fontsize=12)
    ax.set_ylabel('Магнитуда (M)', fontsize=12)
    ax.set_title(f'🧠 Обнаружение аномалий — {region_name}', fontsize=14)
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
        axes[0].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Порог Kp=5')
        axes[0].axhline(y=7, color='red', linestyle='--', alpha=0.7, label='Порог Kp=7')
        axes[0].set_ylabel('Kp-индекс', fontsize=11)
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Нет данных Kp', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_ylabel('Kp')
    
    # F10.7
    if not f107_df.empty:
        axes[1].plot(f107_df.index, f107_df['f107'], 'r-', linewidth=2, alpha=0.8, label='F10.7')
        axes[1].fill_between(f107_df.index, 0, f107_df['f107'], alpha=0.15, color='red')
        axes[1].axhline(y=150, color='orange', linestyle='--', alpha=0.7, label='Порог 150')
        axes[1].set_ylabel('F10.7 (sfu)', fontsize=11)
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Нет данных F10.7', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_ylabel('F10.7')
    
    # Dst
    if not dst_df.empty:
        axes[2].plot(dst_df.index, dst_df['dst'], 'g-', linewidth=1.5, alpha=0.8, label='Dst')
        axes[2].fill_between(dst_df.index, -100, dst_df['dst'], alpha=0.2, color='green', where=(dst_df['dst'] < -50))
        axes[2].axhline(y=-50, color='orange', linestyle='--', alpha=0.7, label='Порог -50')
        axes[2].axhline(y=-100, color='red', linestyle='--', alpha=0.7, label='Порог -100')
        axes[2].set_ylabel('Dst (нТл)', fontsize=11)
        axes[2].set_xlabel('Дата', fontsize=11)
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Нет данных Dst', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_ylabel('Dst')
        axes[2].set_xlabel('Дата', fontsize=11)
    
    # Вертикальная линия события
    if 'event_time' in space_data:
        for ax in axes:
            ax.axvline(x=space_data['event_time'], color='black', linestyle='-', 
                      linewidth=2, alpha=0.7, label='📌 Событие')
    
    plt.suptitle(f'☀️ Солнечная активность — {region_name} (M{mag:.1f})', fontsize=14)
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
    ax1.plot(tec_df.index, tec_df['tec_value'], 'b-', linewidth=2, label='TEC (событие)')
    if not bg_df.empty:
        ax1.plot(bg_df.index, bg_df['tec_value'], 'g--', linewidth=1.5, alpha=0.7, label='TEC (фон, прошлый год)')
    ax1.set_ylabel('TEC (TECU)', fontsize=11)
    ax1.set_title(f'🛰️ Ионосферная аномалия — {region_name}', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Z-score
    if 'z_score' in tec_df.columns:
        ax2.plot(tec_df.index, tec_df['z_score'], 'r-', linewidth=2, label='Z-score')
        ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Порог +2σ')
        ax2.axhline(y=-2.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Порог -2σ')
        ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Порог +3σ')
        ax2.axhline(y=-3.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Порог -3σ')
        ax2.fill_between(tec_df.index, -2, 2, alpha=0.1, color='green')
        ax2.set_ylabel('Z-score (σ)', fontsize=11)
        ax2.set_xlabel('Дата', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Z-score не рассчитан', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_ylabel('Z-score')
        ax2.set_xlabel('Дата', fontsize=11)
    
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
        return "<p>⚠️ Нет данных для 3D-карты</p>"

    mag_col = _get_mag_column(df)
    depth_col = _get_depth_column(df)
    
    df_plot = df.copy()
    
    # Глубина (положительная для отображения вниз)
    if depth_col:
        df_plot['depth_display'] = df_plot[depth_col].abs()
    else:
        df_plot['depth_display'] = 5  # Значение по умолчанию
    
    # Цвета по магнитуде (для Plotly)
    color_scale = []
    for mag in df_plot[mag_col]:
        if mag >= 6:
            color_scale.append('red')
        elif mag >= 5:
            color_scale.append('orange')
        elif mag >= 4:
            color_scale.append('yellow')
        else:
            color_scale.append('green')
    
    # Текст при наведении
    hover_texts = []
    for _, row in df_plot.iterrows():
        text = f"<b>{row['place']}</b><br>"
        text += f"Магнитуда: {row[mag_col]}<br>"
        if depth_col:
            text += f"Глубина: {row[depth_col]} км<br>"
        text += f"Время: {row['time']}"
        hover_texts.append(text)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_plot['longitude'],
        y=df_plot['latitude'],
        z=-df_plot['depth_display'],
        mode='markers',
        marker=dict(
            size=df_plot[mag_col] * 2 + 3,
            color=color_scale,
            opacity=0.85,
            line=dict(width=1, color='darkgrey')
        ),
        text=hover_texts,
        hoverinfo='text'
    )])

    fig.update_layout(
        title=dict(
            text=f'🌍 3D-карта землетрясений — {region_name}',
            font=dict(size=18, color='#1a237e')
        ),
        scene=dict(
            xaxis=dict(
                title='Долгота',
                backgroundcolor='#f5f5f5',
                gridcolor='#e0e0e0',
                showbackground=True
            ),
            yaxis=dict(
                title='Широта',
                backgroundcolor='#f5f5f5',
                gridcolor='#e0e0e0',
                showbackground=True
            ),
            zaxis=dict(
                title='Глубина (км)',
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
        height=650,
        legend=dict(
            title='Магнитуда',
            itemsizing='constant'
        )
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
        <b>Магнитуда:</b> {mag}<br>
        <b>Глубина:</b> {row[depth_col] if depth_col else 'N/A'} км<br>
        <b>Время:</b> {row['time']}
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
        <b>Магнитуда</b><br>
        <span style="color:red;">●</span> ≥ 6.0<br>
        <span style="color:orange;">●</span> 5.0 – 5.9<br>
        <span style="color:yellow;">●</span> 4.0 – 4.9<br>
        <span style="color:green;">●</span> < 4.0
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ========== 2. 3D-КАРТА ==========
    if len(df) >= 3:
        plotly_3d_div = create_3d_map(df, region_name)
    else:
        plotly_3d_div = "<p>⚠️ Недостаточно данных для 3D-визуализации (минимум 3 события)</p>"

    # ========== 3. МУЛЬТИПАНЕЛЬНЫЕ ГРАФИКИ (Plotly) ==========
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '📈 Магнитуды (время)', '📊 Гистограмма магнитуд',
            '⚡ Накопленная энергия', '📏 Глубины',
            '☀️ Kp-индекс', '🛰️ TEC Z-score'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 2a. Магнитуды (время)
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=df_sorted[mag_col],
                   mode='lines+markers', name='Магнитуда',
                   marker=dict(size=5, color='red')),
        row=1, col=1
    )
    for threshold in [4, 5, 6]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 2b. Гистограмма магнитуд
    bins = np.arange(0, 9, 0.5)
    hist_data = np.histogram(df_sorted[mag_col], bins=bins)
    fig.add_trace(
        go.Bar(x=bins[:-1], y=hist_data[0], name='Распределение', marker_color='steelblue'),
        row=1, col=2
    )
    
    # 2c. Накопленная энергия
    energy = df_sorted[mag_col].apply(lambda m: 10**(1.5*m + 4.8) / 4.184e12)
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=energy.cumsum(), 
                   mode='lines', name='Накопленная энергия', line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Энергия (кт TNT)", row=2, col=1)
    
    # 2d. Глубины
    if depth_col:
        fig.add_trace(
            go.Scatter(x=df_sorted['time'], y=df_sorted[depth_col],
                       mode='markers', name='Глубина',
                       marker=dict(size=6, color='purple')),
            row=2, col=2
        )
        fig.update_yaxes(title_text="Глубина (км)", row=2, col=2)
    else:
        fig.add_annotation(text="Нет данных о глубине", row=2, col=2, showarrow=False)
    
    # 2e. Kp (если есть)
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
    
    # 2f. TEC Z-score (если есть)
    if iono_data and 'tec' in iono_data:
        tec_df = iono_data['tec']
        if 'z_score' in tec_df.columns:
            fig.add_trace(
                go.Scatter(x=tec_df.index, y=tec_df['z_score'],
                           mode='lines', name='Z-score', line=dict(color='red', width=2)),
                row=3, col=2
            )
            fig.add_hline(y=2, line_dash="dash", line_color="orange", row=3, col=2)
            fig.add_hline(y=-2, line_dash="dash", line_color="orange", row=3, col=2)
            fig.update_yaxes(title_text="Z-score (σ)", row=3, col=2)
    
    fig.update_layout(height=900, width=1100, template='plotly_white',
                      title=f'📊 {region_name} – Комплексный анализ')
    plotly_div = plot(fig, output_type='div', include_plotlyjs='cdn')

    # ========== 4. ТАБЛИЦА ==========
    table_cols = ['time', 'place', mag_col]
    if depth_col:
        table_cols.append(depth_col)
    table_data = df_sorted[table_cols].copy()
    rename_map = {mag_col: 'mag'}
    if depth_col:
        rename_map[depth_col] = 'depth'
    table_data.rename(columns=rename_map, inplace=True)
    table_html = table_data.to_html(classes='table table-striped table-bordered', 
                                    index=False, table_id='quakes-table')

    # ========== 5. СТАТИСТИКА ==========
    max_mag_val = df[mag_col].max()
    mean_mag_val = df[mag_col].mean()
    lst_str = ''
    if lst_data and lst_data.get('lst_celsius') is not None:
        lst_str = f'<div class="stat-item"><span class="stat-label">LST</span><br><span class="stat-value">{lst_data["lst_celsius"]:.1f}°C</span></div>'
    
    anomaly_count = 0
    if anomaly_data is not None and not anomaly_data.empty:
        anomaly_count = anomaly_data[anomaly_data['is_anomaly']].shape[0]

    # ========== 6. СБОРКА HTML ==========
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LAIC отчёт – {region_name}</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
            body {{ background: #f0f2f5; padding: 20px; }}
            .container {{ max-width: 1400px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a237e; font-weight: 700; }}
            .stats {{ background: linear-gradient(135deg, #e8eaf6, #f5f5f5); padding: 20px; border-radius: 10px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 20px; }}
            .stat-item {{ min-width: 80px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
            .stat-label {{ font-size: 13px; color: #666; }}
            .section-title {{ margin-top: 30px; margin-bottom: 15px; color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
            .btn-group {{ margin: 15px 0; }}
            .footer {{ margin-top: 30px; text-align: center; color: #999; font-size: 12px; }}
            #quakes-table_wrapper {{ margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 {region_name} – LAIC сейсмический отчёт</h1>
            
            <div class="stats">
                <div class="stat-item"><span class="stat-label">Период</span><br><span class="stat-value">{df['time'].min().strftime('%d.%m.%Y')} — {df['time'].max().strftime('%d.%m.%Y')}</span></div>
                <div class="stat-item"><span class="stat-label">Событий</span><br><span class="stat-value">{len(df)}</span></div>
                <div class="stat-item"><span class="stat-label">Макс M</span><br><span class="stat-value">{max_mag_val:.1f}</span></div>
                <div class="stat-item"><span class="stat-label">Ср. M</span><br><span class="stat-value">{mean_mag_val:.2f}</span></div>
                <div class="stat-item"><span class="stat-label">Аномалий (ИИ)</span><br><span class="stat-value">{anomaly_count}</span></div>
                {lst_str}
            </div>

            <h3 class="section-title">🗺️ Интерактивная карта</h3>
            <div style="height:500px;">{m._repr_html_()}</div>

            <h3 class="section-title">🌍 3D-карта землетрясений</h3>
            <div style="height:650px;">{plotly_3d_div}</div>

            <h3 class="section-title">📈 Комплексные графики</h3>
            {plotly_div}

            <h3 class="section-title">📋 Список событий</h3>
            <div class="btn-group">
                <button class="btn btn-success" onclick="exportCSV()">⬇ Экспорт CSV</button>
                <button class="btn btn-secondary" onclick="window.print()">🖨 Печать</button>
            </div>
            {table_html}

            <div class="footer">
                Отчёт сгенерирован автоматически ботом LAIC • Данные: USGS, NOAA, NASA<br>
                Методология: LAIC (Lithosphere-Atmosphere-Ionosphere Coupling)
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('#quakes-table').DataTable({{
                    "order": [[0, "desc"]],
                    "pageLength": 25,
                    "lengthMenu": [[10, 25, 50, -1], [10, 25, 50, "Все"]]
                }});
            }});
            
            function exportCSV() {{
                var table = document.getElementById('quakes-table');
                var rows = table.querySelectorAll('tr');
                var csv = [];
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    var cols = row.querySelectorAll('td, th');
                    var rowData = [];
                    for (var j = 0; j < cols.length; j++) {{
                        rowData.push('"' + cols[j].innerText.replace(/"/g, '""') + '"');
                    }}
                    csv.push(rowData.join(','));
                }}
                var csvContent = csv.join('\\n');
                var blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                var link = document.createElement('a');
                var url = URL.createObjectURL(blob);
                link.href = url;
                link.setAttribute('download', 'earthquakes.csv');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}
        </script>
    </body>
    </html>
    """
    
    if HAS_HTMLMIN:
        try:
            html_template = minify(html_template, remove_empty_space=True, remove_comments=True)
        except Exception:
            pass

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_template)
        return f.name
