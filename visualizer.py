#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer.py — визуализация данных LAIC
"""

import os
import io
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import plotly.graph_objs as go
from plotly.offline import plot
import logging

logger = logging.getLogger(__name__)

# Попытка импортировать htmlmin (опционально)
try:
    from htmlmin import minify
    HAS_HTMLMIN = True
except ImportError:
    HAS_HTMLMIN = False


def _get_mag_column(df):
    """Возвращает имя колонки с магнитудой."""
    for col in ['mag', 'magnitude', 'M', 'Magnitude']:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        logger.warning(f"⚠️ Колонка магнитуды не найдена. Использую '{numeric_cols[0]}'.")
        return numeric_cols[0]
    raise KeyError("Не найдена колонка с магнитудой в данных.")


def _get_depth_column(df):
    """Возвращает имя колонки с глубиной (или None, если нет)."""
    for col in ['depth', 'depth_km', 'Depth', 'd']:
        if col in df.columns:
            return col
    return None


def plot_magnitude_series(df: pd.DataFrame, region_name: str) -> bytes:
    """Создаёт временной ряд магнитуд (PNG)."""
    if df.empty:
        return None

    mag_col = _get_mag_column(df)
    if 'time' not in df.columns:
        logger.error("Колонка 'time' не найдена в данных")
        return None

    df_sorted = df.sort_values('time')

    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['time'], df_sorted[mag_col], 'o-', color='royalblue', markersize=4, linewidth=1.5)
    plt.title(f'{region_name} – временной ряд магнитуд', fontsize=12)
    plt.xlabel('Дата', fontsize=10)
    plt.ylabel('Магнитуда (M)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=20, fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def create_interactive_report(
    region_name: str,
    df: pd.DataFrame,
    lst_data: dict = None,
    period_days: int = 30
) -> str:
    """
    Создаёт самодостаточный HTML-отчёт с картой, графиком и таблицей.
    """
    if df.empty:
        return None

    mag_col = _get_mag_column(df)
    depth_col = _get_depth_column(df)

    if 'time' not in df.columns:
        logger.error("Колонка 'time' не найдена в данных")
        return None
    if 'place' not in df.columns:
        logger.error("Колонка 'place' не найдена в данных")
        return None
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.error("Колонки 'latitude' / 'longitude' не найдены")
        return None

    df_sorted = df.sort_values('time')
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # ========== 1. КАРТА (Folium) ==========
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )

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
        radius = mag ** 2

        # Формируем popup с учётом наличия глубины
        popup_text = f"<b>{row['place']}</b><br>Магнитуда: {mag}"
        if depth_col is not None:
            popup_text += f"<br>Глубина: {row[depth_col]} км"
        popup_text += f"<br>Время: {row['time']}"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1
        ).add_to(m)

    # Тепловой слой (плотность)
    heat_data = [[row['latitude'], row['longitude'], row[mag_col]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)

    # LST-маркер (если есть)
    if lst_data and lst_data.get('lst_celsius') is not None:
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Средняя LST: {lst_data['lst_celsius']}°C",
            icon=folium.Icon(color='purple', icon='thermometer', prefix='fa')
        ).add_to(m)

    # Легенда
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3); font-size: 12px;">
        <b>Магнитуда</b><br>
        <i class="fa fa-circle" style="color:red;"></i> ≥ 6<br>
        <i class="fa fa-circle" style="color:orange;"></i> 5–5.9<br>
        <i class="fa fa-circle" style="color:yellow;"></i> 4–4.9<br>
        <i class="fa fa-circle" style="color:green;"></i> < 4
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ========== 2. ГРАФИК (Plotly) ==========
    max_mag = df[mag_col].max()
    trace = go.Scatter(
        x=df_sorted['time'],
        y=df_sorted[mag_col],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='blue', width=2),
        name='Магнитуда'
    )

    # Добавляем горизонтальные линии для порогов
    shapes = []
    for threshold in [4, 5, 6]:
        shapes.append(dict(
            type='line',
            x0=df_sorted['time'].min(),
            x1=df_sorted['time'].max(),
            y0=threshold,
            y1=threshold,
            line=dict(dash='dash', color='gray', width=1)
        ))

    annotations = [
        dict(
            x=df_sorted['time'].max(),
            y=max_mag,
            text=f'Макс. {max_mag:.1f}',
            showarrow=True,
            arrowhead=1,
            bgcolor='white'
        )
    ]

    layout = go.Layout(
        title=f'{region_name} – динамика магнитуд за последние {period_days} дней',
        xaxis_title='Дата',
        yaxis_title='Магнитуда (M)',
        hovermode='closest',
        template='plotly_white',
        height=500,
        width=900,
        font=dict(size=14),
        margin=dict(l=60, r=30, t=60, b=80),
        shapes=shapes,
        annotations=annotations
    )

    fig = go.Figure(data=[trace], layout=layout)
    plotly_div = plot(fig, output_type='div', include_plotlyjs='cdn')

    # ========== 3. ТАБЛИЦА (DataTables) ==========
    # Формируем список колонок для таблицы
    table_columns = ['time', 'place', mag_col]
    if depth_col is not None:
        table_columns.append(depth_col)

    table_data = df_sorted[table_columns].copy()
    # Переименовываем для красоты
    rename_map = {mag_col: 'mag'}
    if depth_col is not None:
        rename_map[depth_col] = 'depth'
    table_data.rename(columns=rename_map, inplace=True)

    table_html = table_data.to_html(
        classes='table table-striped table-bordered',
        index=False,
        table_id='quakes-table'
    )

    # ========== 4. ССЫЛКИ ==========
    usgs_url = f"https://earthquake.usgs.gov/earthquakes/search/#%7B%22starttime%22%3A%22{df['time'].min()}%22%2C%22endtime%22%3A%22{df['time'].max()}%22%2C%22minmagnitude%22%3A0%2C%22region%22%3A%22{region_name}%22%7D"
    google_maps_url = f"https://www.google.com/maps/@{center_lat},{center_lon},5z"

    # ========== 5. КНОПКИ ЭКСПОРТА ==========
    export_script = '''
    <script>
    function exportCSV() {
        var table = document.getElementById('quakes-table');
        var rows = table.querySelectorAll('tr');
        var csv = [];
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            var cols = row.querySelectorAll('td, th');
            var rowData = [];
            for (var j = 0; j < cols.length; j++) {
                rowData.push('"' + cols[j].innerText.replace(/"/g, '""') + '"');
            }
            csv.push(rowData.join(','));
        }
        var csvContent = csv.join('\\n');
        var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        var link = document.createElement('a');
        var url = URL.createObjectURL(blob);
        link.href = url;
        link.setAttribute('download', 'earthquakes.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    </script>
    <button class="btn btn-success" onclick="exportCSV()">⬇ Экспорт CSV</button>
    <button class="btn btn-secondary" onclick="window.print()">🖨 Печать</button>
    '''

    # ========== 6. СТАТИСТИКА ==========
    max_mag_val = df[mag_col].max()
    mean_mag_val = df[mag_col].mean()
    lst_str = ''
    if lst_data and lst_data.get('lst_celsius') is not None:
        lst_str = f'<p><strong>Средняя температура поверхности (LST):</strong> {lst_data["lst_celsius"]:.2f}°C</p>'

    # ========== 7. СБОРКА HTML ==========
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LAIC отчёт – {region_name}</title>
        <!-- Bootstrap 4 -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <!-- DataTables -->
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
        <!-- Font Awesome -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
            body {{ padding: 20px; background: #f8f9fa; }}
            .container {{ max-width: 1300px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h3 {{ color: #2c3e50; }}
            .stats {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .row {{ margin-top: 25px; }}
            .btn-group {{ margin-top: 15px; }}
            .footer {{ margin-top: 30px; text-align: center; color: #6c757d; font-size: 12px; }}
            #quakes-table_wrapper {{ margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 {region_name} – сейсмический отчёт</h1>
            <div class="stats">
                <p><strong>Период:</strong> {df['time'].min()} — {df['time'].max()}</p>
                <p><strong>Количество событий:</strong> {len(df)}</p>
                <p><strong>Максимальная магнитуда:</strong> {max_mag_val:.1f}</p>
                <p><strong>Средняя магнитуда:</strong> {mean_mag_val:.2f}</p>
                {lst_str}
                <p><strong>Ссылки:</strong> <a href="{usgs_url}" target="_blank">USGS</a> | <a href="{google_maps_url}" target="_blank">Google Maps</a></p>
            </div>

            <div class="row">
                <div class="col-12">
                    <h3>Интерактивная карта</h3>
                    <div style="height: 500px;">
                        {m._repr_html_()}
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-12">
                    <h3>Динамика магнитуд</h3>
                    {plotly_div}
                </div>
            </div>

            <div class="row">
                <div class="col-12">
                    <h3>Список событий</h3>
                    <div class="btn-group">
                        {export_script}
                    </div>
                    {table_html}
                </div>
            </div>

            <div class="footer">
                Отчёт сгенерирован автоматически ботом LAIC  •  Данные: USGS
            </div>
        </div>

        <!-- jQuery, DataTables -->
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
        </script>
    </body>
    </html>
    """

    # Минификация (если доступна)
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


def create_anomaly_plot(df_anomalies, region_name):
    """
    Создаёт график с выделением аномалий.
    """
    if df_anomalies.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    # Нормальные события (зелёные)
    normal = df_anomalies[~df_anomalies['is_anomaly']]
    anomalies = df_anomalies[df_anomalies['is_anomaly']]

    ax.scatter(normal.index, normal['magnitude'],
               color='green', label='Норма', alpha=0.6, s=30)

    ax.scatter(anomalies.index, anomalies['magnitude'],
               color='red', label='Аномалия', s=50, marker='X', edgecolor='black')

    ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='Порог M=5.0')
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.5, label='Порог M=6.0')

    ax.set_xlabel('Событие (#)')
    ax.set_ylabel('Магнитуда (M)')
    ax.set_title(f'Обнаружение аномалий — {region_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    return buf.getvalue()
