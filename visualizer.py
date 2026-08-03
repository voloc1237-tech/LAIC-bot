import os
import io
import tempfile
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import plotly.graph_objs as go
from plotly.offline import plot
from htmlmin import minify  # если не установлен, закомментируйте строку с минификацией

def plot_magnitude_series(df: pd.DataFrame, region_name: str) -> bytes:
    """
    Создаёт временной ряд магнитуд (PNG) для быстрого просмотра в Telegram.
    """
    if df.empty:
        return None
    df_sorted = df.sort_values('time')
    plt.figure(figsize=(8, 4))
    plt.plot(df_sorted['time'], df_sorted['mag'], 'o-', color='royalblue', markersize=4, linewidth=1.5)
    plt.title(f'{region_name} – временной ряд магнитуд', fontsize=12)
    plt.xlabel('Дата', fontsize=10)
    plt.ylabel('Магнитуда (M)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=20, fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
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
    Создаёт самодостаточный HTML-отчёт с:
    - интерактивной картой (Folium) с маркерами, тепловой картой и легендой,
    - графиком Plotly,
    - таблицей DataTables (сортировка, фильтрация, поиск),
    - ссылками на USGS и Google Maps,
    - кнопкой экспорта CSV,
    - минификацией HTML (если установлен htmlmin).
    Возвращает путь к временному HTML-файлу.
    """
    if df.empty:
        return None

    # Подготовка данных
    df_sorted = df.sort_values('time')
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # ========== 1. КАРТА (Folium) ==========
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )

    # Добавляем маркеры
    for _, row in df.iterrows():
        mag = row['mag']
        # Цвет в зависимости от магнитуды
        if mag >= 6:
            color = 'red'
        elif mag >= 5:
            color = 'orange'
        elif mag >= 4:
            color = 'yellow'
        else:
            color = 'green'
        radius = mag ** 2  # чтобы сильные выделялись
        popup = f"""
        <b>{row['place']}</b><br>
        Магнитуда: {mag}<br>
        Глубина: {row['depth']} км<br>
        Время: {row['time']}
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1
        ).add_to(m)

    # Тепловой слой (плотность землетрясений)
    heat_data = [[row['latitude'], row['longitude'], row['mag']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)

    # Если есть LST-данные – добавляем слой с температурой (можно как тепловую карту или маркер)
    if lst_data and lst_data.get('lst_celsius') is not None:
        # Добавляем маркер с температурой в центре
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Средняя LST: {lst_data['lst_celsius']}°C",
            icon=folium.Icon(color='purple', icon='thermometer', prefix='fa')
        ).add_to(m)
        # Можно также добавить слой с растром, но это сложнее – опустим

    # Легенда (используем CSS-блок)
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
    trace = go.Scatter(
        x=df_sorted['time'],
        y=df_sorted['mag'],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        name='Магнитуда'
    )
    layout = go.Layout(
        title=f'{region_name} – динамика магнитуд за последние {period_days} дней',
        xaxis_title='Дата',
        yaxis_title='Магнитуда (M)',
        hovermode='closest',
        template='plotly_white'
    )
    fig = go.Figure(data=[trace], layout=layout)
    plotly_div = plot(fig, output_type='div', include_plotlyjs='cdn')

    # ========== 3. ТАБЛИЦА (HTML + DataTables) ==========
    # Подготовка данных для таблицы (только нужные колонки)
    table_data = df_sorted[['time', 'place', 'mag', 'depth']].copy()
    # Преобразуем в HTML-таблицу с классом для DataTables
    table_html = table_data.to_html(classes='table table-striped table-bordered', index=False, table_id='quakes-table')

    # ========== 4. ССЫЛКИ НА ВНЕШНИЕ ИСТОЧНИКИ ==========
    # Ссылка на USGS (поиск по региону)
    usgs_url = f"https://earthquake.usgs.gov/earthquakes/search/#%7B%22starttime%22%3A%22{df['time'].min()}%22%2C%22endtime%22%3A%22{df['time'].max()}%22%2C%22minmagnitude%22%3A0%2C%22region%22%3A%22{region_name}%22%7D"
    google_maps_url = f"https://www.google.com/maps/@{center_lat},{center_lon},5z"

    # ========== 5. КНОПКИ ЭКСПОРТА (CSV и печать) ==========
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

    # ========== 6. СБОРКА HTML ==========
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
        <!-- Font Awesome для иконок -->
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
                <p><strong>Максимальная магнитуда:</strong> {df['mag'].max():.1f}</p>
                <p><strong>Средняя магнитуда:</strong> {df['mag'].mean():.2f}</p>
                {f'<p><strong>Средняя температура поверхности (LST):</strong> {lst_data["lst_celsius"]:.2f}°C</p>' if lst_data and lst_data.get('lst_celsius') else ''}
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

        <!-- jQuery, Bootstrap, DataTables -->
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

    # Минификация HTML (если доступна)
    try:
        html_minified = minify(html_template, remove_empty_space=True, remove_comments=True)
    except:
        html_minified = html_template

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_minified)
        return f.name
