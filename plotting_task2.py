"""
Визуализация для Задачи №2: реальная пластина с толщиной.

Два графика:
  1. plot_temperature_profile — T(z) вдоль высоты с подсветкой режимов
     конвекции на лицевой грани и горизонтальной линией T_∞.
  2. plot_heat_balance — горизонтальный stacked-bar с распределением
     потерь по граням (конвекция и радиация раздельно).
"""

from __future__ import annotations

import plotly.graph_objects as go

from task2_solver import Task2Result, compute_back_front_temps


FONT_BODY = '"IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, "JetBrains Mono", monospace'

# Цвета зон режима (совпадают с Задачей 1).
ZONE_FILL = {
    'lam':   'rgba(110, 156, 207, 0.18)',
    'trans': 'rgba(168, 199, 122, 0.20)',
    'turb':  'rgba(214, 123,  92, 0.20)',
    'out':   'rgba(155, 152, 145, 0.18)',
}

# Цвета для разных видов потерь — публикационная палитра.
PATH_COLORS = {
    'front_conv': '#3b6da3',
    'front_rad':  '#8a2c2c',
    'sides_conv': '#6fa1d1',
    'sides_rad':  '#b65f5f',
    'top_conv':   '#2c8d52',
    'top_rad':    '#d6a039',
    'bottom_rad': '#7d6a8c',
}


# ── 1. Профиль температуры по высоте ──────────────────────────────────────

def plot_temperature_profile(result: Task2Result) -> go.Figure:
    fig = go.Figure()

    z_mm = [n.z_m * 1000.0 for n in result.nodes]
    T = [n.T_s_C for n in result.nodes]
    regimes = [n.regime for n in result.nodes]

    # ── Фоновые «полосы» режимов вдоль вертикальной оси (по высоте z) ──
    # Группируем подряд идущие точки одного режима.
    seg_start = 0
    cur = regimes[0]
    for i in range(1, len(regimes)):
        if regimes[i] != cur:
            fig.add_shape(
                type='rect', xref='paper', yref='y',
                x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[i],
                fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
                line=dict(width=0),
                layer='below',
            )
            seg_start = i
            cur = regimes[i]
    fig.add_shape(
        type='rect', xref='paper', yref='y',
        x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[-1],
        fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
        line=dict(width=0), layer='below',
    )

    # Линия T_∞.
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=result.t_fluid_C, x1=result.t_fluid_C, y0=0, y1=1,
        line=dict(color='rgba(40,40,40,0.4)', width=1, dash='dot'),
    )
    fig.add_annotation(
        x=result.t_fluid_C, y=1.0,
        xref='x', yref='paper',
        yanchor='bottom', xanchor='left',
        text=f' T<sub>∞</sub> = {result.t_fluid_C:.0f} °C',
        showarrow=False,
        font=dict(size=10, color='rgba(40,40,40,0.7)', family=FONT_BODY),
    )

    # Профиль T(z).
    fig.add_trace(go.Scatter(
        x=T, y=z_mm,
        mode='lines',
        line=dict(color='#B23A2C', width=2.4),
        name='T(z)',
        hovertemplate='T = %{x:.2f} °C<br>z = %{y:.0f} мм<extra></extra>',
    ))

    # Маркеры экстремумов.
    i_max = int(max(range(len(T)), key=lambda k: T[k]))
    i_min = int(min(range(len(T)), key=lambda k: T[k]))
    for i, label in ((i_max, 'T_max'), (i_min, 'T_min')):
        fig.add_trace(go.Scatter(
            x=[T[i]], y=[z_mm[i]],
            mode='markers',
            marker=dict(size=8, color='#B23A2C',
                        line=dict(color='white', width=1.5)),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_annotation(
            x=T[i], y=z_mm[i],
            xref='x', yref='y',
            xanchor='left', yanchor='middle',
            xshift=8,
            text=f'<b>{label}</b> = {T[i]:.1f} °C @ z = {z_mm[i]:.0f} мм',
            showarrow=False,
            font=dict(size=10, color='#B23A2C', family=FONT_BODY),
        )

    # Легенда зон режима (отдельные scatter-метки для chips).
    for label, key in [
        ('Ламинарный',  'lam'),
        ('Переходный',  'trans'),
        ('Турбулентный','turb'),
        ('Вне диапазона','out'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=ZONE_FILL[key].replace('0.18','0.55').replace('0.20','0.55'),
                        symbol='square', line=dict(color='rgba(0,0,0,0.4)', width=0.7)),
            name=label, hoverinfo='skip',
        ))

    fig.update_xaxes(
        title=dict(text='<b>Температура T<sub>s</sub>, °C</b>',
                   font=dict(size=11.5, family=FONT_BODY)),
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
    )
    fig.update_yaxes(
        title=dict(text='<b>Координата по высоте z, мм</b>',
                   font=dict(size=11.5, family=FONT_BODY)),
        range=[0, max(z_mm) * 1.02],
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
    )
    fig.update_layout(
        height=600,
        margin=dict(t=70, b=60, l=80, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_BODY),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0.0,
            font=dict(size=10.5, family=FONT_BODY),
            bgcolor='rgba(255,255,255,0)', borderwidth=0,
            itemsizing='constant',
        ),
    )
    return fig


# ── 1b. Профиль с разделением T_back / T_front (Задача 3) ────────────────

def plot_temperature_profile_back_front(result: Task2Result) -> go.Figure:
    """То же что plot_temperature_profile, но две кривые:
    T_front (на лицевой грани, primary) и T_back (на тыльной грани, secondary).
    Подсветка режимов конвекции лицевой грани — на фоне, как и раньше.
    """
    fig = go.Figure()

    z_mm = [n.z_m * 1000.0 for n in result.nodes]
    regimes = [n.regime for n in result.nodes]
    T_back_arr, T_front_arr, dT_through_arr = compute_back_front_temps(result)
    T_back = list(T_back_arr)
    T_front = list(T_front_arr)

    # Фоновые полосы режимов — как в plot_temperature_profile.
    seg_start = 0
    cur = regimes[0]
    for i in range(1, len(regimes)):
        if regimes[i] != cur:
            fig.add_shape(
                type='rect', xref='paper', yref='y',
                x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[i],
                fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
                line=dict(width=0), layer='below',
            )
            seg_start = i
            cur = regimes[i]
    fig.add_shape(
        type='rect', xref='paper', yref='y',
        x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[-1],
        fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
        line=dict(width=0), layer='below',
    )

    # Линия T_∞.
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=result.t_fluid_C, x1=result.t_fluid_C, y0=0, y1=1,
        line=dict(color='rgba(40,40,40,0.4)', width=1, dash='dot'),
    )
    fig.add_annotation(
        x=result.t_fluid_C, y=1.0, xref='x', yref='paper',
        yanchor='bottom', xanchor='left',
        text=f' T<sub>∞</sub> = {result.t_fluid_C:.0f} °C',
        showarrow=False,
        font=dict(size=10, color='rgba(40,40,40,0.7)', family=FONT_BODY),
    )

    # T_back — пунктирная серая линия (вторичная).
    fig.add_trace(go.Scatter(
        x=T_back, y=z_mm, mode='lines',
        line=dict(color='rgba(50,50,50,0.55)', width=1.3, dash='dot'),
        name='T<sub>back</sub> (тыльная грань)',
        hovertemplate='T<sub>back</sub> = %{x:.2f} °C<br>z = %{y:.0f} мм<extra></extra>',
    ))

    # T_front — основная красная кривая.
    fig.add_trace(go.Scatter(
        x=T_front, y=z_mm, mode='lines',
        line=dict(color='#B23A2C', width=2.4),
        name='T<sub>front</sub> (лицевая грань)',
        hovertemplate='T<sub>front</sub> = %{x:.2f} °C<br>z = %{y:.0f} мм<extra></extra>',
    ))

    # Маркеры экстремумов T_front.
    i_max = int(max(range(len(T_front)), key=lambda k: T_front[k]))
    i_min = int(min(range(len(T_front)), key=lambda k: T_front[k]))
    for i, label in ((i_max, 'T_front_max'), (i_min, 'T_front_min')):
        fig.add_trace(go.Scatter(
            x=[T_front[i]], y=[z_mm[i]],
            mode='markers',
            marker=dict(size=8, color='#B23A2C',
                        line=dict(color='white', width=1.5)),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_annotation(
            x=T_front[i], y=z_mm[i],
            xref='x', yref='y',
            xanchor='left', yanchor='middle', xshift=8,
            text=f'<b>{label}</b> = {T_front[i]:.1f} °C @ z = {z_mm[i]:.0f} мм',
            showarrow=False,
            font=dict(size=10, color='#B23A2C', family=FONT_BODY),
        )

    # Легенда зон режима.
    for label, key in [
        ('Ламинарный',   'lam'),
        ('Переходный',   'trans'),
        ('Турбулентный', 'turb'),
        ('Вне диапазона','out'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12,
                        color=ZONE_FILL[key].replace('0.18','0.55').replace('0.20','0.55'),
                        symbol='square',
                        line=dict(color='rgba(0,0,0,0.4)', width=0.7)),
            name=label, hoverinfo='skip',
        ))

    fig.update_xaxes(
        title=dict(text='<b>Температура T, °C</b>',
                   font=dict(size=11.5, family=FONT_BODY)),
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
    )
    fig.update_yaxes(
        title=dict(text='<b>Координата по высоте z, мм</b>',
                   font=dict(size=11.5, family=FONT_BODY)),
        range=[0, max(z_mm) * 1.02],
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
    )
    fig.update_layout(
        height=620,
        margin=dict(t=70, b=60, l=80, r=30),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family=FONT_BODY),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0.0,
            font=dict(size=10.5, family=FONT_BODY),
            bgcolor='rgba(255,255,255,0)', borderwidth=0,
            itemsizing='constant',
        ),
    )
    return fig


# ── 1c. Профили α(z) и Ra(z) — для диагностики режима ────────────────────

def plot_alpha_ra_profile(result: Task2Result) -> go.Figure:
    """Два графика рядом: коэффициент теплоотдачи α(z) и число Рэлея Ra(z).
    Высота z по вертикальной оси (общая), значения — по горизонтальной.
    Подсветка режима на фоне (lam/trans/turb/out) — как в основном графике.
    """
    from plotly.subplots import make_subplots

    z_mm = [n.z_m * 1000.0 for n in result.nodes]
    alpha = [n.alpha_front for n in result.nodes]
    Ra = [n.Ra_x for n in result.nodes]
    regimes = [n.regime for n in result.nodes]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(
            '<b>α(z) — коэф. теплоотдачи лицевой грани</b>',
            '<b>Ra(z) — число Рэлея (через ΔT)</b>',
        ),
    )

    # Фоновые полосы режимов — на обеих панелях.
    def _add_zones(col: int):
        seg_start = 0
        cur = regimes[0]
        for i in range(1, len(regimes)):
            if regimes[i] != cur:
                fig.add_shape(
                    type='rect', xref='x domain', yref='y',
                    x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[i],
                    fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
                    line=dict(width=0), layer='below',
                    row=1, col=col,
                )
                seg_start = i
                cur = regimes[i]
        fig.add_shape(
            type='rect', xref='x domain', yref='y',
            x0=0, x1=1, y0=z_mm[seg_start], y1=z_mm[-1],
            fillcolor=ZONE_FILL.get(cur, 'rgba(200,200,200,0.15)'),
            line=dict(width=0), layer='below',
            row=1, col=col,
        )

    _add_zones(1)
    _add_zones(2)

    # Кривая α(z).
    fig.add_trace(
        go.Scatter(
            x=alpha, y=z_mm, mode='lines',
            line=dict(color='#3b6da3', width=2.4),
            name='α', showlegend=False,
            hovertemplate='α = %{x:.2f} Вт/(м²·К)<br>z = %{y:.0f} мм<extra></extra>',
        ),
        row=1, col=1,
    )

    # Кривая Ra(z) — log-шкала.
    fig.add_trace(
        go.Scatter(
            x=Ra, y=z_mm, mode='lines',
            line=dict(color='#8a2c2c', width=2.4),
            name='Ra', showlegend=False,
            hovertemplate='Ra = %{x:.2e}<br>z = %{y:.0f} мм<extra></extra>',
        ),
        row=1, col=2,
    )

    fig.update_xaxes(
        title=dict(text='α, Вт/(м²·К)',
                   font=dict(size=11, family=FONT_BODY)),
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text='Ra (через ΔT)',
                   font=dict(size=11, family=FONT_BODY)),
        type='log',
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text='<b>Координата по высоте z, мм</b>',
                   font=dict(size=11.5, family=FONT_BODY)),
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
        range=[0, max(z_mm) * 1.02],
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor='rgba(220,220,220,0.5)',
        tickfont=dict(family=FONT_MONO, size=10),
        range=[0, max(z_mm) * 1.02],
        row=1, col=2,
    )
    fig.update_layout(
        height=480,
        margin=dict(t=60, b=60, l=80, r=30),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family=FONT_BODY),
        showlegend=False,
    )
    # Subplot titles font.
    for ann in fig.layout.annotations:
        ann.font.family = FONT_BODY
        ann.font.size = 11.5
    return fig


# ── 2. Баланс тепла — горизонтальный stacked-bar ──────────────────────────

def plot_heat_balance(result: Task2Result) -> go.Figure:
    """Один горизонтальный stacked-bar: показывает, как 100% подведённого
    тепла распределяются по путям сброса."""
    Q_in = result.Q_input_W
    if Q_in <= 0:
        Q_in = 1.0  # защита

    paths = [
        ('Лицевая, конвекция',  result.Q_front_conv_W,  PATH_COLORS['front_conv']),
        ('Лицевая, радиация',   result.Q_front_rad_W,   PATH_COLORS['front_rad']),
        ('Боковые, конвекция',  result.Q_sides_conv_W,  PATH_COLORS['sides_conv']),
        ('Боковые, радиация',   result.Q_sides_rad_W,   PATH_COLORS['sides_rad']),
        ('Верх, конвекция',     result.Q_top_conv_W,    PATH_COLORS['top_conv']),
        ('Верх, радиация',      result.Q_top_rad_W,     PATH_COLORS['top_rad']),
        ('Днище, радиация',     result.Q_bottom_rad_W,  PATH_COLORS['bottom_rad']),
    ]

    fig = go.Figure()
    for label, Q, color in paths:
        pct = 100.0 * Q / Q_in
        fig.add_trace(go.Bar(
            x=[Q], y=['Распределение Q'],
            orientation='h',
            name=f'{label} · {Q:.1f} Вт · {pct:.1f}%',
            marker=dict(color=color, line=dict(width=0)),
            text=[f'{pct:.1f}%' if pct >= 3 else ''],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(family=FONT_BODY, size=11, color='white'),
            hovertemplate=f'<b>{label}</b><br>Q = {Q:.2f} Вт<br>{pct:.2f}%<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        height=210,
        margin=dict(t=20, b=60, l=20, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_BODY),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.4,
            xanchor='left', x=0.0,
            font=dict(size=10.5, family=FONT_BODY),
            bgcolor='rgba(255,255,255,0)', borderwidth=0,
            itemsizing='constant',
            traceorder='normal',
        ),
        xaxis=dict(
            title=dict(text='Тепловой поток, Вт',
                       font=dict(size=11, family=FONT_BODY)),
            tickfont=dict(family=FONT_MONO, size=10),
            gridcolor='rgba(220,220,220,0.5)',
        ),
        yaxis=dict(visible=False),
    )
    # Подпись «Σ = ...»
    fig.add_annotation(
        x=Q_in, y=0,
        xref='x', yref='paper',
        yanchor='top', xanchor='right',
        yshift=-4,
        text=f'Σ = <b><span style="font-family:{FONT_MONO}">'
             f'{Q_in:.1f}</span></b> Вт (вход)',
        showarrow=False,
        font=dict(size=10.5, color='rgba(40,40,40,0.85)', family=FONT_BODY),
    )
    return fig
