"""
Визуализация для Задачи №1: пластина с цветными полосами по методикам.

Композиция figure (сверху вниз):
  1. Горизонтальная легенда зон (4 цветных чипа).
  2. Полосы плиты с зонами lam / trans / turb / out и контуром цвета методики.
     Внутри полос — пунктир lam→trans и тире начала turb; целевая высота — красным.
  3. Подписи методик ПОД плитой (короткое имя, цвет методики).
  4. Под именем — координаты переходов lam→trans / turb (моноширинно, в столбик).
  5. Внизу — компактная подпись геометрии b × L.
"""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from task1_solver import MethodologyResult


# ── Палитра зон ─────────────────────────────────────────────────────────────
ZONE_COLORS = {
    'lam':   'rgba(110, 156, 207, 0.55)',
    'trans': 'rgba(168, 199, 122, 0.55)',
    'turb':  'rgba(214, 123,  92, 0.55)',
    'out':   'rgba(155, 152, 145, 0.40)',
}

TARGET_COLOR = '#B23A2C'
LINE_TRANS = 'rgba(40,40,40,0.70)'
LINE_TURB  = 'rgba(20,20,20,0.95)'

FONT_BODY = '"IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, "JetBrains Mono", monospace'


def _zones_from_points(pts) -> list[tuple[float, float, str]]:
    if not pts:
        return []
    zones = []
    cur = pts[0].regime
    y0 = pts[0].x_m * 1000.0
    for i in range(1, len(pts)):
        y = pts[i].x_m * 1000.0
        if pts[i].regime != cur:
            zones.append((y0, y, cur))
            cur = pts[i].regime
            y0 = y
    zones.append((y0, pts[-1].x_m * 1000.0, cur))
    return zones


def plot_methodology_strips(
    width_mm: float,
    height_mm: float,
    results: Sequence[MethodologyResult],
    target_fraction: float = 0.8,
) -> go.Figure:
    fig = go.Figure()
    n = len(results)
    if n == 0:
        fig.add_annotation(text='Нет данных', showarrow=False, font=dict(size=14))
        return fig

    pad_left, pad_right = 0.05, 0.05
    strip_w = (1.0 - pad_left - pad_right) / n
    gap = strip_w * 0.26
    bar_w = strip_w - gap

    y_max = height_mm
    y_target = target_fraction * height_mm

    # ── Легенда: только 4 зоны (квадратные chip-маркеры) ───────────────────
    legend_zone_traces = [
        ('Ламинарный',  ZONE_COLORS['lam']),
        ('Переходный',  ZONE_COLORS['trans']),
        ('Турбулентный', ZONE_COLORS['turb']),
        ('Вне диапазона применимости', ZONE_COLORS['out']),
    ]
    for label, color in legend_zone_traces:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=13, color=color, symbol='square',
                        line=dict(color='rgba(0,0,0,0.45)', width=0.8)),
            name=label, hoverinfo='skip',
        ))

    # ── Полосы по методикам ────────────────────────────────────────────────
    for i, mr in enumerate(results):
        x_center = pad_left + (i + 0.5) * strip_w
        x0 = x_center - bar_w / 2
        x1 = x_center + bar_w / 2

        converged = [p for p in mr.points if p.converged]
        zones = _zones_from_points(converged)

        # Зоны.
        for zy0, zy1, regime in zones:
            color = ZONE_COLORS.get(regime, 'rgba(180,180,180,0.35)')
            fig.add_shape(
                type='rect', xref='paper', yref='y',
                x0=x0, x1=x1, y0=zy0, y1=zy1,
                fillcolor=color, line=dict(width=0),
            )

        # Тонкий контур плиты в цвете методики.
        fig.add_shape(
            type='rect', xref='paper', yref='y',
            x0=x0, x1=x1, y0=0, y1=y_max,
            fillcolor='rgba(0,0,0,0)',
            line=dict(color=mr.meta.color, width=1.4),
        )

        # Линии переходов внутри полосы (без правых подписей-координат — они
        # выводятся под полосой компактно).
        is_stitch = (
            mr.x_lam_to_trans_m is not None
            and mr.x_trans_to_turb_m is not None
            and abs(mr.x_lam_to_trans_m - mr.x_trans_to_turb_m) < 1e-9
        )
        if mr.x_lam_to_trans_m is not None and not is_stitch:
            y = mr.x_lam_to_trans_m * 1000.0
            fig.add_shape(
                type='line', xref='paper', yref='y',
                x0=x0, x1=x1, y0=y, y1=y,
                line=dict(color=LINE_TRANS, width=1.1, dash='dot'),
            )
        if mr.x_trans_to_turb_m is not None:
            y = mr.x_trans_to_turb_m * 1000.0
            fig.add_shape(
                type='line', xref='paper', yref='y',
                x0=x0, x1=x1, y0=y, y1=y,
                line=dict(color=LINE_TURB, width=1.6, dash='dash'),
            )

        # ── Подпись методики ПОД полосой ───────────────────────────────────
        fig.add_annotation(
            x=x_center, y=-0.025,
            xref='paper', yref='paper',
            xanchor='center', yanchor='top',
            text=f'<b>{mr.meta.name_short}</b>',
            showarrow=False,
            font=dict(size=12, color=mr.meta.color, family=FONT_BODY),
        )

        # ── Координаты переходов: каждая строка отдельным annotation ──────
        # (вместо <br>, чтобы выравнивание было идеальным; уменьшаем шанс
        # горизонтального наезда — строки узкие, помещаются под полосой).
        coord_y_first  = -0.075
        coord_y_second = -0.115

        if is_stitch:
            y = mr.x_trans_to_turb_m * 1000.0
            fig.add_annotation(
                x=x_center, y=coord_y_first,
                xref='paper', yref='paper',
                xanchor='center', yanchor='top',
                text='<b>lam→turb</b>',
                showarrow=False,
                font=dict(size=9, color='rgba(40,40,40,0.85)', family=FONT_BODY),
            )
            fig.add_annotation(
                x=x_center, y=coord_y_second,
                xref='paper', yref='paper',
                xanchor='center', yanchor='top',
                text=f'<b>{y:.0f}</b> мм',
                showarrow=False,
                font=dict(size=10, color='rgba(20,20,20,0.95)', family=FONT_MONO),
            )
        else:
            if mr.x_lam_to_trans_m is not None:
                y = mr.x_lam_to_trans_m * 1000.0
                fig.add_annotation(
                    x=x_center, y=coord_y_first,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='top',
                    text=f'lam→t · <span style="font-family:{FONT_MONO}">{y:.0f}</span>',
                    showarrow=False,
                    font=dict(size=9, color='rgba(60,60,60,0.85)', family=FONT_BODY),
                )
            if mr.x_trans_to_turb_m is not None:
                y = mr.x_trans_to_turb_m * 1000.0
                fig.add_annotation(
                    x=x_center, y=coord_y_second,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='top',
                    text=f'<b>turb</b> · <span style="font-family:{FONT_MONO}">{y:.0f}</span>',
                    showarrow=False,
                    font=dict(size=9.5, color='rgba(20,20,20,0.95)', family=FONT_BODY),
                )
            if mr.x_lam_to_trans_m is None and mr.x_trans_to_turb_m is None:
                note = 'только лам.' if mr.meta.turb_min is None else 'не достиг.'
                fig.add_annotation(
                    x=x_center, y=coord_y_first,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='top',
                    text=f'<i>{note}</i>',
                    showarrow=False,
                    font=dict(size=9, color='rgba(80,80,80,0.85)', family=FONT_BODY),
                )

    # (Целевая горизонтальная линия 0.8·L намеренно не отображается на
    # диаграмме; значение target_fraction используется только в обратной
    # задаче «минимальный I/N для турбулента на target·L» в таблице ниже.)

    # ── Геометрия плиты ────────────────────────────────────────────────────
    fig.add_annotation(
        x=0.5, y=-0.18,
        xref='paper', yref='paper',
        xanchor='center', yanchor='top',
        text=(f'b = <span style="font-family:{FONT_MONO}">{width_mm:.0f}</span> мм '
              f'· L = <span style="font-family:{FONT_MONO}">{height_mm:.0f}</span> мм'),
        showarrow=False,
        font=dict(size=10.5, color='rgba(60,60,60,0.8)', family=FONT_BODY),
    )

    # ── Оси и общая раскладка ──────────────────────────────────────────────
    fig.update_yaxes(
        title=dict(text='<b>Высота x, мм</b>',
                   font=dict(size=11, color='rgba(40,40,40,0.9)', family=FONT_BODY)),
        range=[0, y_max * 1.03],
        zeroline=False,
        gridcolor='rgba(220,220,220,0.50)',
        tickfont=dict(family=FONT_MONO, size=10, color='rgba(40,40,40,0.75)'),
        ticks='outside', ticklen=4, tickcolor='rgba(80,80,80,0.6)',
    )
    fig.update_xaxes(visible=False, range=[0, 1])

    fig.update_layout(
        height=680,
        margin=dict(t=80, b=160, l=70, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_BODY, color='rgba(30,30,30,0.95)'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0.0,
            font=dict(size=10.5, family=FONT_BODY, color='rgba(30,30,30,0.9)'),
            bgcolor='rgba(255,255,255,0.0)',
            borderwidth=0,
            itemsizing='constant',
            traceorder='normal',
        ),
    )
    return fig
