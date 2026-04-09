"""
Графики Plotly для визуализации результатов расчёта.
"""

from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver import CalculationResult


def _russian_separators(fig: go.Figure) -> go.Figure:
    """Установить десятичную запятую в Plotly."""
    fig.update_layout(separators=', ')
    return fig


def plot_t_vs_x(result: CalculationResult) -> go.Figure:
    """График температуры поверхности t_c от координаты x."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]
    t_c = [p.t_c for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_mm, y=t_c, name='t_c, °C',
        line=dict(color='red', width=2),
        mode='lines+markers', marker=dict(size=5),
        hovertemplate='x = %{x:.1f} мм<br>t_c = %{y:.2f} °C<extra></extra>',
    ))

    fig.update_layout(
        title='Температура поверхности t_c(x)',
        xaxis_title='x, мм',
        yaxis_title='t_c, °C',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_ra_vs_x(result: CalculationResult) -> go.Figure:
    """График числа Рэлея Ra от координаты x (лог. шкала Y)."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]
    Ra = [p.Ra for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_mm, y=Ra, name='Ra_x',
        line=dict(color='coral', width=2),
        mode='lines+markers', marker=dict(size=5),
        hovertemplate='x = %{x:.1f} мм<br>Ra = %{y:.3e}<extra></extra>',
    ))

    # Линия Ra = 1e9
    fig.add_hline(
        y=1e9, line_dash='dash', line_color='gray',
        annotation_text='Ra = 10⁹ (граница турбулентности)',
        annotation_position='top left',
    )

    fig.update_layout(
        title='Число Рэлея Ra(x)',
        xaxis_title='x, мм',
        yaxis_title='Ra',
        yaxis_type='log',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_alpha_comparison(result: CalculationResult) -> go.Figure:
    """Сравнение alpha по методичке и по Черчиллю–Чу."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]

    is_cc = result.correlation == 'churchill_chu'
    name_main = 'Черчилль–Чу (полная)' if is_cc else 'По методичке (кусочная)'
    name_alt = 'По методичке (кусочная)' if is_cc else 'Черчилль–Чу (полная)'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.alpha for p in pts],
        name=name_main,
        line=dict(color='blue', width=2),
        mode='lines+markers', marker=dict(size=4),
        hovertemplate='x = %{x:.1f} мм<br>α = %{y:.2f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.alpha_alt for p in pts],
        name=name_alt,
        line=dict(color='red', width=2, dash='dash'),
        mode='lines+markers', marker=dict(size=4),
        hovertemplate='x = %{x:.1f} мм<br>α_alt = %{y:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title='Сравнение α: методичка vs Черчилль–Чу',
        xaxis_title='x, мм',
        yaxis_title='α, Вт/(м²·К)',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_nu_comparison(result: CalculationResult) -> go.Figure:
    """Сравнение Nu по методичке и по Черчиллю–Чу."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]

    is_cc = result.correlation == 'churchill_chu'
    name_main = 'Черчилль–Чу (полная)' if is_cc else 'По методичке (кусочная)'
    name_alt = 'По методичке (кусочная)' if is_cc else 'Черчилль–Чу (полная)'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.Nu for p in pts],
        name=name_main,
        line=dict(color='blue', width=2),
        hovertemplate='x = %{x:.1f} мм<br>Nu = %{y:.2f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.Nu_alt for p in pts],
        name=name_alt,
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='x = %{x:.1f} мм<br>Nu_alt = %{y:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title='Сравнение Nu: методичка vs Черчилль–Чу',
        xaxis_title='x, мм',
        yaxis_title='Nu',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_main_2d(result: CalculationResult) -> go.Figure:
    """
    Основной двухосевой 2D-график: t_c и alpha от x.
    Левая ось — температура (красная), правая — коэфф. теплоотдачи (синий).
    Жёлтая зона где Ra < 1e9.
    """
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]
    t_c = [p.t_c for p in pts]
    alpha = [p.alpha for p in pts]

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Зоны режимов
    lam_x = [p.x_m * 1000 for p in pts if p.regime == 'lam']
    turb_x = [p.x_m * 1000 for p in pts if p.regime == 'turb']
    if lam_x:
        fig.add_vrect(
            x0=min(lam_x), x1=max(lam_x),
            fillcolor='lightblue', opacity=0.15,
            line_width=0,
            annotation_text='Ламинарный',
            annotation_position='top left',
        )
    if turb_x:
        fig.add_vrect(
            x0=min(turb_x), x1=max(turb_x),
            fillcolor='lightsalmon', opacity=0.15,
            line_width=0,
            annotation_text='Турбулентный',
            annotation_position='top right',
        )

    fig.add_trace(
        go.Scatter(
            x=x_mm, y=t_c, name='t_c, °C',
            line=dict(color='red', width=2),
            mode='lines+markers', marker=dict(size=4),
            hovertemplate='x = %{x:.1f} мм<br>t_c = %{y:.2f} °C<extra></extra>',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x_mm, y=alpha, name='α, Вт/(м²·К)',
            line=dict(color='blue', width=2),
            mode='lines+markers', marker=dict(size=4),
            hovertemplate='x = %{x:.1f} мм<br>α = %{y:.2f} Вт/(м²·К)<extra></extra>',
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text='x, мм', gridcolor='lightgray')
    fig.update_yaxes(title_text='Температура поверхности t_c, °C',
                     secondary_y=False, title_font=dict(color='red'),
                     gridcolor='lightgray')
    fig.update_yaxes(title_text='Коэфф. теплоотдачи α, Вт/(м²·К)',
                     secondary_y=True, title_font=dict(color='blue'))
    fig.update_layout(
        title='Распределение t_c и α по высоте пластины',
        legend=dict(x=0.5, y=1.12, xanchor='center', orientation='h'),
        template='plotly_white',
        height=500,
    )

    return _russian_separators(fig)


def plot_heat_fluxes(result: CalculationResult) -> go.Figure:
    """Три компоненты теплового потока: q_эл, q_конв, q_рад от x."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.q_el for p in pts], name='q_эл',
        line=dict(color='orange', width=2),
        hovertemplate='x = %{x:.1f} мм<br>q_эл = %{y:.1f} Вт/м²<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.q_conv for p in pts], name='q_конв',
        line=dict(color='green', width=2),
        hovertemplate='x = %{x:.1f} мм<br>q_конв = %{y:.1f} Вт/м²<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.q_rad for p in pts], name='q_рад',
        line=dict(color='purple', width=2),
        hovertemplate='x = %{x:.1f} мм<br>q_рад = %{y:.1f} Вт/м²<extra></extra>',
    ))

    fig.update_layout(
        title='Компоненты теплового потока',
        xaxis_title='x, мм',
        yaxis_title='q, Вт/м²',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_criteria(result: CalculationResult) -> go.Figure:
    """Nu(x) и Ra(x) — безразмерные критерии (Ra на лог. шкале)."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(
        go.Scatter(
            x=x_mm, y=[p.Nu for p in pts], name='Nu_x',
            line=dict(color='teal', width=2),
            hovertemplate='x = %{x:.1f} мм<br>Nu = %{y:.2f}<extra></extra>',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x_mm, y=[p.Ra for p in pts], name='Ra_x',
            line=dict(color='coral', width=2),
            hovertemplate='x = %{x:.1f} мм<br>Ra = %{y:.3e}<extra></extra>',
        ),
        secondary_y=True,
    )

    # Линия Ra = 1e9
    fig.add_hline(y=1e9, line_dash='dash', line_color='gray',
                  annotation_text='Ra = 10⁹', secondary_y=True)

    fig.update_xaxes(title_text='x, мм')
    fig.update_yaxes(title_text='Nu_x', secondary_y=False)
    fig.update_yaxes(title_text='Ra_x', secondary_y=True, type='log')
    fig.update_layout(
        title='Критерии подобия Nu и Ra',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)


def plot_air_props(result: CalculationResult) -> go.Figure:
    """Распределение свойств воздуха (ν, λ, Pr) по высоте."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('ν, м²/с', 'λ, Вт/(м·К)', 'Pr'))

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.air.nu for p in pts], name='ν',
        line=dict(color='steelblue', width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.air.lam for p in pts], name='λ',
        line=dict(color='firebrick', width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_mm, y=[p.air.Pr for p in pts], name='Pr',
        line=dict(color='darkgreen', width=2),
    ), row=3, col=1)

    fig.update_xaxes(title_text='x, мм', row=3, col=1)
    fig.update_layout(
        title='Свойства воздуха при плёночной температуре',
        template='plotly_white',
        height=700,
        showlegend=False,
    )

    return _russian_separators(fig)


def plot_3d_surface(
    results: List[CalculationResult],
    param_name: str,
    param_values: List[float],
) -> go.Figure:
    """
    3D-поверхность t_c(x, param).

    Args:
        results: список CalculationResult для каждого значения параметра
        param_name: название параметра (для подписи оси)
        param_values: значения варьируемого параметра
    """
    # Собрать матрицу температур
    n_param = len(results)
    n_x = len(results[0].points)

    x_mm = np.array([p.x_m * 1000 for p in results[0].points])
    param_arr = np.array(param_values)

    Z = np.zeros((n_param, n_x))
    for i, res in enumerate(results):
        for j, pt in enumerate(res.points):
            Z[i, j] = pt.t_c if pt.converged else float('nan')

    X, Y = np.meshgrid(x_mm, param_arr)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Turbo',
        colorbar=dict(title='t_c, °C'),
        hovertemplate=(
            'x = %{x:.1f} мм<br>'
            + param_name + ' = %{y:.2f}<br>'
            + 't_c = %{z:.2f} °C<extra></extra>'
        ),
    ))

    fig.update_layout(
        title=f'Параметрическое исследование: t_c(x, {param_name})',
        scene=dict(
            xaxis_title='x, мм',
            yaxis_title=param_name,
            zaxis_title='t_c, °C',
        ),
        template='plotly_white',
        height=600,
    )

    return _russian_separators(fig)


def plot_grpr_vs_x(result: CalculationResult, x_crit_m: float = None) -> go.Figure:
    """GrPr(x) с линией порога 10⁹ и маркером x_crit."""
    pts = [p for p in result.points if p.converged]
    x_mm = [p.x_m * 1000 for p in pts]
    grpr = [p.GrPr for p in pts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_mm, y=grpr, name='GrPr',
        line=dict(color='darkblue', width=2),
        mode='lines+markers', marker=dict(size=4),
        hovertemplate='x = %{x:.1f} мм<br>GrPr = %{y:.3e}<extra></extra>',
    ))

    # Горизонтальная линия GrPr = 10⁹ (явный trace для корректности на лог. шкале)
    x_range = [min(x_mm), max(x_mm)]
    fig.add_trace(go.Scatter(
        x=x_range, y=[1e9, 1e9],
        name='GrPr = 10⁹',
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Граница турбулентности: GrPr = 10⁹<extra></extra>',
    ))

    if x_crit_m is not None:
        # Вертикальная линия x_crit (явный trace)
        y_min = min(grpr) if grpr else 1e2
        y_max = max(max(grpr), 1e9) * 2 if grpr else 1e10
        fig.add_trace(go.Scatter(
            x=[x_crit_m * 1000, x_crit_m * 1000],
            y=[y_min, y_max],
            name=f'x_крит = {x_crit_m*1000:.0f} мм',
            mode='lines',
            line=dict(color='green', width=2, dash='dot'),
            hovertemplate=f'x_крит = {x_crit_m*1000:.0f} мм<extra></extra>',
        ))

    fig.update_layout(
        title='GrPr(x) — определение режима течения',
        xaxis_title='x, мм',
        yaxis_title='GrPr',
        yaxis_type='log',
        template='plotly_white',
        height=450,
    )

    return _russian_separators(fig)
