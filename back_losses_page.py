"""
Вкладка «Потери через тыл и торцы» — визуализация и расчёт тепловых потерь
сэндвич-конструкции.

Использует модель из back_losses.py:
    • тыльный путь — кондукция через ПИР+сталь, конвекция+излучение наружу;
    • торцевые потери — для каждого из 4 слоёв (АМг3, нагреватель, ПИР, сталь)
      по периметру панели, со своей средней температурой и степенью черноты.

Графики:
    1. Поперечное сечение сэндвича с пропорциональными толщинами,
       подсвеченными температурами на границах слоёв.
    2. Профиль T(x) сквозь пакет.
    3. Декомпозиция теплового баланса (stacked bar): Q_useful + Q_back + Q_edges.
    4. Декомпозиция торцевых потерь по слоям.
    5. Кривые η(q_w) — тыл, торцы, итого.
"""

from copy import deepcopy
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from back_losses import (
    Layer, SandwichConfig, RegimeResult, EdgeLoss,
    EdgeProfile, ProfileLoss,
    default_config, solve_all_regimes,
    DEFAULT_REGIMES, SIGMA0,
)
from formatting import _fc, _fe


# ── Цветовая схема ─────────────────────────────────────────────────────────
_REGIME_COLORS = ['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6']


# ── Графики ────────────────────────────────────────────────────────────────


def plot_cross_section(cfg: SandwichConfig, result: RegimeResult) -> go.Figure:
    """Поперечное сечение сэндвича с пропорциональными толщинами слоёв.

    Если включён профиль — над пакетом дорисовывается полоска прокладки
    и алюминиевого профиля по всей длине, с подписью T_profile."""
    fig = go.Figure()
    x_left = 0.0
    annotations = []
    pack_width_mm = cfg.total_thickness * 1000.0

    profile_active = (cfg.profile.enabled and result is not None
                      and result.profile_loss is not None)

    for i, layer in enumerate(cfg.layers):
        width = layer.thickness * 1000.0  # мм
        x_right = x_left + width
        fig.add_shape(
            type='rect',
            x0=x_left, x1=x_right, y0=0, y1=1,
            fillcolor=layer.color, line=dict(color='black', width=1.5),
            layer='below',
        )
        label = f'<b>{layer.name}</b><br>{width:.1f} мм'
        annotations.append(dict(
            x=(x_left + x_right) / 2, y=0.5,
            text=label, showarrow=False,
            font=dict(size=11, color='black'),
            xanchor='center', yanchor='middle',
        ))
        x_left = x_right

    # Температуры на границах слоёв
    if result is not None and result.T_interfaces_C:
        x_pos = 0.0
        for i, T in enumerate(result.T_interfaces_C):
            annotations.append(dict(
                x=x_pos, y=1.05,
                text=f'<b>{T:.1f} °C</b>',
                showarrow=False, font=dict(size=10, color='#2c3e50'),
                xanchor='center', yanchor='bottom',
            ))
            if i > 0:
                x_pos += cfg.layers[i - 1].thickness * 1000.0

    # Если профиль включён — рисуем рамку над пакетом
    y_max = 1.3
    if profile_active:
        d_break_mm = cfg.profile.delta_break * 1000.0
        d_prof_mm = cfg.profile.delta_profile * 1000.0
        # Прокладка (МБОР-5Ф)
        fig.add_shape(
            type='rect', x0=0, x1=pack_width_mm,
            y0=1.35, y1=1.35 + 0.20,
            fillcolor='#bdc3c7', line=dict(color='black', width=1),
        )
        annotations.append(dict(
            x=pack_width_mm / 2, y=1.45,
            text=f'<b>МБОР-5Ф</b> · {d_break_mm:.1f} мм '
                 f'(λ={cfg.profile.lambda_break})',
            showarrow=False, font=dict(size=10),
            xanchor='center', yanchor='middle',
        ))
        # Профиль AД31
        fig.add_shape(
            type='rect', x0=0, x1=pack_width_mm,
            y0=1.58, y1=1.58 + 0.20,
            fillcolor='#95a5a6', line=dict(color='black', width=1),
        )
        annotations.append(dict(
            x=pack_width_mm / 2, y=1.68,
            text=f'<b>AД31</b> · {d_prof_mm:.1f} мм '
                 f'(ε={cfg.profile.eps_profile})',
            showarrow=False, font=dict(size=10, color='white'),
            xanchor='center', yanchor='middle',
        ))
        # Подписи температур по краям рамки
        p = result.profile_loss
        annotations.append(dict(
            x=-2, y=1.46,
            text=f'<b>T_внутр<br>{p.T_inner_weighted_C:.0f} °C</b>',
            showarrow=False, font=dict(size=10, color='#c0392b'),
            xanchor='right', yanchor='middle',
        ))
        annotations.append(dict(
            x=pack_width_mm + 2, y=1.68,
            text=f'<b>T_проф<br>{p.T_profile_C:.0f} °C</b>',
            showarrow=False, font=dict(size=10, color='#9b59b6'),
            xanchor='left', yanchor='middle',
        ))
        annotations.append(dict(
            x=pack_width_mm / 2, y=1.92,
            text=f'↓ конвекция + излучение → t∞ = {cfg.t_inf_C:.0f} °C',
            showarrow=False, font=dict(size=10, color='gray'),
            xanchor='center', yanchor='middle',
        ))
        y_max = 2.0

    # Подписи "лицо" / "тыл"
    annotations.append(dict(
        x=-2, y=0.5, text='← лицо<br>(t_face)',
        showarrow=False, font=dict(size=11, color='#c0392b'),
        xanchor='right', yanchor='middle',
    ))
    if result is not None:
        annotations.append(dict(
            x=pack_width_mm + 2, y=0.5,
            text=f'тыл →<br>(T_steel = {result.T_steel_outer_C:.1f} °C)',
            showarrow=False, font=dict(size=11, color='#2980b9'),
            xanchor='left', yanchor='middle',
        ))

    title = 'Поперечное сечение сэндвича'
    if profile_active:
        title += ' + алюминиевая рамка с термическим разрывом'

    fig.update_layout(
        title=title,
        xaxis=dict(title='Толщина, мм',
                   range=[-14, pack_width_mm + 16],
                   showgrid=False),
        yaxis=dict(visible=False, range=[-0.15, y_max]),
        annotations=annotations,
        height=380 if profile_active else 300,
        template='plotly_white',
        margin=dict(l=20, r=20, t=70, b=50),
        separators=', ',
    )
    return fig


def plot_temperature_profile(cfg: SandwichConfig,
                             results: List[RegimeResult]) -> go.Figure:
    """Профиль T(x) сквозь пакет для всех режимов."""
    fig = go.Figure()
    x_iface_mm = [0.0]
    cum = 0.0
    for L in cfg.layers:
        cum += L.thickness * 1000.0
        x_iface_mm.append(cum)

    for r, color in zip(results, _REGIME_COLORS):
        if len(r.T_interfaces_C) != len(x_iface_mm):
            continue
        fig.add_trace(go.Scatter(
            x=x_iface_mm, y=r.T_interfaces_C,
            mode='lines+markers', name=f'q_w = {r.q_w:.0f} Вт/м²',
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
            hovertemplate='x = %{x:.1f} мм<br>T = %{y:.1f} °C<extra></extra>',
        ))
        # Пунктирная линия от наружной грани стали до t∞
        x_far = cum + 30.0
        fig.add_trace(go.Scatter(
            x=[cum, x_far], y=[r.T_interfaces_C[-1], cfg.t_inf_C],
            mode='lines', line=dict(color=color, width=1.5, dash='dot'),
            showlegend=False,
            hovertemplate='конвекция/излучение<extra></extra>',
        ))

    # Разметка границ слоёв
    cum = 0.0
    for L in cfg.layers:
        cum += L.thickness * 1000.0
        fig.add_vline(x=cum, line=dict(color='lightgray', width=1, dash='dash'))

    fig.add_hline(y=cfg.t_inf_C, line=dict(color='gray', width=1, dash='dot'),
                  annotation_text=f't∞ = {cfg.t_inf_C:.0f} °C',
                  annotation_position='right')

    fig.update_layout(
        title='Температура по толщине сэндвича',
        xaxis_title='Глубина от лицевой стороны, мм',
        yaxis_title='Температура, °C',
        template='plotly_white',
        height=420,
        separators=', ',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
    )
    return fig


def plot_heat_balance(results: List[RegimeResult],
                      profile_active: bool) -> go.Figure:
    """Stacked bar: Q_useful + Q_back + Q_edges/Q_profile по режимам."""
    labels = [f'q_w={r.q_w:.0f}' for r in results]
    Q_useful = [r.Q_useful_W for r in results]
    Q_back = [r.Q_back_W for r in results]
    Q_edges = [r.Q_edges_effective_W for r in results]

    edges_label = 'Q_profile (рамка)' if profile_active else 'Q_edges (торцы)'
    edges_color = '#9b59b6' if profile_active else '#e74c3c'

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=Q_useful, name='Q_useful (полезная)',
        marker_color='#27ae60',
        text=[f'{v:.0f} Вт' for v in Q_useful], textposition='inside',
        hovertemplate='%{y:.1f} Вт<extra>Q_useful</extra>',
    ))
    fig.add_trace(go.Bar(
        x=labels, y=Q_back, name='Q_back (тыл)',
        marker_color='#3498db',
        text=[f'{v:.1f}' for v in Q_back], textposition='inside',
        hovertemplate='%{y:.2f} Вт<extra>Q_back</extra>',
    ))
    fig.add_trace(go.Bar(
        x=labels, y=Q_edges, name=edges_label,
        marker_color=edges_color,
        text=[f'{v:.1f}' for v in Q_edges], textposition='inside',
        hovertemplate='%{y:.2f} Вт<extra>'+edges_label+'</extra>',
    ))

    title = 'Баланс электрической мощности' + (' (с профилем)' if profile_active
                                                 else ' (голые торцы)')
    fig.update_layout(
        title=title,
        xaxis_title='Режим (q_w, Вт/м²)',
        yaxis_title='Q, Вт',
        barmode='stack',
        template='plotly_white',
        height=400,
        separators=', ',
    )
    return fig


def plot_profile_effect(results: List[RegimeResult]) -> go.Figure:
    """Сравнение Q_edges голых vs Q_profile (рамка) по режимам."""
    labels = [f'q_w={r.q_w:.0f}' for r in results]
    Q_bare = [r.Q_edges_bare_W for r in results]
    Q_prof = [r.profile_loss.Q_profile_W if r.profile_loss else 0.0
              for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=Q_bare, name='Голые торцы',
        marker_color='#e74c3c',
        text=[f'{v:.1f}' for v in Q_bare], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        x=labels, y=Q_prof, name='С профилем (МБОР+AД31)',
        marker_color='#9b59b6',
        text=[f'{v:.1f}' for v in Q_prof], textposition='outside',
    ))
    fig.update_layout(
        title='Эффект алюминиевого профиля с термическим разрывом',
        xaxis_title='Режим (q_w, Вт/м²)',
        yaxis_title='Q, Вт',
        barmode='group',
        template='plotly_white',
        height=400,
        separators=', ',
    )
    return fig


def plot_profile_temps(results: List[RegimeResult]) -> go.Figure:
    """Температуры в цепочке: t_face → T_inner_avg → T_profile → t_inf."""
    labels = [f'q_w={r.q_w:.0f}' for r in results]
    t_face = [r.t_face_C for r in results]
    T_inner = [r.profile_loss.T_inner_weighted_C if r.profile_loss
               else float('nan') for r in results]
    T_prof = [r.profile_loss.T_profile_C if r.profile_loss
              else float('nan') for r in results]
    t_inf = [results[0].t_face_C * 0 + 20.0 for _ in results]  # placeholder

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=t_face, name='t_face (АМг3)',
        mode='lines+markers+text',
        text=[f'{v:.0f}' for v in t_face], textposition='top center',
        line=dict(color='#c0392b', width=2),
        marker=dict(size=12),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=T_inner, name='T_inner_avg (внутр. рамки)',
        mode='lines+markers+text',
        text=[f'{v:.0f}' for v in T_inner], textposition='top center',
        line=dict(color='#f39c12', width=2),
        marker=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=T_prof, name='T_profile (внешн. рамки)',
        mode='lines+markers+text',
        text=[f'{v:.0f}' for v in T_prof], textposition='bottom center',
        line=dict(color='#9b59b6', width=2),
        marker=dict(size=10),
    ))
    if results and results[0].profile_loss is not None:
        # горизонталь t_inf
        fig.add_hline(y=20.0, line=dict(color='gray', dash='dot'),
                      annotation_text='t∞ = 20 °C',
                      annotation_position='right')

    fig.update_layout(
        title='Падение температуры через прокладку и профиль',
        xaxis_title='Режим (q_w, Вт/м²)',
        yaxis_title='Температура, °C',
        template='plotly_white',
        height=380,
        separators=', ',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
    )
    return fig


def plot_edge_breakdown(results: List[RegimeResult]) -> go.Figure:
    """Stacked bar: декомпозиция Q_edges по слоям."""
    if not results or not results[0].edges:
        return go.Figure()

    layer_names = [e.layer_name for e in results[0].edges]
    labels = [f'q_w={r.q_w:.0f}' for r in results]

    fig = go.Figure()
    palette = ['#a8b2c0', '#e8a87c', '#f1c40f', '#7f8c8d', '#9b59b6', '#1abc9c']
    for i, lname in enumerate(layer_names):
        ys = [r.edges[i].Q_edge_W for r in results]
        fig.add_trace(go.Bar(
            x=labels, y=ys, name=lname,
            marker_color=palette[i % len(palette)],
            text=[f'{v:.1f}' for v in ys], textposition='inside',
            hovertemplate='%{y:.2f} Вт<extra>'+lname+'</extra>',
        ))

    fig.update_layout(
        title='Декомпозиция торцевых потерь по слоям',
        xaxis_title='Режим (q_w, Вт/м²)',
        yaxis_title='Q_edge, Вт',
        barmode='stack',
        template='plotly_white',
        height=400,
        separators=', ',
    )
    return fig


def plot_eta_curves(results: List[RegimeResult]) -> go.Figure:
    """η(q_w) — тыл, торцы, итого."""
    q_w = [r.q_w for r in results]
    eta_back = [r.eta_back_pct for r in results]
    eta_edges = [r.eta_edges_pct for r in results]
    eta_total = [r.eta_total_pct for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q_w, y=eta_total, name='η_total (итого)',
        mode='lines+markers', line=dict(color='#e74c3c', width=3),
        marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=q_w, y=eta_edges, name='η_edges (торцы)',
        mode='lines+markers', line=dict(color='#f39c12', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=q_w, y=eta_back, name='η_back (тыл)',
        mode='lines+markers', line=dict(color='#3498db', width=2),
    ))

    fig.add_hrect(y0=5, y1=10, fillcolor='lightgreen', opacity=0.18,
                  line_width=0, annotation_text='Целевой диапазон 5–10 %',
                  annotation_position='top left')
    fig.add_hline(y=15, line=dict(color='red', width=1, dash='dash'),
                  annotation_text='Потолок 15 %', annotation_position='right')

    fig.update_layout(
        title='Доля потерь η от полной электрической мощности',
        xaxis_title='q_w, Вт/м²',
        yaxis_title='η, %',
        template='plotly_white',
        height=400,
        separators=', ',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
    )
    return fig


def plot_q_curves(results: List[RegimeResult]) -> go.Figure:
    """Удельные/полные потери vs q_w."""
    q_w = [r.q_w for r in results]
    Q_back = [r.Q_back_W for r in results]
    Q_edges = [r.Q_edges_W for r in results]
    Q_total = [r.Q_loss_W for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q_w, y=Q_total, name='Q_loss (полные)',
        mode='lines+markers', line=dict(color='#e74c3c', width=3),
        marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=q_w, y=Q_edges, name='Q_edges (торцы)',
        mode='lines+markers', line=dict(color='#f39c12', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=q_w, y=Q_back, name='Q_back (тыл)',
        mode='lines+markers', line=dict(color='#3498db', width=2),
    ))

    fig.update_layout(
        title='Тепловые потери в абсолютных единицах',
        xaxis_title='q_w, Вт/м²',
        yaxis_title='Q, Вт',
        template='plotly_white',
        height=380,
        separators=', ',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
    )
    return fig


# ── Таблицы ────────────────────────────────────────────────────────────────


def build_summary_table(results: List[RegimeResult],
                        profile_active: bool) -> pd.DataFrame:
    rows = []
    edges_col = 'Q_profile, Вт' if profile_active else 'Q_edges, Вт'
    for r in results:
        rows.append({
            '#': r.regime_idx,
            'q_w, Вт/м²': _fc(r.q_w, 0),
            't_face, °C': _fc(r.t_face_C, 1),
            'T_steel, °C': _fc(r.T_steel_outer_C, 2),
            'q_back, Вт/м²': _fc(r.q_back_Wm2, 2),
            'Q_back, Вт': _fc(r.Q_back_W, 2),
            edges_col: _fc(r.Q_edges_effective_W, 2),
            'Q_loss, Вт': _fc(r.Q_loss_W, 2),
            'Q_useful, Вт': _fc(r.Q_useful_W, 1),
            'Q_электр, Вт': _fc(r.Q_electr_W, 1),
            'η_back, %': _fc(r.eta_back_pct, 2),
            'η_edges, %': _fc(r.eta_edges_pct, 2),
            'η_total, %': _fc(r.eta_total_pct, 2),
        })
    return pd.DataFrame(rows)


def build_profile_table(results: List[RegimeResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.profile_loss is None:
            continue
        p = r.profile_loss
        rows.append({
            '#': r.regime_idx,
            'q_w, Вт/м²': _fc(r.q_w, 0),
            't_face, °C': _fc(r.t_face_C, 1),
            'T_inner_avg, °C': _fc(p.T_inner_weighted_C, 2),
            'T_profile, °C': _fc(p.T_profile_C, 2),
            'q_through, Вт/м²': _fc(p.q_through_Wm2, 2),
            'q_conv, Вт/м²': _fc(p.q_conv_Wm2, 2),
            'q_rad, Вт/м²': _fc(p.q_rad_Wm2, 2),
            'A_proΦ, м²': _fc(p.A_profile_m2, 4),
            'Q_profile, Вт': _fc(p.Q_profile_W, 2),
        })
    return pd.DataFrame(rows)


def build_edge_table(result: RegimeResult) -> pd.DataFrame:
    rows = []
    for e in result.edges:
        rows.append({
            'Слой': e.layer_name,
            'T_avg, °C': _fc(e.T_avg_C, 2),
            'A_edge, м²': _fc(e.A_edge_m2, 4),
            'α, Вт/(м²·К)': _fc(e.alpha, 2),
            'q_conv, Вт/м²': _fc(e.q_conv_Wm2, 1),
            'q_rad, Вт/м²': _fc(e.q_rad_Wm2, 1),
            'Q_edge, Вт': _fc(e.Q_edge_W, 3),
        })
    return pd.DataFrame(rows)


# ── UI вкладки ─────────────────────────────────────────────────────────────


def render_back_losses_tab(ui_params: dict) -> None:
    st.markdown('### Тепловые потери через тыл и торцы сэндвича')
    st.caption(
        'Расчёт потерь через тыльную поверхность (кондукция через ПИР+сталь, '
        'конвекция и излучение наружу) и через торцы пакета (для каждого слоя — '
        'свой средний температурный уровень). Корреляция Черчилля–Чу.'
    )

    cfg = default_config()
    col_in, col_viz = st.columns([1, 2], gap='large')

    # ── Левая колонка: параметры ────────────────────────────────────────────
    with col_in:
        st.markdown('##### Геометрия')
        c1, c2 = st.columns(2)
        with c1:
            L_height = st.number_input(
                'L (высота), мм', value=int(cfg.L_height * 1000),
                min_value=100, max_value=10000, step=50, key='bl_L',
            ) / 1000.0
        with c2:
            b_width = st.number_input(
                'b (ширина), мм', value=int(cfg.b_width * 1000),
                min_value=20, max_value=2000, step=10, key='bl_b',
            ) / 1000.0

        st.markdown('##### Слои сэндвича (лицо → тыл)')
        layers_new: List[Layer] = []
        for i, L in enumerate(cfg.layers):
            with st.expander(f'{i + 1}. {L.name}',
                             expanded=(L.name == 'ПИР')):
                c1, c2 = st.columns(2)
                with c1:
                    delta_mm = st.number_input(
                        'Толщина, мм', value=L.thickness * 1000.0,
                        min_value=0.1, max_value=200.0, step=0.5,
                        key=f'bl_d_{i}', format='%.1f',
                    )
                with c2:
                    lam = st.number_input(
                        'λ, Вт/(м·К)', value=L.lam,
                        min_value=0.005, max_value=500.0,
                        step=0.001 if L.lam < 1 else 1.0,
                        format='%.4f' if L.lam < 1 else '%.1f',
                        key=f'bl_lam_{i}',
                    )
                eps_e = st.slider(
                    'ε торца', min_value=0.05, max_value=0.99,
                    value=L.eps_edge, step=0.01, key=f'bl_eps_{i}',
                )
                layers_new.append(Layer(
                    name=L.name, thickness=delta_mm / 1000.0,
                    lam=lam, eps_edge=eps_e, color=L.color,
                ))

        st.markdown('##### Среда и тыл')
        c1, c2 = st.columns(2)
        with c1:
            t_inf = st.number_input(
                't∞, °C', value=cfg.t_inf_C, min_value=0.0, max_value=50.0,
                step=1.0, key='bl_t_inf',
            )
        with c2:
            P_Pa = st.number_input(
                'P, Па', value=int(cfg.P_Pa), min_value=80000, max_value=110000,
                step=100, key='bl_P',
            )
        eps_back = st.slider(
            'ε внешней стороны стали', min_value=0.05, max_value=0.99,
            value=cfg.eps_back, step=0.01, key='bl_eps_back',
        )
        include_edges = st.checkbox(
            'Учитывать торцевые потери', value=True, key='bl_inc_edges',
        )

        # ── Алюминиевый профиль с термическим разрывом ─────────────────────
        st.markdown('##### Алюминиевый профиль с термическим разрывом')
        profile_enabled = st.checkbox(
            'Включить рамку из профиля (МБОР-5Ф + AД31)',
            value=False, key='bl_profile_on',
            help='Замыкает голые торцы прокладкой и алюминиевым профилем. '
                 'Резко снижает η_edges за счёт термического разрыва.',
        )
        with st.expander('Параметры прокладки + профиля',
                         expanded=profile_enabled):
            st.caption('Прокладка (МБОР-5Ф фольгированный)')
            c1, c2 = st.columns(2)
            with c1:
                delta_break_mm = st.number_input(
                    'δ прокладки, мм', value=5.0,
                    min_value=0.5, max_value=50.0, step=0.5,
                    key='bl_d_break',
                )
            with c2:
                lambda_break = st.number_input(
                    'λ прокладки, Вт/(м·К)', value=0.045,
                    min_value=0.005, max_value=1.0, step=0.001,
                    format='%.3f', key='bl_lam_break',
                    help='λ зависит от температуры: 0.033 при 25 °C, '
                         '~0.045 при 80–100 °C (консервативно).',
                )
            eps_break_outer = st.slider(
                'ε фольгированной стороны (информ.)',
                min_value=0.02, max_value=0.99,
                value=0.05, step=0.01, key='bl_eps_break',
                help='Применимо если между прокладкой и профилем есть '
                     'воздушный зазор. При плотном контакте не влияет.',
            )

            st.caption('Профиль (AД31 анодированный)')
            c1, c2 = st.columns(2)
            with c1:
                delta_profile_mm = st.number_input(
                    'δ стенки профиля, мм', value=2.0,
                    min_value=0.5, max_value=10.0, step=0.5,
                    key='bl_d_prof',
                )
            with c2:
                lambda_profile = st.number_input(
                    'λ профиля, Вт/(м·К)', value=200.0,
                    min_value=10.0, max_value=400.0, step=10.0,
                    key='bl_lam_prof',
                )
            eps_profile = st.slider(
                'ε внешней поверхности профиля',
                min_value=0.04, max_value=0.99,
                value=0.25, step=0.01, key='bl_eps_prof',
                help='Анодированный натуральный ~0.25; mill-finish ~0.10–0.15; '
                     'полированный ~0.05; чёрный анодированный ~0.85 (плохо!).',
            )

        st.markdown('##### Режимы (q_w, Вт/м² → t_face, °C)')
        st.caption('Введите до 6 режимов. Пустые строки игнорируются.')
        # Редактируемая таблица режимов
        df_init = pd.DataFrame(
            [{'q_w, Вт/м²': q, 't_face, °C': t} for q, t in DEFAULT_REGIMES],
        )
        df_edit = st.data_editor(
            df_init, num_rows='dynamic', use_container_width=True,
            key='bl_regimes', hide_index=True,
        )

    # ── Сборка конфигурации и расчёт ───────────────────────────────────────
    profile_cfg = EdgeProfile(
        enabled=profile_enabled,
        delta_break=delta_break_mm / 1000.0,
        lambda_break=float(lambda_break),
        eps_break_outer=float(eps_break_outer),
        delta_profile=delta_profile_mm / 1000.0,
        lambda_profile=float(lambda_profile),
        eps_profile=float(eps_profile),
    )
    cfg_run = SandwichConfig(
        layers=layers_new,
        back_path_indices=cfg.back_path_indices,
        L_height=L_height, b_width=b_width,
        eps_back=eps_back,
        t_inf_C=float(t_inf), P_Pa=float(P_Pa),
        g=cfg.g, include_edges=include_edges,
        profile=profile_cfg,
    )

    # Парсим режимы
    regimes = []
    try:
        for _, row in df_edit.iterrows():
            q = row['q_w, Вт/м²']
            t = row['t_face, °C']
            if pd.isna(q) or pd.isna(t):
                continue
            q, t = float(q), float(t)
            if t > cfg_run.t_inf_C + 0.5 and q > 0:
                regimes.append((q, t))
    except Exception as e:
        st.error(f'Ошибка в таблице режимов: {e}')
        return

    if not regimes:
        with col_viz:
            st.info('Задайте хотя бы один режим в таблице слева.')
        return

    try:
        results = solve_all_regimes(regimes, cfg_run)
    except Exception as e:
        with col_viz:
            st.error(f'Ошибка расчёта: {e}')
        return

    # ── Правая колонка: визуализация ───────────────────────────────────────
    with col_viz:
        # Метрики верхнего режима
        r_top = results[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Q_back (макс. режим)', f'{r_top.Q_back_W:.1f} Вт',
                  f'η = {r_top.eta_back_pct:.2f} %')
        m2.metric('Q_edges (макс. режим)', f'{r_top.Q_edges_W:.1f} Вт',
                  f'η = {r_top.eta_edges_pct:.2f} %')
        m3.metric('Q_loss (итого)', f'{r_top.Q_loss_W:.1f} Вт',
                  f'η = {r_top.eta_total_pct:.2f} %')
        m4.metric('T_steel наружу', f'{r_top.T_steel_outer_C:.1f} °C')

        # Поперечное сечение
        sel_idx = st.select_slider(
            'Режим для иллюстрации сечения / профиля T(x)',
            options=list(range(len(results))),
            value=len(results) - 1,
            format_func=lambda i: f'q_w = {results[i].q_w:.0f} Вт/м²',
            key='bl_regime_pick',
        )
        st.plotly_chart(plot_cross_section(cfg_run, results[sel_idx]),
                        use_container_width=True)

        # Профиль T(x)
        st.plotly_chart(plot_temperature_profile(cfg_run, results),
                        use_container_width=True)

    # ── Нижняя секция: разбор и графики ────────────────────────────────────
    st.markdown('---')
    st.markdown(
        f'#### Сводная таблица '
        f'({"с профилем (МБОР+AД31)" if profile_enabled else "голые торцы"})'
    )
    st.dataframe(build_summary_table(results, profile_enabled),
                 use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_heat_balance(results, profile_enabled),
                        use_container_width=True)
    with c2:
        st.plotly_chart(plot_edge_breakdown(results), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_eta_curves(results), use_container_width=True)
    with c2:
        st.plotly_chart(plot_q_curves(results), use_container_width=True)

    # Блок состояния профиля
    if profile_enabled and any(r.profile_loss for r in results):
        st.markdown('---')
        st.markdown('#### Алюминиевый профиль с термическим разрывом')

        # Метрики для верхнего режима
        p_top = r_top.profile_loss
        if p_top is not None:
            ratio = (r_top.Q_edges_bare_W / p_top.Q_profile_W
                     if p_top.Q_profile_W > 0 else float('inf'))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric('T_profile (внешн.)', f'{p_top.T_profile_C:.1f} °C',
                       f'из t_face = {r_top.t_face_C:.0f} °C')
            mc2.metric('Q_profile (макс. режим)', f'{p_top.Q_profile_W:.1f} Вт',
                       f'было: {r_top.Q_edges_bare_W:.1f} Вт')
            mc3.metric('Снижение потерь', f'×{ratio:.2f}',
                       f'−{(1 - 1/ratio)*100:.0f} %' if ratio > 0 else '')
            mc4.metric('R_cond профиля', f'{cfg_run.profile.R_cond:.4f}',
                       'м²·К/Вт')

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_profile_effect(results),
                            use_container_width=True)
        with c2:
            st.plotly_chart(plot_profile_temps(results),
                            use_container_width=True)

        st.dataframe(build_profile_table(results),
                     use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown(f'#### Декомпозиция торцов как голых '
                f'(режим q_w = {r_top.q_w:.0f} Вт/м²)')
    st.caption('Что было бы, если торцы оставить открытыми — для сравнения.')
    st.dataframe(build_edge_table(r_top),
                 use_container_width=True, hide_index=True)

    # ── Диагностика ────────────────────────────────────────────────────────
    eta_back_max = max(r.eta_back_pct for r in results)
    eta_edges_max = max(r.eta_edges_pct for r in results)
    eta_total_max = max(r.eta_total_pct for r in results)
    eta_total_min = min(r.eta_total_pct for r in results)

    st.markdown('#### Заключение')
    if not include_edges:
        if eta_back_max <= 10.0 and eta_back_max >= 5.0:
            st.success(
                f'**Тыл:** η_back = {min(r.eta_back_pct for r in results):.2f}…'
                f'{eta_back_max:.2f} % — в целевом диапазоне 5–10 %.'
            )
        elif eta_back_max < 5.0:
            st.success(
                f'**Тыл:** η_back = {min(r.eta_back_pct for r in results):.2f}…'
                f'{eta_back_max:.2f} % — существенный запас.'
            )
        else:
            st.warning(
                f'**Тыл:** η_back = {eta_back_max:.2f} % > 10 % — увеличьте δ_ПИР.'
            )
    elif profile_enabled and any(r.profile_loss for r in results):
        # Когда включён профиль — отдельная диагностика
        Q_bare_top = r_top.Q_edges_bare_W
        Q_prof_top = r_top.profile_loss.Q_profile_W if r_top.profile_loss else 0.0
        ratio = Q_bare_top / Q_prof_top if Q_prof_top > 0 else float('inf')
        st.info(
            f'**Эффект профиля (МБОР-5Ф {cfg_run.profile.delta_break*1000:.1f} мм + '
            f'AД31 ε={cfg_run.profile.eps_profile}):** '
            f'торцевые потери снижены с {Q_bare_top:.1f} Вт до {Q_prof_top:.1f} Вт '
            f'(в {ratio:.2f} раза). T_profile у внешней поверхности ≈ '
            f'{r_top.profile_loss.T_profile_C:.1f} °C при t_face = '
            f'{r_top.t_face_C:.0f} °C.'
        )
        if eta_total_max > 15.0:
            st.warning(
                f'**Полные потери с профилем:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % > 15 %. Дополнительные меры: '
                f'увеличить δ_прокладки до 8–10 мм; снизить ε_профиля '
                f'(полированный AД31 → ε≈0.05).'
            )
        elif eta_total_max > 10.0:
            st.warning(
                f'**Полные потери с профилем:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % — на верхней границе целевого диапазона.'
            )
        elif eta_total_max >= 5.0:
            st.success(
                f'**Полные потери с профилем:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % — в целевом диапазоне 5–10 %. ✓'
            )
        else:
            st.success(
                f'**Полные потери с профилем:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % < 5 % — отличный результат.'
            )
    else:
        if eta_edges_max > 2.0 * eta_back_max:
            st.warning(
                f'⚠ **Торцевые потери доминируют:** η_edges = '
                f'{min(r.eta_edges_pct for r in results):.1f}…'
                f'{eta_edges_max:.1f} % vs η_back = '
                f'{min(r.eta_back_pct for r in results):.2f}…'
                f'{eta_back_max:.2f} %. '
                f'Это типично для узких пластин (b={cfg_run.b_width*1000:.0f} мм '
                f'при L={cfg_run.L_height*1000:.0f} мм даёт периметр '
                f'{cfg_run.perimeter:.2f} м и площадь торцов '
                f'{sum(L.thickness for L in cfg_run.layers) * cfg_run.perimeter:.4f} м² — '
                f'{(sum(L.thickness for L in cfg_run.layers) * cfg_run.perimeter / cfg_run.A_face) * 100:.0f} % '
                f'от лицевой площади).'
            )
            st.info(
                '**Рекомендация для ВКР:** обернуть торцы пакета слоем ПИР '
                'или фольгированной изоляции толщиной 10–20 мм, либо понизить '
                'степень черноты торцов (фольга, ε ≈ 0.05) — это снизит '
                'η_edges в 5–10 раз.'
            )
        if eta_total_max > 15.0:
            st.error(
                f'**Полные потери:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % > 15 % — конструкция требует '
                f'доработки.'
            )
        elif eta_total_max > 10.0:
            st.warning(
                f'**Полные потери:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % — на верхней границе нормы.'
            )
        elif eta_total_max >= 5.0:
            st.success(
                f'**Полные потери:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % — целевой диапазон 5–10 %.'
            )
        else:
            st.success(
                f'**Полные потери:** η_total = {eta_total_min:.1f}…'
                f'{eta_total_max:.1f} % < 5 % — существенный запас.'
            )
