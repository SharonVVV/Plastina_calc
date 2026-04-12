"""
UI вкладки «Сравнение методик» — сравнение высот переходов режимов
для 8 корреляций свободной конвекции у вертикальной пластины.

Основная цель: показать, на какой высоте x каждая методика
предсказывает переход ламинарный → переходный → турбулентный.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver import solve_plate, CalculationResult
from formatting import _fc, _fe


# ── Описания всех 8 методик ─────────────────────────────────────────────────

METHOD_INFO = {
    'kerimov': {
        'name': 'Керимов (методичка)',
        'short': 'Керимов',
        'formula_lam': r'Nu = 0{,}60 \cdot (GrPr)^{0{,}25} \cdot \varepsilon_t',
        'formula_turb': r'Nu = 0{,}15 \cdot (GrPr)^{1/3} \cdot \varepsilon_t',
        'ref_T': 't_ж',
        'crits': 'GrPr: 10⁹ → 6·10¹⁰',
        'color': '#1f77b4',
        'dash': None,
    },
    'kuznetov': {
        'name': 'Кузнецов (q=const)',
        'short': 'Кузнецов',
        'formula_lam': r'Nu = 0{,}563 \cdot [Ra \cdot \Phi(Pr)]^{0{,}25}',
        'formula_turb': r'Nu = 0{,}15 \cdot [Ra \cdot \Phi(Pr)]^{1/3}',
        'ref_T': 't_пл',
        'crits': 'Ra: 10⁹ → 10¹²',
        'color': '#ff7f0e',
        'dash': None,
    },
    'churchill_chu': {
        'name': 'Черчилль–Чу (средний)',
        'short': 'Ч-Ч',
        'formula_lam': r'\overline{Nu} = \left[0{,}825 + 0{,}387 (Ra \cdot \Psi)^{1/6}\right]^2',
        'formula_turb': None,
        'ref_T': 't_пл',
        'crits': 'весь диапазон',
        'color': '#2ca02c',
        'dash': None,
    },
    'leontiev': {
        'name': 'Леонтьев (Брдлик / Эккерт-Дж.)',
        'short': 'Леонтьев',
        'formula_lam': r'Nu = 0{,}616 (Pr/(Pr+0{,}8))^{1/5} (Gr^* Pr)^{1/5}',
        'formula_turb': r'Nu = 0{,}0295 \cdot Ra^{2/5} \cdot Pr^{1/15} \cdot (1+0{,}494 Pr^{2/3})^{-2/5}',
        'ref_T': 't_пл',
        'crits': 'Ra: 2·10⁷',
        'color': '#d62728',
        'dash': 'dash',
    },
    'churchill_ozoe': {
        'name': 'Churchill & Ozoe (1973)',
        'short': 'Ch-Ozoe',
        'formula_lam': r'Nu = 0{,}563 \cdot Ra^{1/4} / [1+(0{,}437/Pr)^{9/16}]^{4/9}',
        'formula_turb': None,
        'ref_T': 't_пл',
        'crits': 'только лам. (Ra < 10⁹)',
        'color': '#9467bd',
        'dash': 'dash',
    },
    'vliet': {
        'name': 'Vliet (1969/1975)',
        'short': 'Vliet',
        'formula_lam': r'Nu = 0{,}60 \cdot (Ra^*)^{0{,}2}',
        'formula_turb': r'Nu = 0{,}568 \cdot (Ra^*)^{0{,}22}',
        'ref_T': 't_пл',
        'crits': 'Ra*: 10¹³ (≈ Ra 3.4·10⁹)',
        'color': '#8c564b',
        'dash': 'dash',
    },
    'fujii': {
        'name': 'Fujii & Fujii (1976)',
        'short': 'Fujii',
        'formula_lam': r'Nu = (Gr^* Pr^2 / (4+9\sqrt{Pr}+10Pr))^{1/5}',
        'formula_turb': None,
        'ref_T': 't_пл',
        'crits': 'только ламинарный',
        'color': '#e377c2',
        'dash': 'dash',
    },
    'isachenko': {
        'name': 'Исаченко и др. (1981)',
        'short': 'Исаченко',
        'formula_lam': r'Nu = 0{,}60 \cdot Ra^{0{,}25} \cdot \varepsilon_t',
        'formula_turb': r'Nu = 0{,}15 \cdot Ra^{1/3} \cdot \varepsilon_t',
        'ref_T': 't_ж',
        'crits': 'Ra: 10⁹ → 6·10¹⁰',
        'color': '#7f7f7f',
        'dash': 'dash',
    },
}

# Границы режимов по умолчанию для каждой методики
_DEFAULT_CRITS = {
    'kerimov': (1e9, 6e10),
    'kuznetov': (1e9, 1e12),
    'churchill_chu': (1e9, 1e12),
    'leontiev': (2e7, 2e7),
    'churchill_ozoe': (1e9, 1e9),
    'vliet': (3.4e9, 3.4e9),
    'fujii': (1e15, 1e15),  # только ламинарный
    'isachenko': (1e9, 6e10),
}


def _find_regime_transitions(result: CalculationResult):
    """Найти координаты переходов режимов в результате расчёта."""
    pts = [p for p in result.points if p.converged]
    if not pts:
        return None, None

    x_lam_turb = None  # первая точка, где режим != 'lam'
    x_turb = None      # первая точка, где режим == 'turb'

    for i, p in enumerate(pts):
        if p.regime in ('trans', 'turb', 'full') and x_lam_turb is None:
            x_lam_turb = p.x_m
        if p.regime == 'turb' and x_turb is None:
            x_turb = p.x_m

    return x_lam_turb, x_turb


def _run_all_methods(params, selected_methods, N):
    """Запустить все выбранные методики и собрать результаты."""
    results = {}

    for corr_key in selected_methods:
        c1, c2 = _DEFAULT_CRITS.get(corr_key, (1e9, 6e10))
        try:
            result = solve_plate(
                I=float(params['I']),
                R20=float(params['R20']),
                alpha_R=float(params['alpha_R']),
                b_mm=float(params['b_mm']),
                L_mm=float(params['L_mm']),
                t_fluid_C=float(params['t_fluid_C']),
                g=float(params['g']),
                C_pr=float(params['C_pr']),
                P_Pa=float(params['P_Pa']),
                x_min_mm=float(params['x_min_mm']),
                N=N,
                correlation=corr_key,
                t_ref_mode='auto',
                crit_1=c1,
                crit_2=c2,
            )
            results[corr_key] = result
        except Exception as e:
            st.warning(f'Ошибка в методике {corr_key}: {e}')

    return results


def _build_regime_summary(results: dict) -> pd.DataFrame:
    """Таблица: методика → x_крит(лам→перех) → x_крит(перех→турб)."""
    rows = []
    for corr_key, result in results.items():
        info = METHOD_INFO[corr_key]
        x_trans, x_turb = _find_regime_transitions(result)

        # Считаем точки по режимам
        pts = [p for p in result.points if p.converged]
        n_lam = sum(1 for p in pts if p.regime == 'lam')
        n_trans = sum(1 for p in pts if p.regime == 'trans')
        n_turb = sum(1 for p in pts if p.regime == 'turb')
        n_full = sum(1 for p in pts if p.regime == 'full')

        rows.append({
            'Методика': info['name'],
            'T_опр': info['ref_T'],
            'Критерий': info['crits'],
            'x_крит (лам→), мм': _fc(x_trans * 1000, 1) if x_trans else '—',
            'x_турб, мм': _fc(x_turb * 1000, 1) if x_turb else '—',
            'Лам.': n_lam,
            'Перех.': n_trans + n_full,
            'Турб.': n_turb,
        })
    return pd.DataFrame(rows)


def _build_regime_chart(results: dict, L_mm: float) -> go.Figure:
    """Горизонтальная диаграмма зон режимов с подписями и x_крит."""

    REGIME_COLORS = {
        'lam': 'rgba(46, 134, 222, 0.75)',
        'trans': 'rgba(255, 177, 66, 0.75)',
        'turb': 'rgba(231, 76, 60, 0.75)',
        'full': 'rgba(155, 89, 182, 0.55)',
    }
    REGIME_NAMES = {
        'lam': 'Ламинарный',
        'trans': 'Переходный',
        'turb': 'Турбулентный',
        'full': 'Ч-Ч (единая формула)',
    }

    fig = go.Figure()
    legend_added = set()
    annotations = []

    for corr_key, result in results.items():
        info = METHOD_INFO[corr_key]
        pts = [p for p in result.points if p.converged]
        if not pts:
            continue

        # Собираем зоны по режимам
        zones = []
        current_regime = pts[0].regime
        zone_start = pts[0].x_m * 1000
        for p in pts[1:]:
            if p.regime != current_regime:
                zones.append((current_regime, zone_start, p.x_m * 1000))
                current_regime = p.regime
                zone_start = p.x_m * 1000
        zones.append((current_regime, zone_start, pts[-1].x_m * 1000))

        y_label = info['short']

        for regime, x_start, x_end in zones:
            width = x_end - x_start
            show_legend = regime not in legend_added
            if show_legend:
                legend_added.add(regime)

            fig.add_trace(go.Bar(
                y=[y_label],
                x=[width],
                base=[x_start],
                orientation='h',
                marker_color=REGIME_COLORS.get(regime, 'gray'),
                name=REGIME_NAMES.get(regime, regime),
                showlegend=show_legend,
                legendgroup=regime,
                hovertemplate=(
                    f"<b>{info['name']}</b><br>"
                    f"{REGIME_NAMES.get(regime, regime)}<br>"
                    f"x: {x_start:.0f} – {x_end:.0f} мм<br>"
                    f"Длина зоны: {width:.0f} мм"
                    f"<extra></extra>"
                ),
            ))

            # Подпись внутри зоны: режим + диапазон
            if width > L_mm * 0.08:
                x_mid = x_start + width / 2
                if regime == 'full':
                    label_text = 'единая формула'
                else:
                    label_text = f"{REGIME_NAMES.get(regime, '')[0:3]}. {x_start:.0f}–{x_end:.0f}"
                annotations.append(dict(
                    x=x_mid, y=y_label,
                    text=f"<b>{label_text}</b>",
                    showarrow=False,
                    font=dict(size=11, color='white'),
                    xanchor='center', yanchor='middle',
                ))

        # Вертикальные маркеры x_крит на границах зон
        for i in range(len(zones) - 1):
            x_boundary = zones[i][2]
            from_regime = REGIME_NAMES.get(zones[i][0], '')[:3]
            to_regime = REGIME_NAMES.get(zones[i + 1][0], '')[:3]

            annotations.append(dict(
                x=x_boundary, y=y_label,
                text=f"<b>▸ {x_boundary:.0f}</b>",
                showarrow=False,
                font=dict(size=10, color='#222'),
                xanchor='left', yanchor='bottom',
                yshift=12,
            ))

    fig.update_layout(
        title=dict(
            text='Зоны режимов течения по высоте пластины',
            font=dict(size=16),
        ),
        xaxis_title='Высота x, мм',
        barmode='stack',
        template='plotly_white',
        height=max(350, len(results) * 55 + 120),
        margin=dict(l=130, t=50, b=60, r=30),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.2,
            x=0.5, xanchor='center',
            font=dict(size=13),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc', borderwidth=1,
        ),
        annotations=annotations,
        separators=', ',
        bargap=0.25,
    )
    fig.update_xaxes(range=[0, L_mm * 1.05], ticksuffix=' мм', dtick=200)
    fig.update_yaxes(autorange='reversed')

    return fig


def _build_profiles_chart(results: dict) -> go.Figure:
    """Три графика: Nu(x), alpha(x), T_ст(x) по всем методикам."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Nu<sub>x</sub>(x)', 'α<sub>x</sub>(x), Вт/(м²·К)', 'T<sub>ст</sub>(x), °C'],
        horizontal_spacing=0.06,
    )

    for corr_key, result in results.items():
        info = METHOD_INFO[corr_key]
        pts = [p for p in result.points if p.converged]
        if not pts:
            continue
        x_mm = [p.x_m * 1000 for p in pts]
        nu_arr = [p.Nu for p in pts]
        alpha_arr = [p.alpha for p in pts]
        tc_arr = [p.t_c for p in pts]

        line_style = dict(color=info['color'], width=2)
        if info.get('dash'):
            line_style['dash'] = info['dash']

        fig.add_trace(go.Scatter(
            x=x_mm, y=nu_arr, name=info['short'],
            line=line_style, mode='lines',
            legendgroup=corr_key, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_mm, y=alpha_arr, name=info['short'],
            line=line_style, mode='lines',
            legendgroup=corr_key, showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x_mm, y=tc_arr, name=info['short'],
            line=line_style, mode='lines',
            legendgroup=corr_key, showlegend=False,
        ), row=1, col=3)

    fig.update_xaxes(title_text='x, мм')
    fig.update_layout(
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, x=0.5, xanchor='center'),
        margin=dict(t=40, b=80),
        separators=', ',
    )
    return fig


def render_comparison_tab(params: dict) -> None:
    """Отрисовка вкладки сравнения методик — фокус на высотах переходов режимов."""

    st.markdown("""
    Сравнение **8 методик**: на какой высоте пластины каждая предсказывает
    переход из **ламинарного** в **переходный/турбулентный** режим.
    """)

    # ── Выбор методик ──
    col1, col2 = st.columns([1, 3])

    with col1:
        N_comp = st.number_input('Точек N', value=150, min_value=20, max_value=500,
                                 step=10, key='n_comp')

    with col2:
        all_keys = list(METHOD_INFO.keys())
        selected = st.multiselect(
            'Методики для сравнения',
            options=all_keys,
            default=all_keys,
            format_func=lambda k: METHOD_INFO[k]['name'],
        )

    if not selected:
        st.warning('Выберите хотя бы одну методику.')
        return

    # ── Карточки выбранных методик ──
    with st.expander('Формулы и параметры выбранных методик', expanded=False):
        for key in selected:
            info = METHOD_INFO[key]
            cols = st.columns([2, 3, 2])
            with cols[0]:
                st.markdown(f"**{info['name']}**")
                st.caption(f"T_опр = {info['ref_T']}, {info['crits']}")
            with cols[1]:
                st.latex(info['formula_lam'])
            with cols[2]:
                if info['formula_turb']:
                    st.latex(info['formula_turb'])
                else:
                    st.caption('Только ламинарный')
            st.markdown('---')

    # ── Расчёт ──
    do_compare = st.button('Рассчитать сравнение', type='primary',
                           use_container_width=True, key='btn_compare')

    if do_compare:
        with st.spinner('Расчёт по всем методикам...'):
            results = _run_all_methods(params, selected, N_comp)
        st.session_state['comparison_results'] = results

    results = st.session_state.get('comparison_results')
    if not results:
        return

    # ── 1. Главное: диаграмма зон режимов ──
    st.markdown('#### Зоны режимов течения')
    L_mm = float(params['L_mm'])
    fig_regime = _build_regime_chart(results, L_mm)
    st.plotly_chart(fig_regime, use_container_width=True)

    # ── 2. Таблица критических высот ──
    st.markdown('#### Критические высоты переходов')
    st.caption(
        'x_крит — высота, на которой заканчивается ламинарный режим. '
        'x_турб — высота, начиная с которой течение полностью турбулентное. '
        '«—» означает, что переход не происходит на данной пластине.'
    )
    df_summary = _build_regime_summary(results)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ── 3. Профили Nu, alpha, T_ст ──
    st.markdown('#### Профили вдоль пластины')
    fig_profiles = _build_profiles_chart(results)
    st.plotly_chart(fig_profiles, use_container_width=True)

    # ── CSV ──
    csv = df_summary.to_csv(index=False, sep=';')
    st.download_button('Скачать таблицу CSV', data=csv,
                       file_name='regime_comparison.csv', mime='text/csv',
                       key='dl_regime')
