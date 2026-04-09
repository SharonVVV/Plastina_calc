"""
UI вкладки «Турбулентный режим» — расчёт габаритов пластины
для гарантированного турбулентного режима.
"""

import streamlit as st

import numpy as np
import plotly.graph_objects as go

from turbulence_solver import find_x_crit, find_I_for_turbulence, _grpr_at_x, _solve_tc_at_x
from config import DEFAULTS_TURBULENCE
from formatting import _fc, _fe


def render_turbulence_tab(params: dict) -> None:
    """Отрисовка вкладки анализа турбулентного режима."""

    st.markdown("""
    Этот модуль определяет, при какой высоте пластины (или при каком токе)
    свободная конвекция переходит в **турбулентный режим** (GrPr > 10⁹).
    """)

    b_m = float(params['b_mm']) / 1000.0
    L_m = float(params['L_mm']) / 1000.0

    # ─── Прямая задача ───
    st.subheader('Прямая задача: найти x_крит')
    st.markdown('При текущих параметрах нагревателя — на какой высоте GrPr достигает 10⁹?')

    if st.button('Найти x_крит', key='find_xcrit'):
        with st.spinner('Поиск критической координаты...'):
            res = find_x_crit(
                I=float(params['I']),
                R20=float(params['R20']),
                alpha_R=float(params['alpha_R']),
                b_m=b_m,
                t_fluid_C=float(params['t_fluid_C']),
                g=float(params['g']),
                C_pr=float(params['C_pr']),
                P_Pa=float(params['P_Pa']),
                L_m=L_m,
                x_max=DEFAULTS_TURBULENCE['x_max_search_mm'] / 1000.0,
            )
        st.session_state['turb_result'] = res

    res = st.session_state.get('turb_result')
    if res is not None:
        if res.found:
            col1, col2, col3 = st.columns(3)
            col1.metric('x_крит', f'{_fc(res.x_crit_m * 1000, 1)} мм')
            col2.metric('t_c при x_крит', f'{_fc(res.t_c_crit, 1)} °C')
            col3.metric('GrPr при x_крит', _fe(res.GrPr_crit))

            if res.L_sufficient:
                st.success(
                    f'Пластина L = {_fc(L_m * 1000, 0)} мм **достаточна**. '
                    f'Запас: {_fc(res.margin_m * 1000, 0)} мм '
                    f'({_fc(res.margin_m / L_m * 100, 0)}% высоты).'
                )
            else:
                st.error(
                    f'Пластина L = {_fc(L_m * 1000, 0)} мм **недостаточна**. '
                    f'Нужно минимум {_fc(res.x_crit_m * 1000, 0)} мм.'
                )
        else:
            st.warning(res.message)

        # График GrPr(x) — считаем напрямую через ту же функцию,
        # что использует find_x_crit (ламинарный баланс для каждого x)
        st.subheader('График GrPr(x)')
        with st.spinner('Построение графика...'):
            x_plot_max = max(
                (res.x_crit_m * 1.5 if res.found else L_m),
                L_m,
            )
            x_min_m = float(params['x_min_mm']) / 1000.0
            x_arr = np.linspace(x_min_m, x_plot_max, 60)
            grpr_arr = []
            common_args = (
                float(params['t_fluid_C']), float(params['P_Pa']),
                float(params['g']), float(params['I']),
                float(params['R20']), b_m,
                float(params['alpha_R']), float(params['C_pr']),
            )
            for x in x_arr:
                try:
                    grpr_arr.append(_grpr_at_x(x, *common_args))
                except Exception:
                    grpr_arr.append(float('nan'))

            x_mm = (x_arr * 1000).tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_mm, y=grpr_arr, name='GrPr (ламинарный баланс)',
                line=dict(color='darkblue', width=2),
                mode='lines+markers', marker=dict(size=4),
                hovertemplate='x = %{x:.1f} мм<br>GrPr = %{y:.3e}<extra></extra>',
            ))

            # Порог 10⁹
            fig.add_trace(go.Scatter(
                x=[x_mm[0], x_mm[-1]], y=[1e9, 1e9],
                name='GrPr = 10⁹', mode='lines',
                line=dict(color='red', width=2, dash='dash'),
            ))

            # Маркер x_crit
            if res.found:
                finite_vals = [g for g in grpr_arr if g == g]  # отфильтровать NaN
                y_lo = min(finite_vals) if finite_vals else 1e2
                y_hi = max(max(finite_vals), 1e9) * 2 if finite_vals else 1e10
                fig.add_trace(go.Scatter(
                    x=[res.x_crit_m * 1000, res.x_crit_m * 1000],
                    y=[y_lo, y_hi],
                    name=f'x_крит = {res.x_crit_m*1000:.0f} мм',
                    mode='lines',
                    line=dict(color='green', width=2, dash='dot'),
                ))

            fig.update_layout(
                title='GrPr(x) — при ламинарном балансе (до переключения режима)',
                xaxis_title='x, мм',
                yaxis_title='GrPr',
                yaxis_type='log',
                template='plotly_white',
                height=450,
                separators=', ',
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── Обратная задача ───
    st.markdown('---')
    st.subheader('Обратная задача: минимальный ток для турбулентности')
    st.markdown(
        f'Какой минимальный ток нужен, чтобы на пластине L = {_fc(L_m * 1000, 0)} мм '
        f'возник турбулентный режим?'
    )

    if st.button('Найти I_крит', key='find_icrit'):
        with st.spinner('Поиск минимального тока...'):
            I_crit = find_I_for_turbulence(
                L_m=L_m,
                R20=float(params['R20']),
                alpha_R=float(params['alpha_R']),
                b_m=b_m,
                t_fluid_C=float(params['t_fluid_C']),
                g=float(params['g']),
                C_pr=float(params['C_pr']),
                P_Pa=float(params['P_Pa']),
                I_min=DEFAULTS_TURBULENCE['I_search_min'],
                I_max=DEFAULTS_TURBULENCE['I_search_max'],
            )
        st.session_state['I_crit'] = I_crit

    if 'I_crit' in st.session_state:
        I_crit = st.session_state['I_crit']
        if I_crit is not None:
            st.metric('Минимальный ток I_крит', f'{_fc(I_crit, 1)} А')
            if float(params['I']) >= I_crit:
                st.success(f'Текущий ток I = {_fc(float(params["I"]), 0)} А — достаточен.')
            else:
                st.warning(
                    f'Текущий ток I = {_fc(float(params["I"]), 0)} А — недостаточен. '
                    f'Увеличьте до {_fc(I_crit, 1)} А.'
                )
        else:
            st.error(
                f'Турбулентный режим **невозможен** на пластине L = {_fc(L_m * 1000, 0)} мм '
                f'при любом токе. Увеличьте высоту пластины.'
            )
