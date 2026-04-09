"""
Streamlit-приложение: Расчёт теплового баланса вертикальной нагреваемой пластины
(стенд Керимова, НИУ «МЭИ»).
"""

import streamlit as st
import pandas as pd

from config import (
    DEFAULTS_HEATER, RANGES_HEATER,
    DEFAULTS_ENV, RANGES_ENV,
    DEFAULTS_CALC,
)
from solver import solve_plate, CalculationResult
from plotting import (
    plot_main_2d, plot_heat_fluxes, plot_ra_vs_x,
    plot_alpha_comparison, plot_nu_comparison,
)
from calculation_log import render_point_detail
from formatting import _fc, _fe


# ─────────────── Боковая панель ───────────────

def sidebar_inputs() -> dict:
    """Все входные параметры — в боковой панели."""
    params = {}

    st.sidebar.header('Параметры нагревателя')

    params['I'] = st.sidebar.slider(
        'Сила тока I, А',
        min_value=int(RANGES_HEATER['I'][0]),
        max_value=int(RANGES_HEATER['I'][1]),
        value=int(DEFAULTS_HEATER['I']),
        step=10,
    )

    params['R20'] = st.sidebar.number_input(
        'R₂₀, Ом/м (при 20 °C)',
        value=DEFAULTS_HEATER['R20'],
        min_value=RANGES_HEATER['R20'][0],
        max_value=RANGES_HEATER['R20'][1],
        format='%.4e',
        step=1e-4,
    )

    params['alpha_R'] = st.sidebar.number_input(
        'α_R (ТКС), К⁻¹',
        value=DEFAULTS_HEATER['alpha_R'],
        min_value=RANGES_HEATER['alpha_R'][0],
        max_value=RANGES_HEATER['alpha_R'][1],
        format='%.4e',
        step=1e-4,
    )

    params['b_mm'] = st.sidebar.slider(
        'Ширина пластины b, мм',
        min_value=int(RANGES_HEATER['b_mm'][0]),
        max_value=int(RANGES_HEATER['b_mm'][1]),
        value=int(DEFAULTS_HEATER['b_mm']),
        step=5,
    )

    params['L_mm'] = st.sidebar.slider(
        'Высота пластины L, мм',
        min_value=int(RANGES_HEATER['L_mm'][0]),
        max_value=int(RANGES_HEATER['L_mm'][1]),
        value=int(DEFAULTS_HEATER['L_mm']),
        step=10,
    )

    st.sidebar.header('Среда и излучение')

    params['t_fluid_C'] = st.sidebar.slider(
        'Температура среды t_ж, °C',
        min_value=int(RANGES_ENV['t_fluid_C'][0]),
        max_value=int(RANGES_ENV['t_fluid_C'][1]),
        value=int(DEFAULTS_ENV['t_fluid_C']),
    )

    params['g'] = st.sidebar.number_input(
        'g, м/с²',
        value=DEFAULTS_ENV['g'],
        format='%.5f',
    )

    params['C_pr'] = st.sidebar.slider(
        'C_пр, Вт/(м²·К⁴)',
        min_value=float(RANGES_ENV['C_pr'][0]),
        max_value=float(RANGES_ENV['C_pr'][1]),
        value=float(DEFAULTS_ENV['C_pr']),
        step=0.1,
    )

    params['P_Pa'] = st.sidebar.slider(
        'Давление P, Па',
        min_value=int(RANGES_ENV['P_Pa'][0]),
        max_value=int(RANGES_ENV['P_Pa'][1]),
        value=int(DEFAULTS_ENV['P_Pa']),
        step=100,
    )

    st.sidebar.header('Параметры расчёта')

    params['x_min_mm'] = st.sidebar.number_input(
        'x_min, мм',
        value=DEFAULTS_CALC['x_min_mm'],
        min_value=1.0,
        max_value=100.0,
        step=1.0,
    )

    params['N'] = st.sidebar.slider(
        'Число точек N',
        min_value=10,
        max_value=500,
        value=DEFAULTS_CALC['N'],
        step=10,
    )

    st.sidebar.header('Корреляция для Nu')
    corr_label = st.sidebar.radio(
        'Выберите корреляцию',
        (
            'По методичке (кусочная: лам/турб + ε_t)',
            'Черчилль–Чу (полная, весь диапазон Ra)',
        ),
        index=0,
        help='Кусочная: Nu=0,60·(GrPr)^0,25 (лам.) и Nu=0,15·(GrPr)^1/3 (турб.) с поправкой ε_t.\n\n'
             'Черчилль–Чу: Nu=[0,825+0,387·(Ra·ψ)^1/6]² — плавная формула для всех режимов.',
    )
    params['correlation'] = 'churchill_chu' if 'Черчилль' in corr_label else 'piecewise'

    return params


# ─────────────── Расчёт ───────────────

def run_calculation(params: dict) -> CalculationResult:
    """Запустить расчёт с прогресс-баром."""
    progress = st.progress(0, text='Расчёт...')

    result = solve_plate(
        I=float(params['I']),
        R20=params['R20'],
        alpha_R=params['alpha_R'],
        b_mm=float(params['b_mm']),
        L_mm=float(params['L_mm']),
        t_fluid_C=float(params['t_fluid_C']),
        g=params['g'],
        C_pr=params['C_pr'],
        P_Pa=float(params['P_Pa']),
        x_min_mm=params['x_min_mm'],
        N=params['N'],
        correlation=params.get('correlation', 'piecewise'),
        progress_callback=lambda frac: progress.progress(frac, text=f'Расчёт... {frac*100:.0f}%'),
    )

    progress.empty()
    return result


# ─────────────── Сводная таблица ───────────────

def build_summary_df(result: CalculationResult) -> pd.DataFrame:
    """Сводная таблица всех точек."""
    rows = []
    for pt in result.points:
        if not pt.converged:
            continue
        rows.append({
            'x, мм': _fc(pt.x_m * 1000, 1),
            'x, м': _fc(pt.x_m, 4),
            't_c, °C': _fc(pt.t_c, 2),
            't_опр, °C': _fc(pt.t_film, 2),
            'ν, м²/с': _fe(pt.air.nu, 4),
            'λ, Вт/(м·К)': _fe(pt.air.lam, 4),
            'Pr': _fc(pt.air.Pr, 4),
            'β, К⁻¹': _fe(pt.beta, 4),
            'Gr': _fe(pt.Gr, 3),
            'Ra': _fe(pt.Ra, 3),
            'Nu': _fc(pt.Nu, 2),
            'Nu (альт.)': _fc(pt.Nu_alt, 2),
            'Режим': {'lam': 'лам.', 'turb': 'турб.', 'full': 'Ч-Ч'}.get(pt.regime, '?'),
            'α, Вт/(м²·К)': _fc(pt.alpha, 2),
            'q_конв, Вт/м²': _fc(pt.q_conv, 1),
            'q_рад, Вт/м²': _fc(pt.q_rad, 1),
            'q_эл, Вт/м²': _fc(pt.q_el, 1),
            'Невязка, Вт/м²': _fe(pt.residual, 2),
        })
    return pd.DataFrame(rows)


# ─────────────── Главное приложение ───────────────

def main():
    st.set_page_config(
        page_title='Тепловой баланс пластины (стенд Керимова)',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.title('Расчёт теплового баланса вертикальной обогреваемой пластины')
    st.caption('Стенд Керимова, НИУ «МЭИ». Свободная конвекция воздуха (ламинарный/турбулентный режим).')

    params = sidebar_inputs()

    # Кнопка расчёта
    if st.sidebar.button('Рассчитать', type='primary', use_container_width=True):
        result = run_calculation(params)
        st.session_state['result'] = result

    result: CalculationResult = st.session_state.get('result')

    if result is None:
        st.info('Задайте параметры в боковой панели и нажмите «Рассчитать».')
        return

    # Статистика режимов
    n_fail = sum(1 for p in result.points if not p.converged)
    n_ok = sum(1 for p in result.points if p.converged)
    if result.correlation == 'piecewise':
        n_lam = sum(1 for p in result.points if p.converged and p.regime == 'lam')
        n_turb = sum(1 for p in result.points if p.converged and p.regime == 'turb')
        st.info(f'Корреляция: **методичка**. Режимы: ламинарный — {n_lam}, турбулентный — {n_turb} (из {n_ok} точек)')
    else:
        st.info(f'Корреляция: **Черчилль–Чу** (полная формула, {n_ok} точек)')
    if n_fail > 0:
        st.error(f'{n_fail} из {result.N} точек: солвер не сошёлся.')

    # --- Вкладки ---
    tab1, tab2, tab3 = st.tabs([
        'Графики', 'Ход расчёта', 'Сводная таблица',
    ])

    # ─── Вкладка 1: Графики ───
    with tab1:
        st.subheader('Температура и коэфф. теплоотдачи')
        st.plotly_chart(plot_main_2d(result), use_container_width=True)

        st.subheader('Число Рэлея Ra(x)')
        st.plotly_chart(plot_ra_vs_x(result), use_container_width=True)

        st.subheader('Компоненты теплового потока')
        st.plotly_chart(plot_heat_fluxes(result), use_container_width=True)

        st.subheader('Сравнение корреляций: методичка vs Черчилль–Чу')
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_alpha_comparison(result), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_nu_comparison(result), use_container_width=True)

    # ─── Вкладка 2: Ход расчёта ───
    with tab2:
        converged_pts = [i for i, p in enumerate(result.points) if p.converged]
        if not converged_pts:
            st.error('Нет сошедшихся точек для отображения.')
        else:
            idx = st.slider(
                'Выберите точку x_i',
                min_value=0,
                max_value=len(converged_pts) - 1,
                value=0,
                format='%d',
            )
            actual_idx = converged_pts[idx]
            pt = result.points[actual_idx]
            st.markdown(f'**Точка {idx + 1} из {len(converged_pts)}:** '
                        f'x = {_fc(pt.x_m * 1000, 1)} мм')
            render_point_detail(result, actual_idx)

    # ─── Вкладка 3: Сводная таблица ───
    with tab3:
        df = build_summary_df(result)
        st.dataframe(df, use_container_width=True, height=600)

        csv = df.to_csv(index=False, sep=';')
        st.download_button(
            'Скачать CSV',
            data=csv,
            file_name='heat_balance.csv',
            mime='text/csv',
        )


if __name__ == '__main__':
    main()
