"""
Streamlit-приложение: Расчёт теплового баланса вертикальной нагреваемой пластины
(стенд Керимова, НИУ "МЭИ").
"""

import numpy as np
import streamlit as st
import pandas as pd

from config import (
    DEFAULTS_HEATER, RANGES_HEATER,
    DEFAULTS_ENV, RANGES_ENV,
    DEFAULTS_CALC,
    PARAMETRIC_OPTIONS, DEFAULTS_PARAMETRIC,
    GR_PR_CRIT_1, GR_PR_CRIT_2,
)
from solver import solve_plate, CalculationResult
from plotting import (
    plot_t_vs_x, plot_ra_vs_x,
    plot_main_2d, plot_alpha_comparison, plot_nu_comparison,
    plot_heat_fluxes, plot_criteria, plot_air_props,
    plot_3d_surface,
)
from calculation_log import render_point_detail
from comparison_page import render_comparison_tab
from verification_page import render_verification_tab
from turbulence_page import render_turbulence_tab
from verification_data import compare_coolprop_vs_gsssd, get_gsssd_table
from formatting import _fc, _fe


# ── Корреляции: 8 методик ──────────────────────────────────────────────────

CORR_OPTIONS = {
    'kerimov': 'Керимов (Nu_ж, eps_t)',
    'kuznetov': 'Кузнецов (Nu, Phi(Pr), q=const)',
    'churchill_chu': 'Черчилль-Чу (средний Nu)',
    'leontiev': 'Леонтьев (Брдлик / Эккерт-Дж.)',
    'churchill_ozoe': 'Churchill & Ozoe (1973)',
    'vliet': 'Vliet (1969/1975)',
    'fujii': 'Fujii & Fujii (1976)',
    'isachenko': 'Исаченко и др. (1981)',
}

CORR_HINTS = {
    'kerimov': 'Свойства при t_ж, поправка eps_t. Лам + перех + турб.',
    'kuznetov': 'Свойства при t_плён, Phi(Pr), q_c=const. Лам + турб.',
    'churchill_chu': 'Обобщённая, средний Nu, весь диапазон Ra.',
    'leontiev': 'UHF: Брдлик (лам.) + Эккерт-Дж. (турб.), t_плён.',
    'churchill_ozoe': 'UHF: только ламинарный (Ra < 10^9), t_плён.',
    'vliet': 'UHF: воздух, безытерац. через Ra*. Лам + турб.',
    'fujii': 'UHF: только ламинарный, Pr-обобщённая, t_плён.',
    'isachenko': 'UHF: свойства при t_ж, eps_t. Лам + перех + турб.',
}

DEFAULT_CRITS = {
    'kerimov': (1e9, 6e10),
    'kuznetov': (1e9, 1e12),
    'leontiev': (2e7, 2e7),
    'vliet': (3.4e9, 3.4e9),
    'isachenko': (1e9, 6e10),
}

REGIME_LABELS = {'lam': 'лам.', 'trans': 'перех.', 'turb': 'турб.', 'full': 'Ч-Ч'}


# ── Вспомогательные функции ────────────────────────────────────────────────


def build_summary_df(result: CalculationResult) -> pd.DataFrame:
    """Сводная таблица всех точек."""
    rows = []
    for pt in result.points:
        if not pt.converged:
            continue
        rows.append({
            'x, мм': _fc(pt.x_m * 1000, 1),
            't_c, °C': _fc(pt.t_c, 2),
            't_опр, °C': _fc(pt.t_film, 2),
            'nu, м2/с': _fe(pt.air.nu),
            'lambda, Вт/(м*К)': _fe(pt.air.lam),
            'Pr': _fc(pt.air.Pr, 4),
            'beta, 1/К': _fe(pt.beta),
            'Gr': _fe(pt.Gr, 3),
            'Ra': _fe(pt.Ra, 3),
            'Nu': _fc(pt.Nu, 2),
            'Nu(alt)': _fc(pt.Nu_alt, 2),
            'Режим': REGIME_LABELS.get(pt.regime, '?'),
            'alpha, Вт/(м2*К)': _fc(pt.alpha, 2),
            'q_конв': _fc(pt.q_conv, 1),
            'q_рад': _fc(pt.q_rad, 1),
            'q_эл': _fc(pt.q_el, 1),
            'Невязка': _fe(pt.residual, 2),
        })
    return pd.DataFrame(rows)


def run_calculation(
    I, R20, alpha_R, b_mm, L_mm,
    t_fluid_C, g, C_pr, P_Pa,
    x_min_mm, N, corr_key, crit_1, crit_2,
) -> CalculationResult:
    """Запуск расчёта с прогресс-баром Streamlit."""
    bar = st.progress(0, text='Расчёт...')

    def progress_cb(frac):
        bar.progress(frac, text=f'Расчёт... {frac * 100:.0f}%')

    result = solve_plate(
        I=float(I), R20=R20, alpha_R=alpha_R,
        b_mm=float(b_mm), L_mm=float(L_mm),
        t_fluid_C=float(t_fluid_C), g=g, C_pr=C_pr,
        P_Pa=float(P_Pa), x_min_mm=x_min_mm, N=N,
        correlation=corr_key,
        t_ref_mode='auto',
        crit_1=crit_1,
        crit_2=crit_2,
        progress_callback=progress_cb,
    )
    bar.empty()
    return result


def run_parametric(
    base_params: dict,
    param_key: str,
    param_values,
    corr_key: str,
    crit_1: float,
    crit_2: float,
):
    """Параметрическое исследование: варьируем один параметр."""
    results = []
    bar = st.progress(0, text='Параметрическое исследование...')

    for i, val in enumerate(param_values):
        p = dict(base_params)
        p[param_key] = val
        res = solve_plate(
            I=float(p['I']), R20=float(p['R20']),
            alpha_R=float(p['alpha_R']),
            b_mm=float(p['b_mm']), L_mm=float(p['L_mm']),
            t_fluid_C=float(p['t_fluid_C']), g=float(p['g']),
            C_pr=float(p['C_pr']), P_Pa=float(p['P_Pa']),
            x_min_mm=float(p['x_min_mm']), N=int(p['N']),
            correlation=corr_key,
            t_ref_mode='auto',
            crit_1=crit_1,
            crit_2=crit_2,
        )
        results.append(res)
        bar.progress((i + 1) / len(param_values),
                     text=f'Параметрическое исследование... {i + 1}/{len(param_values)}')

    bar.empty()
    return results


# ── Боковая панель ─────────────────────────────────────────────────────────


def sidebar_inputs():
    """Все входные параметры в sidebar (слайдеры + number_input)."""
    st.sidebar.header('Параметры расчёта')

    # --- Нагреватель ---
    st.sidebar.subheader('Нагреватель')
    I = st.sidebar.slider(
        'I, А', min_value=int(RANGES_HEATER['I'][0]),
        max_value=int(RANGES_HEATER['I'][1]),
        value=int(DEFAULTS_HEATER['I']), step=10,
    )
    R20 = st.sidebar.number_input(
        'R20, Ом/м', value=DEFAULTS_HEATER['R20'],
        min_value=RANGES_HEATER['R20'][0],
        max_value=RANGES_HEATER['R20'][1],
        format='%.4e', step=1e-4,
    )
    alpha_R = st.sidebar.number_input(
        'alpha_R (ТКС), 1/К', value=DEFAULTS_HEATER['alpha_R'],
        min_value=RANGES_HEATER['alpha_R'][0],
        max_value=RANGES_HEATER['alpha_R'][1],
        format='%.4e', step=1e-4,
    )

    # --- Пластина ---
    st.sidebar.subheader('Пластина')
    b_mm = st.sidebar.slider(
        'b, мм', min_value=int(RANGES_HEATER['b_mm'][0]),
        max_value=int(RANGES_HEATER['b_mm'][1]),
        value=int(DEFAULTS_HEATER['b_mm']), step=5,
    )
    L_mm = st.sidebar.slider(
        'L, мм', min_value=int(RANGES_HEATER['L_mm'][0]),
        max_value=int(RANGES_HEATER['L_mm'][1]),
        value=int(DEFAULTS_HEATER['L_mm']), step=10,
    )

    # --- Среда ---
    st.sidebar.subheader('Среда')
    t_fluid_C = st.sidebar.slider(
        't_ж, С', min_value=int(RANGES_ENV['t_fluid_C'][0]),
        max_value=int(RANGES_ENV['t_fluid_C'][1]),
        value=int(DEFAULTS_ENV['t_fluid_C']),
    )
    C_pr = st.sidebar.number_input(
        'C_пр, Вт/(м2*К4)', value=DEFAULTS_ENV['C_pr'],
        min_value=RANGES_ENV['C_pr'][0],
        max_value=RANGES_ENV['C_pr'][1],
        step=0.1, format='%.1f',
    )
    P_Pa = st.sidebar.number_input(
        'P, Па', value=int(DEFAULTS_ENV['P_Pa']),
        min_value=int(RANGES_ENV['P_Pa'][0]),
        max_value=int(RANGES_ENV['P_Pa'][1]), step=100,
    )
    g = st.sidebar.number_input(
        'g, м/с2', value=DEFAULTS_ENV['g'], format='%.5f',
    )

    # --- Расчёт ---
    st.sidebar.subheader('Расчёт')
    N = st.sidebar.number_input(
        'Точек N', value=DEFAULTS_CALC['N'],
        min_value=10, max_value=500, step=10,
    )
    x_min_mm = st.sidebar.number_input(
        'x_min, мм', value=DEFAULTS_CALC['x_min_mm'],
        min_value=1.0, max_value=100.0, step=1.0,
    )

    # --- Корреляция (selectbox с 8 вариантами) ---
    st.sidebar.subheader('Корреляция')
    corr_key = st.sidebar.selectbox(
        'Методика',
        options=list(CORR_OPTIONS.keys()),
        format_func=lambda k: CORR_OPTIONS[k],
        index=0,
    )
    st.sidebar.caption(CORR_HINTS.get(corr_key, ''))

    # Границы режимов (по умолчанию зависят от выбранной корреляции)
    default_c1, default_c2 = DEFAULT_CRITS.get(corr_key, (GR_PR_CRIT_1, GR_PR_CRIT_2))

    with st.sidebar.expander('Границы режимов'):
        crit_1 = st.number_input(
            'Ra1 (лам -> перех)',
            value=default_c1,
            min_value=1e4, max_value=1e15,
            format='%.2e', step=1e9,
        )
        crit_2 = st.number_input(
            'Ra2 (перех -> турб)',
            value=default_c2,
            min_value=1e4, max_value=1e15,
            format='%.2e', step=1e10,
        )

    # --- Кнопка расчёта ---
    do_calc = st.sidebar.button('Рассчитать', type='primary', use_container_width=True)

    # --- Параметрическое исследование ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Параметрическое исследование')
    param_key = st.sidebar.selectbox(
        'Варьируемый параметр',
        options=list(PARAMETRIC_OPTIONS.keys()),
        format_func=lambda k: PARAMETRIC_OPTIONS[k],
        index=0,
    )
    param_min = st.sidebar.number_input(
        f'{PARAMETRIC_OPTIONS[param_key]}: мин',
        value=DEFAULTS_PARAMETRIC['min'],
    )
    param_max = st.sidebar.number_input(
        f'{PARAMETRIC_OPTIONS[param_key]}: макс',
        value=DEFAULTS_PARAMETRIC['max'],
    )
    n_param = st.sidebar.number_input(
        'Число значений', value=DEFAULTS_PARAMETRIC['n_values'],
        min_value=3, max_value=30, step=1,
    )
    do_parametric = st.sidebar.button('Параметрический расчёт', use_container_width=True)

    return {
        'I': I, 'R20': R20, 'alpha_R': alpha_R,
        'b_mm': b_mm, 'L_mm': L_mm,
        't_fluid_C': t_fluid_C, 'g': g, 'C_pr': C_pr,
        'P_Pa': P_Pa, 'x_min_mm': x_min_mm, 'N': N,
        'corr_key': corr_key,
        'crit_1': crit_1, 'crit_2': crit_2,
        'do_calc': do_calc,
        'do_parametric': do_parametric,
        'param_key': param_key,
        'param_min': param_min,
        'param_max': param_max,
        'n_param': n_param,
    }


# ── Главная функция ───────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title='Тепловой баланс пластины',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.title('Тепловой баланс вертикальной пластины')
    st.caption('Стенд Керимова, НИУ "МЭИ" -- свободная конвекция воздуха')

    # ── Боковая панель ──
    ui = sidebar_inputs()

    corr_key = ui['corr_key']
    crit_1 = ui['crit_1']
    crit_2 = ui['crit_2']

    # Параметры для передачи во вкладки (без управления)
    ui_params = {
        'I': ui['I'], 'R20': ui['R20'], 'alpha_R': ui['alpha_R'],
        'b_mm': ui['b_mm'], 'L_mm': ui['L_mm'],
        't_fluid_C': ui['t_fluid_C'], 'g': ui['g'], 'C_pr': ui['C_pr'],
        'P_Pa': ui['P_Pa'], 'x_min_mm': ui['x_min_mm'], 'N': ui['N'],
    }

    # ── Расчёт ──
    if ui['do_calc']:
        result = run_calculation(
            I=ui['I'], R20=ui['R20'], alpha_R=ui['alpha_R'],
            b_mm=ui['b_mm'], L_mm=ui['L_mm'],
            t_fluid_C=ui['t_fluid_C'], g=ui['g'], C_pr=ui['C_pr'],
            P_Pa=ui['P_Pa'], x_min_mm=ui['x_min_mm'], N=ui['N'],
            corr_key=corr_key, crit_1=crit_1, crit_2=crit_2,
        )
        st.session_state['result'] = result

    # ── Параметрическое исследование ──
    if ui['do_parametric']:
        base = dict(ui_params)
        param_values = np.linspace(ui['param_min'], ui['param_max'], int(ui['n_param']))
        results_param = run_parametric(
            base, ui['param_key'], param_values,
            corr_key, crit_1, crit_2,
        )
        st.session_state['parametric_results'] = results_param
        st.session_state['parametric_param_name'] = PARAMETRIC_OPTIONS[ui['param_key']]
        st.session_state['parametric_param_values'] = param_values.tolist()

    result: CalculationResult = st.session_state.get('result')

    # ── Вкладки ──
    tab_names = [
        'Графики',
        'Ход расчёта',
        'Сводная таблица',
        'Свойства воздуха',
        'Турбулентный режим',
        'Сравнение методик',
        'Верификация',
    ]
    tabs = st.tabs(tab_names)

    # ── Вкладка 1: Графики ──
    with tabs[0]:
        if result is not None:
            n_ok = sum(1 for p in result.points if p.converged)
            n_fail = sum(1 for p in result.points if not p.converged)

            corr_names = {
                'kerimov': 'Керимов', 'kuznetov': 'Кузнецов',
                'churchill_chu': 'Ч-Ч', 'leontiev': 'Леонтьев',
                'churchill_ozoe': 'Ch-Ozoe', 'vliet': 'Vliet',
                'fujii': 'Fujii', 'isachenko': 'Исаченко',
            }
            t_ref_str = 't_ж' if result.t_ref_mode == 'fluid' else 't_пл'
            corr_label = corr_names.get(result.correlation, result.correlation)

            if result.correlation == 'churchill_chu':
                st.caption(f'{corr_label} ({t_ref_str}) | {n_ok} точек')
            else:
                n_lam = sum(1 for p in result.points if p.converged and p.regime == 'lam')
                n_trans = sum(1 for p in result.points if p.converged and p.regime == 'trans')
                n_turb = sum(1 for p in result.points if p.converged and p.regime == 'turb')
                st.caption(
                    f'{corr_label} ({t_ref_str}) | '
                    f'лам: {n_lam}, перех: {n_trans}, турб: {n_turb} (из {n_ok})'
                )
            if n_fail > 0:
                st.error(f'{n_fail} точек не сошлись.')

            # Основные графики
            st.plotly_chart(plot_main_2d(result), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_t_vs_x(result), use_container_width=True)
            with col2:
                st.plotly_chart(plot_ra_vs_x(result), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_alpha_comparison(result), use_container_width=True)
            with col2:
                st.plotly_chart(plot_nu_comparison(result), use_container_width=True)

            st.plotly_chart(plot_heat_fluxes(result), use_container_width=True)
            st.plotly_chart(plot_criteria(result), use_container_width=True)
            st.plotly_chart(plot_air_props(result), use_container_width=True)

            # Параметрический 3D-график
            if 'parametric_results' in st.session_state:
                st.markdown('---')
                st.subheader('Параметрическое исследование (3D)')
                fig_3d = plot_3d_surface(
                    st.session_state['parametric_results'],
                    st.session_state['parametric_param_name'],
                    st.session_state['parametric_param_values'],
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info('Задайте параметры в боковой панели и нажмите "Рассчитать".')

    # ── Вкладка 2: Ход расчёта ──
    with tabs[1]:
        if result is not None:
            converged_pts = [i for i, p in enumerate(result.points) if p.converged]
            if not converged_pts:
                st.error('Нет сошедшихся точек.')
            else:
                idx = st.slider(
                    'Точка x_i', min_value=0,
                    max_value=len(converged_pts) - 1, value=0, format='%d',
                )
                actual_idx = converged_pts[idx]
                pt = result.points[actual_idx]
                st.markdown(
                    f'**Точка {idx + 1}/{len(converged_pts)}:** '
                    f'x = {_fc(pt.x_m * 1000, 1)} мм'
                )
                render_point_detail(result, actual_idx)
        else:
            st.info('Нажмите "Рассчитать" для получения результатов.')

    # ── Вкладка 3: Сводная таблица ──
    with tabs[2]:
        if result is not None:
            df = build_summary_df(result)
            st.dataframe(df, use_container_width=True, height=400)
            csv = df.to_csv(index=False, sep=';')
            st.download_button(
                'Скачать CSV', data=csv,
                file_name='heat_balance.csv', mime='text/csv',
            )
        else:
            st.info('Нажмите "Рассчитать" для получения результатов.')

    # ── Вкладка 4: Свойства воздуха ──
    with tabs[3]:
        st.subheader('Проверка свойств воздуха: CoolProp vs ГСССД 8-79')

        try:
            import CoolProp  # noqa: F401
            st.success('CoolProp установлен и доступен.')
        except ImportError:
            st.error('CoolProp не установлен. Установите: pip install CoolProp')

        st.markdown('**Справочные данные ГСССД 8-79:**')
        df_gsssd_ref = get_gsssd_table()
        st.dataframe(df_gsssd_ref, use_container_width=True, hide_index=True)

        st.markdown('---')
        st.markdown('**Сравнение CoolProp с ГСССД:**')
        df_cmp = compare_coolprop_vs_gsssd(float(ui['P_Pa']))
        if df_cmp is not None:
            delta_col = df_cmp['δ, %'].str.replace(',', '.').astype(float)
            max_delta = delta_col.max()
            if max_delta < 3.0:
                st.success(f'Макс. расхождение: {max_delta:.2f}% (норма < 3%)')
            else:
                st.warning(f'Макс. расхождение: {max_delta:.2f}%')
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # ── Вкладка 5: Турбулентный режим ──
    with tabs[4]:
        render_turbulence_tab(ui_params)

    # ── Вкладка 6: Сравнение методик ──
    with tabs[5]:
        render_comparison_tab(ui_params)

    # ── Вкладка 7: Верификация ──
    with tabs[6]:
        render_verification_tab(ui_params)


if __name__ == '__main__':
    main()
