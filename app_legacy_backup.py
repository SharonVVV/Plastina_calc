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
    GR_PR_CRIT_1, GR_PR_CRIT_2,
)
from solver import solve_plate, CalculationResult
from plotting import plot_plate_composition
from calculation_log import render_point_detail
from comparison_page import render_comparison_tab
from verification_page import render_verification_tab
from turbulence_page import render_turbulence_tab
from qconst_page import render_qconst_tab
from back_losses_page import render_back_losses_tab
from verification_data import compare_coolprop_vs_gsssd, get_gsssd_table
from formatting import _fc, _fe


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
            'Pr': _fc(pt.air.Pr, 4),
            'Gr': _fe(pt.Gr, 3),
            'Ra': _fe(pt.Ra, 3),
            'Nu': _fc(pt.Nu, 2),
            'Режим': {'lam': 'лам.', 'trans': 'перех.', 'turb': 'турб.', 'full': 'Ч-Ч'}.get(pt.regime, '?'),
            'α, Вт/(м²·К)': _fc(pt.alpha, 2),
            'q_конв': _fc(pt.q_conv, 1),
            'q_рад': _fc(pt.q_rad, 1),
            'q_эл': _fc(pt.q_el, 1),
            'Невязка': _fe(pt.residual, 2),
        })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(
        page_title='Тепловой баланс пластины',
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    st.markdown(
        '<h2 style="margin-top:-1rem">Тепловой баланс вертикальной пластины</h2>'
        '<p style="color:gray;margin-top:-0.8rem">Стенд Керимова, НИУ «МЭИ» — свободная конвекция воздуха</p>',
        unsafe_allow_html=True,
    )

    # ─── Главный экран: ввод | визуализация ───────────────────────────────
    col_in, col_viz = st.columns([1, 3], gap='large')

    with col_in:
        # --- Нагреватель ---
        st.markdown('##### Нагреватель')
        c1, c2 = st.columns(2)
        with c1:
            I = st.number_input('I, А', value=int(DEFAULTS_HEATER['I']),
                                min_value=int(RANGES_HEATER['I'][0]),
                                max_value=int(RANGES_HEATER['I'][1]), step=10)
        with c2:
            R20 = st.number_input('R₂₀, Ом/м', value=DEFAULTS_HEATER['R20'],
                                  min_value=RANGES_HEATER['R20'][0],
                                  max_value=RANGES_HEATER['R20'][1],
                                  format='%.4e', step=1e-4)
        alpha_R = st.number_input('α_R (ТКС), К⁻¹', value=DEFAULTS_HEATER['alpha_R'],
                                  min_value=RANGES_HEATER['alpha_R'][0],
                                  max_value=RANGES_HEATER['alpha_R'][1],
                                  format='%.4e', step=1e-4)

        # --- Пластина ---
        st.markdown('##### Пластина')
        c1, c2 = st.columns(2)
        with c1:
            b_mm = st.number_input('b, мм', value=int(DEFAULTS_HEATER['b_mm']),
                                   min_value=int(RANGES_HEATER['b_mm'][0]),
                                   max_value=int(RANGES_HEATER['b_mm'][1]), step=5)
        with c2:
            L_mm = st.number_input('L, мм', value=int(DEFAULTS_HEATER['L_mm']),
                                   min_value=int(RANGES_HEATER['L_mm'][0]),
                                   max_value=int(RANGES_HEATER['L_mm'][1]), step=10)

        # --- Среда ---
        st.markdown('##### Среда')
        c1, c2 = st.columns(2)
        with c1:
            t_fluid_C = st.number_input('t_ж, °C', value=int(DEFAULTS_ENV['t_fluid_C']),
                                        min_value=int(RANGES_ENV['t_fluid_C'][0]),
                                        max_value=int(RANGES_ENV['t_fluid_C'][1]))
        with c2:
            C_pr = st.number_input('C_пр, Вт/(м²·К⁴)', value=DEFAULTS_ENV['C_pr'],
                                   min_value=RANGES_ENV['C_pr'][0],
                                   max_value=RANGES_ENV['C_pr'][1],
                                   step=0.1, format='%.1f')
        c1, c2 = st.columns(2)
        with c1:
            P_Pa = st.number_input('P, Па', value=int(DEFAULTS_ENV['P_Pa']),
                                   min_value=int(RANGES_ENV['P_Pa'][0]),
                                   max_value=int(RANGES_ENV['P_Pa'][1]), step=100)
        with c2:
            g = st.number_input('g, м/с²', value=DEFAULTS_ENV['g'], format='%.5f')

        # --- Расчёт ---
        st.markdown('##### Расчёт')
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input('Точек N', value=DEFAULTS_CALC['N'],
                                min_value=10, max_value=500, step=10)
        with c2:
            x_min_mm = st.number_input('x_min, мм', value=DEFAULTS_CALC['x_min_mm'],
                                       min_value=1.0, max_value=100.0, step=1.0)

        # --- Корреляция (8 методик) ---
        corr_options = {
            'kerimov': 'Керимов (Nu_ж, ε_t)',
            'kuznetov': 'Кузнецов (Nu, Φ(Pr), q=const)',
            'churchill_chu': 'Черчилль–Чу (средний Nu)',
            'leontiev': 'Леонтьев (Брдлик / Эккерт–Дж.)',
            'churchill_ozoe': 'Churchill & Ozoe (1973)',
            'vliet': 'Vliet (1969/1975)',
            'fujii': 'Fujii & Fujii (1976)',
            'isachenko': 'Исаченко и др. (1981)',
        }
        corr_key = st.selectbox(
            'Корреляция',
            options=list(corr_options.keys()),
            format_func=lambda k: corr_options[k],
            index=0,
        )

        _default_crits = {
            'kerimov': (1e9, 6e10),
            'kuznetov': (1e9, 1e12),
            'leontiev': (2e7, 2e7),
            'vliet': (3.4e9, 3.4e9),
            'isachenko': (1e9, 6e10),
        }
        default_c1, default_c2 = _default_crits.get(corr_key, (GR_PR_CRIT_1, GR_PR_CRIT_2))

        with st.expander('Границы режимов'):
            c1, c2 = st.columns(2)
            with c1:
                crit_1 = st.number_input('Ra₁ (лам→перех)',
                                         value=default_c1,
                                         min_value=1e4, max_value=1e15,
                                         format='%.2e', step=1e9,
                                         key=f'crit1_{corr_key}')
            with c2:
                crit_2 = st.number_input('Ra₂ (перех→турб)',
                                         value=default_c2,
                                         min_value=1e4, max_value=1e15,
                                         format='%.2e', step=1e10,
                                         key=f'crit2_{corr_key}')

        do_calc = st.button('Рассчитать', type='primary', use_container_width=True)

    # ─── Расчёт ──────────────────────────────────────────────────────────
    # Автопересчёт при смене любого параметра, влияющего на результат
    _current_params = (corr_key, float(I), R20, alpha_R, float(b_mm),
                       float(L_mm), float(t_fluid_C), g, C_pr, float(P_Pa),
                       x_min_mm, N, crit_1, crit_2)
    if st.session_state.get('_last_params') != _current_params and 'result' in st.session_state:
        do_calc = True
    st.session_state['_last_params'] = _current_params

    if do_calc:
        result = solve_plate(
            I=float(I), R20=R20, alpha_R=alpha_R,
            b_mm=float(b_mm), L_mm=float(L_mm),
            t_fluid_C=float(t_fluid_C), g=g, C_pr=C_pr,
            P_Pa=float(P_Pa), x_min_mm=x_min_mm, N=N,
            correlation=corr_key,
            t_ref_mode='auto',
            crit_1=crit_1,
            crit_2=crit_2,
        )
        st.session_state['result'] = result

    result: CalculationResult = st.session_state.get('result')

    # ─── Визуализация (главный график) ───────────────────────────────────
    with col_viz:
        if result is not None:
            n_ok = sum(1 for p in result.points if p.converged)
            n_fail = sum(1 for p in result.points if not p.converged)
            t_ref_str = 't_ж' if result.t_ref_mode == 'fluid' else 't_пл'
            corr_names = {
                'kerimov': 'Керимов', 'kuznetov': 'Кузнецов', 'churchill_chu': 'Ч-Ч',
                'leontiev': 'Леонтьев', 'churchill_ozoe': 'Ch-Ozoe',
                'vliet': 'Vliet', 'fujii': 'Fujii', 'isachenko': 'Исаченко',
            }
            corr_label = corr_names.get(result.correlation, result.correlation)
            if result.correlation == 'churchill_chu':
                st.caption(f'{corr_label} ({t_ref_str}) | {n_ok} точек')
            else:
                n_lam = sum(1 for p in result.points if p.converged and p.regime == 'lam')
                n_trans = sum(1 for p in result.points if p.converged and p.regime == 'trans')
                n_turb = sum(1 for p in result.points if p.converged and p.regime == 'turb')
                st.caption(f'{corr_label} ({t_ref_str}) | лам: {n_lam}, перех: {n_trans}, турб: {n_turb} (из {n_ok})')
            if n_fail > 0:
                st.error(f'{n_fail} точек не сошлись.')

            st.plotly_chart(plot_plate_composition(result), use_container_width=True)
        else:
            st.info('Задайте параметры слева и нажмите «Рассчитать».')

    # ─── Нижняя часть: вкладки ───────────────────────────────────────────
    ui_params = {
        'I': I, 'R20': R20, 'alpha_R': alpha_R,
        'b_mm': b_mm, 'L_mm': L_mm,
        't_fluid_C': t_fluid_C, 'g': g, 'C_pr': C_pr,
        'P_Pa': P_Pa, 'x_min_mm': x_min_mm,
    }

    st.markdown('---')
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        'Сводная таблица', 'Ход расчёта',
        'Свойства воздуха', 'Турбулентный режим',
        'Сравнение методик', 'Верификация',
        'Стенд q_w=const', 'Потери (тыл+торцы)',
    ])

    with tab1:
        if result is not None:
            df = build_summary_df(result)
            st.dataframe(df, use_container_width=True, height=400)
            csv = df.to_csv(index=False, sep=';')
            st.download_button('Скачать CSV', data=csv,
                               file_name='heat_balance.csv', mime='text/csv')
        else:
            st.info('Нажмите «Рассчитать» для получения результатов.')

    with tab2:
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
                st.markdown(f'**Точка {idx + 1}/{len(converged_pts)}:** '
                            f'x = {_fc(pt.x_m * 1000, 1)} мм')
                render_point_detail(result, actual_idx)
        else:
            st.info('Нажмите «Рассчитать» для получения результатов.')

    with tab3:
        st.subheader('Проверка свойств воздуха: CoolProp vs ГСССД 8-79')
        st.markdown('**Справочные данные ГСССД 8-79:**')
        st.dataframe(get_gsssd_table(), use_container_width=True, hide_index=True)
        st.markdown('---')
        st.markdown('**Сравнение CoolProp с ГСССД:**')
        df_cmp = compare_coolprop_vs_gsssd(float(P_Pa))
        delta_col = df_cmp['δ, %'].str.replace(',', '.').astype(float)
        max_delta = delta_col.max()
        if max_delta < 3.0:
            st.success(f'Макс. расхождение: {max_delta:.2f}% (норма < 3%)')
        else:
            st.warning(f'Макс. расхождение: {max_delta:.2f}%')
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    with tab4:
        render_turbulence_tab(ui_params)

    with tab5:
        render_comparison_tab(ui_params)

    with tab6:
        render_verification_tab(ui_params)

    with tab7:
        render_qconst_tab(ui_params)

    with tab8:
        render_back_losses_tab(ui_params)


if __name__ == '__main__':
    main()
