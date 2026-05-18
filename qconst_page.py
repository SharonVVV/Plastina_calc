"""
Вкладка «Стенд q_w = const» — расчёт теплового баланса при граничном
условии 2-го рода (силиконовые нагреватели на тыльной стороне
теплораспределителя из АМг3 с фиксированной полной мощностью).

Полная электрическая мощность N_эл фиксирована, q_w = N_эл / (b·L) = const
не зависит от t_c, поэтому уравнение баланса упрощается:

    F(t_c) = q_w − q_конв(t_c) − q_рад(t_c) = 0

Использует тот же шаблон UI/расчёта/визуализации, что и turbulence_page.py
и comparison_page.py.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import (
    DEFAULTS_QCONST, RANGES_QCONST, DEFAULTS_QCONST_LOADS,
    DEFAULTS_ENV, RANGES_ENV, DEFAULTS_CALC,
    DEFAULTS_HEATER, GR_PR_CRIT_1, GR_PR_CRIT_2,
)
from solver import solve_plate, solve_plate_qconst, CalculationResult
from plotting import (
    plot_t_vs_x, plot_ra_vs_x, plot_heat_fluxes,
    plot_plate_composition,
)
from formatting import _fc, _fe


# ── Список 8 корреляций (как в главной странице app.py) ─────────────────────
_CORR_OPTIONS = {
    'kuznetov': 'Кузнецов (Nu, Φ(Pr), q=const)',
    'kerimov': 'Керимов (Nu_ж, ε_t)',
    'churchill_chu': 'Черчилль–Чу (средний Nu)',
    'leontiev': 'Леонтьев (Брдлик / Эккерт–Дж.)',
    'churchill_ozoe': 'Churchill & Ozoe (1973)',
    'vliet': 'Vliet (1969/1975)',
    'fujii': 'Fujii & Fujii (1976)',
    'isachenko': 'Исаченко и др. (1981)',
}

_DEFAULT_CRITS = {
    'kerimov': (1e9, 6e10),
    'kuznetov': (1e9, 1e12),
    'churchill_chu': (1e9, 6e10),
    'leontiev': (2e7, 2e7),
    'churchill_ozoe': (1e9, 6e10),
    'vliet': (3.4e9, 3.4e9),
    'fujii': (1e9, 6e10),
    'isachenko': (1e9, 6e10),
}


# ── Утилиты ─────────────────────────────────────────────────────────────────


def _eps_to_C_pr(eps: float) -> float:
    """C_пр = ε · σ₀, где σ₀ = 5.67 Вт/(м²·К⁴) — постоянная Стефана–Больцмана
    в форме «деление на 100⁴» (как используется в calc_q_rad)."""
    return eps * 5.67


def _x_crit_at_Ra(result: CalculationResult, Ra_target: float = 1e9) -> float | None:
    """Координата x, где Ra пересекает заданное значение (линейная интерполяция)."""
    pts = [p for p in result.points if p.converged]
    for i in range(1, len(pts)):
        if pts[i - 1].Ra <= Ra_target <= pts[i].Ra:
            x0, x1 = pts[i - 1].x_m, pts[i].x_m
            r0, r1 = pts[i - 1].Ra, pts[i].Ra
            if r1 == r0:
                return x0
            return x0 + (Ra_target - r0) * (x1 - x0) / (r1 - r0)
    return None


def _summary_row(result: CalculationResult, label: str) -> dict:
    pts = [p for p in result.points if p.converged]
    if not pts:
        return {'Нагрузка': label}
    t_max = max(p.t_c for p in pts)
    Ra_max = max(p.Ra for p in pts)
    x_crit = _x_crit_at_Ra(result, 1e9)
    last = pts[-1]
    regime_top = {
        'lam': 'лам.', 'trans': 'перех.',
        'turb': 'турб.', 'full': 'Ч-Ч',
    }.get(last.regime, '?')
    return {
        'Нагрузка': label,
        'q_w, Вт/м²': _fc(result.q_w, 1),
        't_w_max, °C': _fc(t_max, 2),
        'Ra_max': _fe(Ra_max),
        'x(Ra=10⁹), мм': _fc(x_crit * 1000, 1) if x_crit is not None else '—',
        'Режим у верха': regime_top,
        'α(L), Вт/(м²·К)': _fc(last.alpha, 2),
    }


def _series_overlay_chart(results: list[tuple[str, CalculationResult]]) -> go.Figure:
    """Оверлей t_c(x) для нескольких нагрузочных режимов."""
    fig = go.Figure()
    for label, r in results:
        pts = [p for p in r.points if p.converged]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p.x_m * 1000 for p in pts],
            y=[p.t_c for p in pts],
            mode='lines+markers',
            name=label,
            hovertemplate='x = %{x:.0f} мм<br>t_c = %{y:.1f} °C<extra></extra>',
        ))
    fig.update_layout(
        title='Профили t_c(x) для разных нагрузок',
        xaxis_title='x, мм',
        yaxis_title='t_c, °C',
        template='plotly_white',
        height=420,
        separators=', ',
    )
    return fig


def _two_stand_overlay(r_kerimov: CalculationResult,
                       r_qconst: CalculationResult) -> go.Figure:
    """Оверлей t_c(x) двух стендов на одном графике."""
    fig = go.Figure()
    pk = [p for p in r_kerimov.points if p.converged]
    pq = [p for p in r_qconst.points if p.converged]
    if pk:
        fig.add_trace(go.Scatter(
            x=[p.x_m * 1000 for p in pk],
            y=[p.t_c for p in pk],
            mode='lines+markers',
            name=f'Стенд Керимова (I = {r_kerimov.I:.0f} А)',
            line=dict(color='#1f77b4'),
        ))
    if pq:
        fig.add_trace(go.Scatter(
            x=[p.x_m * 1000 for p in pq],
            y=[p.t_c for p in pq],
            mode='lines+markers',
            name=f'Новый стенд (q_w = {r_qconst.q_w:.0f} Вт/м²)',
            line=dict(color='#d62728', dash='dash'),
        ))
    fig.update_layout(
        title='Сравнение t_c(x): I = const (Керимов) vs q_w = const',
        xaxis_title='x, мм',
        yaxis_title='t_c, °C',
        template='plotly_white',
        height=440,
        separators=', ',
    )
    return fig


# ── Основная функция вкладки ────────────────────────────────────────────────


def render_qconst_tab(ui_params: dict) -> None:
    """UI и расчёт для нового стенда (q_w = const)."""

    st.markdown('### Стенд с силиконовыми нагревателями (q_w = const)')
    st.caption(
        '4 силиконовых нагревателя на тыльной стороне теплораспределителя из АМг3, '
        'фиксированная полная мощность $N_{эл}$ → граничное условие 2-го рода '
        '$q_w = N_{эл}/A = \\mathrm{const}$.'
    )

    col_in, col_viz = st.columns([1, 2], gap='large')

    # ── Левая колонка: ввод параметров ──────────────────────────────────────
    with col_in:
        st.markdown('##### Источник тепла')
        input_mode = st.radio(
            'Способ задания',
            options=['Через N_эл', 'Через q_w напрямую'],
            horizontal=True, key='qc_input_mode',
        )

        if input_mode == 'Через N_эл':
            N_total_W = st.number_input(
                'N_эл (полная мощность), Вт',
                value=DEFAULTS_QCONST['N_total_W'],
                min_value=RANGES_QCONST['N_total_W'][0],
                max_value=RANGES_QCONST['N_total_W'][1],
                step=50.0, key='qc_N_total',
            )
            q_w_direct = None
        else:
            q_w_direct = st.number_input(
                'q_w, Вт/м²',
                value=DEFAULTS_QCONST['q_w_Wm2'],
                min_value=RANGES_QCONST['q_w_Wm2'][0],
                max_value=RANGES_QCONST['q_w_Wm2'][1],
                step=100.0, key='qc_q_w',
            )
            N_total_W = 0.0  # будет пересчитан после ввода b, L

        st.markdown('##### Пластина')
        c1, c2 = st.columns(2)
        with c1:
            b_mm = st.number_input(
                'b, мм', value=DEFAULTS_QCONST['b_mm'],
                min_value=RANGES_QCONST['b_mm'][0],
                max_value=RANGES_QCONST['b_mm'][1],
                step=5.0, key='qc_b_mm',
            )
        with c2:
            L_mm = st.number_input(
                'L, мм', value=DEFAULTS_QCONST['L_mm'],
                min_value=RANGES_QCONST['L_mm'][0],
                max_value=RANGES_QCONST['L_mm'][1],
                step=50.0, key='qc_L_mm',
            )

        A_m2 = (b_mm / 1000.0) * (L_mm / 1000.0)
        if input_mode == 'Через N_эл':
            q_w = N_total_W / A_m2
        else:
            q_w = q_w_direct
            N_total_W = q_w * A_m2

        st.caption(
            f'A = b·L = {_fc(A_m2, 4)} м² · '
            f'q_w = {_fc(q_w, 1)} Вт/м² · '
            f'N_эл = {_fc(N_total_W, 1)} Вт'
        )

        st.markdown('##### Покрытие и среда')
        eps_surface = st.slider(
            'ε (степень черноты)',
            min_value=float(RANGES_QCONST['eps_surface'][0]),
            max_value=float(RANGES_QCONST['eps_surface'][1]),
            value=float(DEFAULTS_QCONST['eps_surface']),
            step=0.01, key='qc_eps',
        )
        C_pr = _eps_to_C_pr(eps_surface)
        st.caption(f'C_пр = ε · 5.67 = {_fc(C_pr, 3)} Вт/(м²·К⁴)')

        c1, c2 = st.columns(2)
        with c1:
            t_fluid_C = st.number_input(
                't_ж, °C', value=float(DEFAULTS_ENV['t_fluid_C']),
                min_value=float(RANGES_ENV['t_fluid_C'][0]),
                max_value=float(RANGES_ENV['t_fluid_C'][1]),
                step=1.0, key='qc_t_fluid',
            )
        with c2:
            P_Pa = st.number_input(
                'P, Па', value=float(DEFAULTS_ENV['P_Pa']),
                min_value=float(RANGES_ENV['P_Pa'][0]),
                max_value=float(RANGES_ENV['P_Pa'][1]),
                step=100.0, key='qc_P_Pa',
            )
        g = DEFAULTS_ENV['g']

        st.markdown('##### Расчёт')
        c1, c2 = st.columns(2)
        with c1:
            N_pts = st.number_input(
                'Точек N', value=DEFAULTS_CALC['N'],
                min_value=10, max_value=500, step=10, key='qc_N',
            )
        with c2:
            x_min_mm = st.number_input(
                'x_min, мм', value=DEFAULTS_QCONST['x_min_mm'],
                min_value=1.0, max_value=100.0, step=1.0, key='qc_x_min',
            )

        corr_key = st.selectbox(
            'Корреляция',
            options=list(_CORR_OPTIONS.keys()),
            format_func=lambda k: _CORR_OPTIONS[k],
            index=0, key='qc_corr',
        )
        crit_1, crit_2 = _DEFAULT_CRITS.get(corr_key, (GR_PR_CRIT_1, GR_PR_CRIT_2))

        run_mode = st.radio(
            'Режим расчёта',
            options=['Одиночный', 'Серия 25/50/75/100%'],
            key='qc_run_mode',
        )

        do_calc = st.button(
            'Рассчитать', type='primary', use_container_width=True, key='qc_calc',
        )

    # Автопересчёт при смене параметров
    _params_tuple = (
        run_mode, corr_key, q_w, b_mm, L_mm, eps_surface,
        t_fluid_C, P_Pa, x_min_mm, N_pts, N_total_W,
    )
    if (st.session_state.get('_qc_last_params') != _params_tuple
            and 'qc_result' in st.session_state):
        do_calc = True
    st.session_state['_qc_last_params'] = _params_tuple

    # ── Расчёт ──────────────────────────────────────────────────────────────
    common_kwargs = dict(
        b_mm=float(b_mm), L_mm=float(L_mm),
        t_fluid_C=float(t_fluid_C), g=g, C_pr=C_pr, P_Pa=float(P_Pa),
        x_min_mm=float(x_min_mm), N=int(N_pts),
        correlation=corr_key, t_ref_mode='auto',
        crit_1=crit_1, crit_2=crit_2,
        eps_surface=eps_surface,
    )

    if do_calc:
        if run_mode == 'Одиночный':
            r = solve_plate_qconst(
                q_w=float(q_w), N_total_W=float(N_total_W), **common_kwargs,
            )
            st.session_state['qc_result'] = r
            st.session_state['qc_series'] = None
        else:
            series = []
            for frac in DEFAULTS_QCONST_LOADS:
                N_load = N_total_W * frac
                q_w_load = N_load / A_m2
                r = solve_plate_qconst(
                    q_w=float(q_w_load), N_total_W=float(N_load), **common_kwargs,
                )
                series.append((f'{int(frac * 100)}% ({q_w_load:.0f} Вт/м²)', r))
            st.session_state['qc_series'] = series
            # Для одиночного блока показа берём 100%-режим
            st.session_state['qc_result'] = series[-1][1]

    # ── Визуализация ────────────────────────────────────────────────────────
    with col_viz:
        result: CalculationResult | None = st.session_state.get('qc_result')
        series = st.session_state.get('qc_series')

        if result is None:
            st.info('Задайте параметры слева и нажмите «Рассчитать».')
        else:
            n_ok = sum(1 for p in result.points if p.converged)
            n_fail = sum(1 for p in result.points if not p.converged)
            st.caption(
                f'Корреляция: {result.correlation} · q_w = {result.q_w:.0f} Вт/м² · '
                f'сошлось {n_ok} из {result.N}'
            )
            if n_fail > 0:
                st.error(f'{n_fail} точек не сошлись.')

            st.plotly_chart(plot_plate_composition(result), use_container_width=True)

    # ── Нижняя секция: серия / профили / верификация ───────────────────────
    if st.session_state.get('qc_result') is None:
        return

    result = st.session_state['qc_result']
    series = st.session_state.get('qc_series')

    st.markdown('---')

    if series is not None:
        st.markdown('#### Серия нагрузок 25 / 50 / 75 / 100 %')
        st.plotly_chart(_series_overlay_chart(series), use_container_width=True)
        rows = [_summary_row(r, label) for label, r in series]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown('#### Профили (одиночный режим)')
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_t_vs_x(result), use_container_width=True)
        with c2:
            st.plotly_chart(plot_ra_vs_x(result), use_container_width=True)
        st.plotly_chart(plot_heat_fluxes(result), use_container_width=True)

        df_one = pd.DataFrame([_summary_row(result, 'текущий')])
        st.dataframe(df_one, use_container_width=True, hide_index=True)

    # ── Двустендовая верификация ───────────────────────────────────────────
    st.markdown('---')
    with st.expander('Двустендовая верификация (Керимов vs q_w = const)'):
        st.caption(
            'Сравнение: при заданном токе I в стенде Керимова берётся '
            'усреднённое q_эл по высоте и подаётся как q_w в новый решатель. '
            'В средней зоне пластины оба профиля должны совпадать с точностью '
            'до различий в радиационной составляющей.'
        )
        I_kerimov = st.number_input(
            'I (стенд Керимова), А',
            value=float(DEFAULTS_HEATER['I']),
            min_value=50.0, max_value=600.0, step=10.0,
            key='qc_verif_I',
        )

        # Используем геометрию стенда Керимова из дефолтов (а не нового стенда),
        # чтобы корректно сопоставить tonowy расчёт.
        b_kerimov = DEFAULTS_HEATER['b_mm']
        L_kerimov = DEFAULTS_HEATER['L_mm']

        try:
            r_kerimov = solve_plate(
                I=float(I_kerimov),
                R20=DEFAULTS_HEATER['R20'],
                alpha_R=DEFAULTS_HEATER['alpha_R'],
                b_mm=float(b_kerimov), L_mm=float(L_kerimov),
                t_fluid_C=float(t_fluid_C), g=g, C_pr=C_pr, P_Pa=float(P_Pa),
                x_min_mm=float(x_min_mm), N=int(N_pts),
                correlation=corr_key, t_ref_mode='auto',
                crit_1=crit_1, crit_2=crit_2,
            )
            ok_pts = [p for p in r_kerimov.points if p.converged]
            if not ok_pts:
                st.error('Решение Керимова не сошлось ни в одной точке.')
                return

            q_w_avg = sum(p.q_el for p in ok_pts) / len(ok_pts)

            r_eq = solve_plate_qconst(
                q_w=float(q_w_avg),
                N_total_W=float(q_w_avg * (b_kerimov / 1000.0) * (L_kerimov / 1000.0)),
                b_mm=float(b_kerimov), L_mm=float(L_kerimov),
                t_fluid_C=float(t_fluid_C), g=g, C_pr=C_pr, P_Pa=float(P_Pa),
                x_min_mm=float(x_min_mm), N=int(N_pts),
                correlation=corr_key, t_ref_mode='auto',
                crit_1=crit_1, crit_2=crit_2,
                eps_surface=eps_surface,
            )

            st.plotly_chart(_two_stand_overlay(r_kerimov, r_eq),
                            use_container_width=True)

            # Метрика max|Δt_c| в средней зоне x ∈ [0.2L, 0.8L]
            L_m = L_kerimov / 1000.0
            x_lo, x_hi = 0.2 * L_m, 0.8 * L_m
            pairs = []
            for pk, pq in zip(r_kerimov.points, r_eq.points):
                if pk.converged and pq.converged and x_lo <= pk.x_m <= x_hi:
                    pairs.append(abs(pk.t_c - pq.t_c))
            if pairs:
                st.metric(
                    'max |Δt_c| в средней зоне (0.2L … 0.8L)',
                    f'{max(pairs):.2f} K',
                )
            st.caption(
                f'Среднее q_эл по Керимову = {q_w_avg:.1f} Вт/м² → '
                f'подано как q_w в новый решатель.'
            )
        except Exception as e:
            st.error(f'Ошибка при верификации: {e}')
