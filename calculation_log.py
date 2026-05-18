"""
Формирование подробного лога расчёта для вкладки «Ход расчёта».
Каждый шаг — формула с подставленными числовыми значениями (st.latex).
"""

import streamlit as st
import pandas as pd

from solver import PointResult, CalculationResult
from formatting import _fc, _fcl, _fe, _fl


def render_point_detail(result: CalculationResult, idx: int) -> None:
    """Вывод полной цепочки расчёта для точки с индексом idx."""
    if getattr(result, 'mode', 'kerimov') == 'qconst':
        _render_qconst_point_detail(result, idx)
        return

    pt: PointResult = result.points[idx]

    if not pt.converged:
        st.error(f'Точка x = {_fc(pt.x_m * 1000, 1)} мм: расчёт не сошёлся. {pt.error_msg}')
        return

    x_mm = pt.x_m * 1000
    st.markdown(f'### Сечение x = {_fc(x_mm, 1)} мм ({_fc(pt.x_m, 4)} м)')

    # --- Входные данные ---
    st.markdown('**Входные данные:**')
    st.markdown(
        f'$I$ = {_fc(result.I, 1)} А, '
        f'$R_{{20}}$ = {_fe(result.R20)} Ом/м, '
        f'$\\alpha_R$ = {_fe(result.alpha_R)} К⁻¹, '
        f'$b$ = {_fc(result.b_m * 1000, 1)} мм ({_fc(result.b_m, 4)} м), '
        f'$t_{{ж}}$ = {_fc(result.t_fluid_C, 1)} °C, '
        f'$g$ = {_fc(result.g, 5)} м/с², '
        f'$C_{{пр}}$ = {_fc(result.C_pr, 1)} Вт/(м²·К⁴), '
        f'$P$ = {_fc(result.P_Pa, 0)} Па'
    )

    st.success(f'**Решение:** $t_{{c,x}}$ = {_fc(pt.t_c, 2)} °C')

    # --- Определяющая температура ---
    st.markdown('---')
    if result.t_ref_mode == 'fluid':
        t_ref_val = result.t_fluid_C
        st.markdown('**1. Определяющая температура (t_ж — по методичке):**')
        st.latex(
            rf't_{{\text{{опр}}}} = t_{{\text{{ж}}}} = {_fcl(result.t_fluid_C, 2)} \; °C'
        )
    else:
        t_ref_val = pt.t_film
        st.markdown('**1. Определяющая (плёночная) температура:**')
        st.latex(
            rf't_{{\text{{опр}}}} = \frac{{t_{{c}} + t_{{\text{{ж}}}}}}{{2}} = '
            rf'\frac{{{_fcl(pt.t_c, 2)} + {_fcl(result.t_fluid_C, 2)}}}{{2}} = '
            rf'{_fcl(pt.t_film, 2)} \; °C'
        )
    T_ref_K = t_ref_val + 273.15
    st.latex(rf'T_{{\text{{опр}}}} = {_fcl(t_ref_val, 2)} + 273{{,}}15 = {_fcl(T_ref_K, 2)} \; К')

    # --- Свойства воздуха ---
    st.markdown('---')
    st.markdown(f'**2. Свойства воздуха (CoolProp, T = {_fc(T_ref_K, 2)} К, P = {_fc(result.P_Pa, 0)} Па):**')
    props_df = pd.DataFrame({
        'Свойство': ['ν', 'λ', 'Pr', 'μ', 'ρ', 'cₚ'],
        'Значение': [
            _fe(pt.air.nu),
            _fe(pt.air.lam),
            _fc(pt.air.Pr, 4),
            _fe(pt.air.mu),
            _fc(pt.air.rho, 4),
            _fc(pt.air.cp, 1),
        ],
        'Единица': ['м²/с', 'Вт/(м·К)', '—', 'Па·с', 'кг/м³', 'Дж/(кг·К)'],
    })
    st.table(props_df)

    # --- Критерии подобия ---
    st.markdown('---')
    st.markdown('**3. Критериальные числа:**')

    delta_T = pt.t_c - result.t_fluid_C

    st.latex(
        rf'\beta = \frac{{1}}{{t_{{\text{{опр}}}} + 273{{,}}15}} = '
        rf'\frac{{1}}{{{_fcl(pt.t_film, 2)} + 273{{,}}15}} = {_fl(pt.beta)} \; К^{{-1}}'
    )

    st.latex(
        rf'Gr_x = \frac{{g \cdot \beta \cdot (t_c - t_{{\text{{ж}}}}) \cdot x^3}}{{\nu^2}} = '
        rf'\frac{{{_fcl(result.g, 5)} \cdot {_fl(pt.beta)} \cdot {_fcl(delta_T, 2)} \cdot \left({_fl(pt.x_m)}\right)^3}}'
        rf'{{\left({_fl(pt.air.nu)}\right)^2}} = {_fl(pt.Gr)}'
    )

    st.latex(
        rf'Ra_x = Gr_x \cdot Pr = {_fl(pt.Gr)} \cdot {_fcl(pt.air.Pr, 4)} = {_fl(pt.Ra)}'
    )

    # Nu — зависит от выбранной корреляции
    regime_label = {
        'lam': 'ламинарный', 'trans': 'переходный',
        'turb': 'турбулентный', 'full': 'полный',
    }.get(pt.regime, '?')

    if result.correlation == 'churchill_chu':
        # ── Черчилль–Чу (средний Nu) ──
        st.markdown('**Корреляция Черчилля–Чу (средний Nu, формула 3.9):**')
        psi = (1.0 + (0.492 / pt.air.Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
        st.latex(
            rf'\Psi(Pr) = \left(1 + \left(\frac{{0{{,}}492}}{{Pr}}\right)^{{9/16}}\right)^{{-16/9}} = '
            rf'{_fcl(psi, 4)}'
        )
        Ra_psi = pt.Ra * psi
        st.latex(
            rf'\overline{{Nu}} = \left[0{{,}}825 + 0{{,}}387 \cdot \left(Ra \cdot \Psi\right)^{{1/6}}\right]^2 = '
            rf'\left[0{{,}}825 + 0{{,}}387 \cdot \left({_fl(Ra_psi)}\right)^{{1/6}}\right]^2 = '
            rf'{_fcl(pt.Nu, 2)}'
        )

    elif result.correlation == 'kuznetov':
        # ── Задачник Кузнецова (местный Nu, q_c=const) ──
        from correlations import calc_Phi
        Phi = calc_Phi(pt.air.Pr)
        st.markdown(f'**Задачник Кузнецова (формулы 3.4/3.8, q_c = const):**')
        st.latex(
            rf'\Phi(Pr) = \left(1 + \left(\frac{{0{{,}}437}}{{Pr}}\right)^{{9/16}}\right)^{{-16/9}} = '
            rf'{_fcl(Phi, 4)}'
        )
        st.markdown(f'**Режим:** {regime_label} (Ra = {_fe(pt.Ra)})')

        if pt.regime == 'lam':
            Ra_Phi = pt.Ra * Phi
            st.latex(
                rf'Nu_x = 0{{,}}563 \cdot \left[Ra_x \cdot \Phi(Pr)\right]^{{0{{,}}25}} = '
                rf'0{{,}}563 \cdot \left({_fl(Ra_Phi)}\right)^{{0{{,}}25}} = '
                rf'{_fcl(pt.Nu, 2)}'
            )
        elif pt.regime == 'turb':
            Ra_Phi = pt.Ra * Phi
            st.latex(
                rf'Nu_x = 0{{,}}15 \cdot \left[Ra_x \cdot \Phi(Pr)\right]^{{1/3}} = '
                rf'0{{,}}15 \cdot \left({_fl(Ra_Phi)}\right)^{{1/3}} = '
                rf'{_fcl(pt.Nu, 2)}'
            )
        else:
            st.markdown(f'*Интерполяция по lg(Ra) между ламинарной и турбулентной формулами*')
            st.latex(rf'Nu_x = {_fcl(pt.Nu, 2)}')

    else:
        # ── Методичка Керимова (местный Nu, свойства при t_ж) ──
        st.markdown('**Методичка Керимова (свойства при t_ж):**')
        st.latex(
            rf'\varepsilon_t = \left(\frac{{Pr_{{\infty}}}}{{Pr_c}}\right)^{{0{{,}}25}} = '
            rf'\left(\frac{{{_fcl(pt.Pr_inf, 4)}}}{{{_fcl(pt.Pr_c, 4)}}}\right)^{{0{{,}}25}} = '
            rf'{_fcl(pt.eps_t, 4)}'
        )
        st.markdown(f'**Режим:** {regime_label} (GrPr = {_fe(pt.GrPr)})')

        if pt.regime == 'lam':
            st.latex(
                rf'Nu_x = 0{{,}}60 \cdot (Gr \cdot Pr)^{{0{{,}}25}} \cdot \varepsilon_t = '
                rf'0{{,}}60 \cdot \left({_fl(pt.GrPr)}\right)^{{0{{,}}25}} \cdot {_fcl(pt.eps_t, 4)} = '
                rf'{_fcl(pt.Nu, 2)}'
            )
        elif pt.regime == 'turb':
            st.latex(
                rf'Nu_x = 0{{,}}15 \cdot (Gr \cdot Pr)^{{1/3}} \cdot \varepsilon_t = '
                rf'0{{,}}15 \cdot \left({_fl(pt.GrPr)}\right)^{{1/3}} \cdot {_fcl(pt.eps_t, 4)} = '
                rf'{_fcl(pt.Nu, 2)}'
            )
        else:
            st.markdown(f'*Интерполяция по lg(GrPr) между ламинарной и турбулентной формулами*')
            st.latex(rf'Nu_x = {_fcl(pt.Nu, 2)}')

    # --- Теплообмен ---
    st.markdown('---')
    st.markdown('**4. Коэффициент теплоотдачи:**')
    st.latex(
        rf'\alpha_x = \frac{{Nu_x \cdot \lambda}}{{x}} = '
        rf'\frac{{{_fcl(pt.Nu, 2)} \cdot {_fl(pt.air.lam)}}}{{{_fcl(pt.x_m, 4)}}} = '
        rf'{_fcl(pt.alpha, 2)} \; Вт/(м^2 \cdot К)'
    )

    st.markdown('---')
    st.markdown('**5. Тепловые потоки:**')

    st.latex(
        rf'q_{{\text{{конв}}}} = \alpha_x \cdot (t_c - t_{{\text{{ж}}}}) = '
        rf'{_fcl(pt.alpha, 2)} \cdot {_fcl(delta_T, 2)} = {_fcl(pt.q_conv, 1)} \; Вт/м^2'
    )

    T_c_K = pt.t_c + 273.15
    T_f_K = result.t_fluid_C + 273.15
    st.latex(
        rf'q_{{\text{{рад}}}} = C_{{\text{{пр}}}} \cdot '
        rf'\left[\left(\frac{{T_c}}{{100}}\right)^4 - \left(\frac{{T_{{\text{{ж}}}}}}{{100}}\right)^4\right] = '
        rf'{_fcl(result.C_pr, 1)} \cdot '
        rf'\left[{_fcl(T_c_K / 100, 4)}^4 - {_fcl(T_f_K / 100, 4)}^4\right] = '
        rf'{_fcl(pt.q_rad, 1)} \; Вт/м^2'
    )

    delta_T_20 = pt.t_c - 20.0
    st.latex(
        rf'q_{{\text{{эл}}}} = \frac{{I^2 \cdot R_{{20}}}}{{b}} '
        rf'\left[1 + (t_c - 20) \cdot \alpha_R \right] = '
        rf'\frac{{{_fcl(result.I, 1)}^2 \cdot {_fl(result.R20)}}}{{{_fcl(result.b_m, 4)}}} '
        rf'\cdot \left[1 + {_fcl(delta_T_20, 2)} \cdot {_fl(result.alpha_R)} \right] = '
        rf'{_fcl(pt.q_el, 1)} \; Вт/м^2'
    )

    # --- Баланс ---
    st.markdown('---')
    st.markdown('**6. Проверка баланса:**')
    st.latex(
        rf'q_{{\text{{эл}}}} - q_{{\text{{конв}}}} - q_{{\text{{рад}}}} = '
        rf'{_fcl(pt.q_el, 1)} - {_fcl(pt.q_conv, 1)} - {_fcl(pt.q_rad, 1)} = '
        rf'{_fl(pt.residual)} \; Вт/м^2'
    )
    if abs(pt.residual) < 0.01:
        st.success('Баланс выполнен ✓')
    else:
        st.warning(f'Невязка = {_fe(pt.residual)} Вт/м²')


def _render_qconst_point_detail(result: CalculationResult, idx: int) -> None:
    """Лог расчёта для режима q_w = const (силиконовые нагреватели)."""
    pt: PointResult = result.points[idx]

    if not pt.converged:
        st.error(f'Точка x = {_fc(pt.x_m * 1000, 1)} мм: расчёт не сошёлся. {pt.error_msg}')
        return

    x_mm = pt.x_m * 1000
    st.markdown(f'### Сечение x = {_fc(x_mm, 1)} мм ({_fc(pt.x_m, 4)} м) — режим q_w = const')

    st.markdown('**Входные данные:**')
    st.markdown(
        f'$N_{{эл}}$ = {_fc(result.N_total_W, 1)} Вт, '
        f'$A$ = {_fc(result.A_m2, 4)} м², '
        f'$q_w$ = {_fc(result.q_w, 1)} Вт/м², '
        f'$b$ = {_fc(result.b_m * 1000, 1)} мм, '
        f'$L$ = {_fc(result.L_m * 1000, 1)} мм, '
        f'$\\varepsilon$ = {_fc(result.eps_surface, 3)}, '
        f'$C_{{пр}}$ = {_fc(result.C_pr, 3)} Вт/(м²·К⁴), '
        f'$t_{{ж}}$ = {_fc(result.t_fluid_C, 1)} °C, '
        f'$P$ = {_fc(result.P_Pa, 0)} Па'
    )
    st.success(f'**Решение:** $t_{{c,x}}$ = {_fc(pt.t_c, 2)} °C')

    # --- 1. Постоянная плотность теплового потока ---
    st.markdown('---')
    st.markdown('**1. Граничное условие 2-го рода (q_w = const):**')
    st.latex(
        rf'q_w = \frac{{N_{{эл}}}}{{b \cdot L}} = '
        rf'\frac{{{_fcl(result.N_total_W, 1)}}}{{{_fcl(result.b_m, 4)} \cdot {_fcl(result.L_m, 4)}}} = '
        rf'{_fcl(result.q_w, 1)} \; Вт/м^2 = \mathrm{{const}}'
    )
    st.markdown('*(не зависит от $t_c$ — нет ТКС в цепи питания силиконовых нагревателей)*')

    # --- 2. Определяющая температура ---
    st.markdown('---')
    if result.t_ref_mode == 'fluid':
        t_ref_val = result.t_fluid_C
        st.markdown('**2. Определяющая температура (t_ж):**')
        st.latex(rf't_{{\text{{опр}}}} = t_{{\text{{ж}}}} = {_fcl(result.t_fluid_C, 2)} \; °C')
    else:
        t_ref_val = pt.t_film
        st.markdown('**2. Определяющая (плёночная) температура:**')
        st.latex(
            rf't_{{\text{{опр}}}} = \frac{{t_{{c}} + t_{{\text{{ж}}}}}}{{2}} = '
            rf'\frac{{{_fcl(pt.t_c, 2)} + {_fcl(result.t_fluid_C, 2)}}}{{2}} = '
            rf'{_fcl(pt.t_film, 2)} \; °C'
        )
    T_ref_K = t_ref_val + 273.15
    st.latex(rf'T_{{\text{{опр}}}} = {_fcl(T_ref_K, 2)} \; К')

    # --- 3. Свойства воздуха ---
    st.markdown('---')
    st.markdown(f'**3. Свойства воздуха (CoolProp, T = {_fc(T_ref_K, 2)} К):**')
    props_df = pd.DataFrame({
        'Свойство': ['ν', 'λ', 'Pr', 'μ', 'ρ', 'cₚ'],
        'Значение': [
            _fe(pt.air.nu), _fe(pt.air.lam), _fc(pt.air.Pr, 4),
            _fe(pt.air.mu), _fc(pt.air.rho, 4), _fc(pt.air.cp, 1),
        ],
        'Единица': ['м²/с', 'Вт/(м·К)', '—', 'Па·с', 'кг/м³', 'Дж/(кг·К)'],
    })
    st.table(props_df)

    # --- 4. Критерии подобия ---
    st.markdown('---')
    st.markdown('**4. Критериальные числа:**')
    delta_T = pt.t_c - result.t_fluid_C
    st.latex(rf'\beta = {_fl(pt.beta)} \; К^{{-1}}')
    st.latex(rf'Gr_x = {_fl(pt.Gr)}, \quad Ra_x = Gr_x \cdot Pr = {_fl(pt.Ra)}')

    regime_label = {
        'lam': 'ламинарный', 'trans': 'переходный',
        'turb': 'турбулентный', 'full': 'полный',
    }.get(pt.regime, '?')
    st.markdown(f'**Корреляция:** `{result.correlation}` — режим: {regime_label}')
    st.latex(rf'Nu_x = {_fcl(pt.Nu, 2)}')

    # --- 5. Теплоотдача ---
    st.markdown('---')
    st.markdown('**5. Коэффициент теплоотдачи:**')
    st.latex(
        rf'\alpha_x = \frac{{Nu_x \cdot \lambda}}{{x}} = '
        rf'\frac{{{_fcl(pt.Nu, 2)} \cdot {_fl(pt.air.lam)}}}{{{_fcl(pt.x_m, 4)}}} = '
        rf'{_fcl(pt.alpha, 2)} \; Вт/(м^2 \cdot К)'
    )

    # --- 6. Тепловые потоки ---
    st.markdown('---')
    st.markdown('**6. Тепловые потоки:**')
    st.latex(
        rf'q_{{\text{{конв}}}} = \alpha_x \cdot (t_c - t_{{\text{{ж}}}}) = '
        rf'{_fcl(pt.alpha, 2)} \cdot {_fcl(delta_T, 2)} = {_fcl(pt.q_conv, 1)} \; Вт/м^2'
    )

    T_c_K = pt.t_c + 273.15
    T_f_K = result.t_fluid_C + 273.15
    st.latex(
        rf'q_{{\text{{рад}}}} = C_{{\text{{пр}}}} \cdot '
        rf'\left[\left(\frac{{T_c}}{{100}}\right)^4 - \left(\frac{{T_{{\text{{ж}}}}}}{{100}}\right)^4\right] = '
        rf'{_fcl(result.C_pr, 3)} \cdot '
        rf'\left[{_fcl(T_c_K / 100, 4)}^4 - {_fcl(T_f_K / 100, 4)}^4\right] = '
        rf'{_fcl(pt.q_rad, 1)} \; Вт/м^2'
    )
    st.latex(rf'q_w = {_fcl(pt.q_el, 1)} \; Вт/м^2 \quad (\text{{задано}})')

    # --- 7. Баланс ---
    st.markdown('---')
    st.markdown('**7. Проверка баланса:**')
    st.latex(
        rf'q_w - q_{{\text{{конв}}}} - q_{{\text{{рад}}}} = '
        rf'{_fcl(pt.q_el, 1)} - {_fcl(pt.q_conv, 1)} - {_fcl(pt.q_rad, 1)} = '
        rf'{_fl(pt.residual)} \; Вт/м^2'
    )
    if abs(pt.residual) < 0.01:
        st.success('Баланс выполнен ✓')
    else:
        st.warning(f'Невязка = {_fe(pt.residual)} Вт/м²')
