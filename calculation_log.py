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
    st.markdown('**1. Определяющая (плёночная) температура:**')
    st.latex(
        rf't_{{\text{{опр}}}} = \frac{{t_{{c}} + t_{{\text{{ж}}}}}}{{2}} = '
        rf'\frac{{{_fcl(pt.t_c, 2)} + {_fcl(result.t_fluid_C, 2)}}}{{2}} = '
        rf'{_fcl(pt.t_film, 2)} \; °C'
    )
    T_film_K = pt.t_film + 273.15
    st.latex(rf'T_{{\text{{опр}}}} = {_fcl(pt.t_film, 2)} + 273{{,}}15 = {_fcl(T_film_K, 2)} \; К')

    # --- Свойства воздуха ---
    st.markdown('---')
    st.markdown(f'**2. Свойства воздуха (CoolProp, T = {_fc(T_film_K, 2)} К, P = {_fc(result.P_Pa, 0)} Па):**')
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
    if result.correlation == 'churchill_chu':
        # Черчилль–Чу
        st.markdown('**Корреляция Черчилля–Чу (полная):**')
        psi = (1.0 + (0.492 / pt.air.Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
        st.latex(
            rf'\psi(Pr) = \left(1 + \left(\frac{{0{{,}}492}}{{Pr}}\right)^{{9/16}}\right)^{{-16/9}} = '
            rf'{_fcl(psi, 4)}'
        )
        Ra_psi = pt.Ra * psi
        st.latex(
            rf'Nu_x = \left[0{{,}}825 + 0{{,}}387 \cdot \left(Ra_x \cdot \psi\right)^{{1/6}}\right]^2 = '
            rf'\left[0{{,}}825 + 0{{,}}387 \cdot \left({_fl(Ra_psi)}\right)^{{1/6}}\right]^2 = '
            rf'{_fcl(pt.Nu, 2)}'
        )
    else:
        # Кусочная по методичке
        st.latex(
            rf'\varepsilon_t = \left(\frac{{Pr_{{\infty}}}}{{Pr_c}}\right)^{{0{{,}}25}} = '
            rf'\left(\frac{{{_fcl(pt.Pr_inf, 4)}}}{{{_fcl(pt.Pr_c, 4)}}}\right)^{{0{{,}}25}} = '
            rf'{_fcl(pt.eps_t, 4)}'
        )

        regime_label = {'lam': 'ламинарный', 'turb': 'турбулентный', 'full': 'полный (Ч-Ч)'}.get(pt.regime, '?')
        st.markdown(f'**Режим:** {regime_label} (GrPr = {_fe(pt.GrPr)})')

        if pt.regime == 'lam':
            st.latex(
                rf'Nu_x = 0{{,}}60 \cdot (Gr \cdot Pr)^{{0{{,}}25}} \cdot \varepsilon_t = '
                rf'0{{,}}60 \cdot \left({_fl(pt.GrPr)}\right)^{{0{{,}}25}} \cdot {_fcl(pt.eps_t, 4)} = '
                rf'{_fcl(pt.Nu, 2)}'
            )
        else:
            st.latex(
                rf'Nu_x = 0{{,}}15 \cdot (Gr \cdot Pr)^{{1/3}} \cdot \varepsilon_t = '
                rf'0{{,}}15 \cdot \left({_fl(pt.GrPr)}\right)^{{1/3}} \cdot {_fcl(pt.eps_t, 4)} = '
                rf'{_fcl(pt.Nu, 2)}'
            )

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
