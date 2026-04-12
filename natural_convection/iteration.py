"""
Общая итерационная процедура для определения температуры стенки.

Используется методиками, где формула Nu содержит стандартный Ra_x
(через dT = T_wall - T_inf), а не модифицированный Ra*.
"""

from .air_properties import get_props, calc_beta, calc_alpha_thermal
from .criteria import calc_Gr, calc_Ra


def iterate_T_wall(x, q_w, T_inf_C, calc_Nu_fn, ref_temp_mode,
                   P_Pa=101325.0, g=9.80665,
                   max_iter=50, tol=1e-3, T_wall_guess_offset=30.0):
    """
    Итерационное определение T_wall при заданном q_w.

    Алгоритм:
      1. Начальное приближение T_wall = T_inf + offset
      2. Свойства при T_ref (film или fluid)
      3. Ra_x = g * beta * dT * x^3 / (nu * a)
      4. Nu_x = calc_Nu_fn(Ra_x, Pr, props_extra)
      5. alpha_x = Nu_x * lam / x
      6. T_wall_new = T_inf + q_w / alpha_x
      7. Проверка сходимости

    Args:
        x: координата по высоте, м
        q_w: тепловой поток на стенке, Вт/м²
        T_inf_C: температура среды, °C
        calc_Nu_fn: callable(Ra, Pr, **extra) -> (Nu, regime_str)
            extra содержит props, T_wall_C для методик с eps_t
        ref_temp_mode: 'film' | 'fluid'
        P_Pa: давление, Па
        g: ускорение свободного падения, м/с²
        max_iter: максимум итераций
        tol: относительная сходимость по T_wall
        T_wall_guess_offset: начальное dT, °C

    Returns:
        dict с ключами: T_wall_C, props, Nu, alpha, Ra, Gr, Pr, beta,
                        regime, n_iter, converged
    """
    T_wall = T_inf_C + T_wall_guess_offset
    converged = False
    prev_dT = None

    for i in range(1, max_iter + 1):
        dT = T_wall - T_inf_C
        if dT <= 0:
            dT = 0.1
            T_wall = T_inf_C + dT

        # Определяющая температура
        if ref_temp_mode == 'film':
            T_ref = (T_wall + T_inf_C) / 2.0
        else:
            T_ref = T_inf_C

        props = get_props(T_ref, P_Pa)
        beta = calc_beta(T_ref)
        a = calc_alpha_thermal(props)

        Gr = calc_Gr(g, beta, dT, x, props.nu)
        Ra = Gr * props.Pr

        # Вызов корреляции
        extra = {
            'props': props,
            'T_wall_C': T_wall,
            'T_inf_C': T_inf_C,
            'P_Pa': P_Pa,
        }
        Nu, regime = calc_Nu_fn(Ra, props.Pr, **extra)

        if Nu <= 0:
            Nu = 0.01

        alpha_x = Nu * props.lam / x
        T_wall_new = T_inf_C + q_w / alpha_x

        # Под-релаксация при осцилляции
        new_dT = T_wall_new - T_inf_C
        if prev_dT is not None and abs(new_dT - dT) > abs(dT - prev_dT):
            T_wall_new = 0.5 * T_wall_new + 0.5 * T_wall
        prev_dT = dT

        # Проверка сходимости
        if abs(T_wall_new - T_wall) / max(abs(T_wall), 1.0) < tol:
            T_wall = T_wall_new
            converged = True
            break

        T_wall = T_wall_new

    # Финальный пересчёт при сошедшейся T_wall
    dT = T_wall - T_inf_C
    if ref_temp_mode == 'film':
        T_ref = (T_wall + T_inf_C) / 2.0
    else:
        T_ref = T_inf_C
    props = get_props(T_ref, P_Pa)
    beta = calc_beta(T_ref)
    Gr = calc_Gr(g, beta, dT, x, props.nu)
    Ra = Gr * props.Pr
    extra = {'props': props, 'T_wall_C': T_wall, 'T_inf_C': T_inf_C, 'P_Pa': P_Pa}
    Nu, regime = calc_Nu_fn(Ra, props.Pr, **extra)
    alpha_x = Nu * props.lam / x

    return {
        'T_wall_C': T_wall,
        'props': props,
        'Nu': Nu,
        'alpha': alpha_x,
        'Ra': Ra,
        'Gr': Gr,
        'Pr': props.Pr,
        'beta': beta,
        'regime': regime,
        'n_iter': i,
        'converged': converged,
    }
