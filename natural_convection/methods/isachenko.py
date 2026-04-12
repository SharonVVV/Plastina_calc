"""
Методика 5: Исаченко, Осипова, Сукомел (1981).

Источник:
  Исаченко В.П., Осипова В.А., Сукомел А.С.
  Теплопередача. М.: Энергоиздат, 1981.

ВАЖНО: определяющая температура — T_ж = T_inf (НЕ плёночная).
Итерационная формула через стандартный Ra_x.
"""

from ..iteration import iterate_T_wall
from ..air_properties import get_props, calc_beta
from ..criteria import calc_Gr_star, calc_Ra_star

# Границы режимов
RA_CRIT_LAM = 1e9
RA_CRIT_TURB = 6e10


def _calc_Nu_isachenko(Ra, Pr, **extra):
    """
    Формулы Исаченко для UHF:

    Ламинарный (10^3 < Ra < 10^9):
        Nu_ж_x = 0.60 * Ra_ж^0.25 * eps_t

    Турбулентный (Ra > 6·10^10):
        Nu_ж_x = 0.15 * Ra_ж^(1/3) * eps_t

    Переходный (10^9 < Ra < 6·10^10):
        Используется турбулентная формула (по рекомендации Исаченко).

    eps_t = (Pr_ж / Pr_wall)^0.25
    """
    # Вычисление eps_t
    T_wall_C = extra.get('T_wall_C', None)
    T_inf_C = extra.get('T_inf_C', None)
    P_Pa = extra.get('P_Pa', 101325.0)

    eps_t = 1.0
    if T_wall_C is not None and T_inf_C is not None:
        props_inf = get_props(T_inf_C, P_Pa)
        props_wall = get_props(T_wall_C, P_Pa)
        if props_wall.Pr > 0:
            eps_t = (props_inf.Pr / props_wall.Pr) ** 0.25

    if Ra <= 0:
        return 0.01, 'laminar'

    if Ra < RA_CRIT_LAM:
        Nu = 0.60 * Ra ** 0.25 * eps_t
        return Nu, 'laminar'
    elif Ra >= RA_CRIT_TURB:
        Nu = 0.15 * Ra ** (1.0 / 3.0) * eps_t
        return Nu, 'turbulent'
    else:
        # Переходный режим — Исаченко рекомендует турбулентную формулу
        Nu = 0.15 * Ra ** (1.0 / 3.0) * eps_t
        return Nu, 'transitional'


def calc_isachenko(x, q_w, T_inf_C, P_Pa=101325.0, g=9.80665,
                   max_iter=50, tol=1e-3, **kwargs):
    """
    Расчёт локального Nu по методике Исаченко.

    Свойства при T_ж (температура среды).
    Поправка eps_t = (Pr_ж / Pr_wall)^0.25.

    Returns:
        dict с результатами.
    """
    result = iterate_T_wall(
        x, q_w, T_inf_C, _calc_Nu_isachenko,
        ref_temp_mode='fluid',
        P_Pa=P_Pa, g=g, max_iter=max_iter, tol=tol,
    )

    # Пересчитаем Gr* и Ra* для полноты
    props = get_props(T_inf_C, P_Pa)
    beta = calc_beta(T_inf_C)
    Gr_star = calc_Gr_star(g, beta, q_w, x, props.lam, props.nu)
    Ra_star = calc_Ra_star(Gr_star, props.Pr)
    result['Ra_star'] = Ra_star
    result['Gr_star'] = Gr_star

    return result
