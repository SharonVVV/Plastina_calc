"""
Методика 3: Vliet (1969) / Vliet & Ross (1975).

Источники:
  Vliet G.C. J. Heat Transfer, 1969, 91(4), pp. 511-516.
  Vliet G.C., Ross D.C. J. Heat Transfer, 1975, 97(4), pp. 549-554.

Безытерационная формула через модифицированное Ra*_x.
Коэффициенты подогнаны на воздухе.
"""

from ..air_properties import get_props, calc_beta, calc_alpha_thermal
from ..criteria import calc_Gr_star, calc_Ra_star

RA_STAR_CRIT = 1e13


def calc_vliet(x, q_w, T_inf_C, P_Pa=101325.0, g=9.80665, **kwargs):
    """
    Расчёт локального Nu по методике Vliet.

    Ламинарный (Ra*_x < 10^13):
        Nu_x = 0.60 * (Ra*_x)^0.2

    Турбулентный (Ra*_x >= 10^13):
        Nu_x = 0.568 * (Ra*_x)^0.22   (Vliet 1969)

    Returns:
        dict с ключами: T_wall_C, Nu, alpha, Ra_star, Gr_star, Pr, beta,
                        regime, n_iter, converged
    """
    # Начальное приближение: свойства при T_inf
    # Затем 2 итерации для уточнения T_film
    T_wall = T_inf_C + 30.0

    for iteration in range(3):
        T_film = (T_wall + T_inf_C) / 2.0
        props = get_props(T_film, P_Pa)
        beta = calc_beta(T_film)

        Gr_star = calc_Gr_star(g, beta, q_w, x, props.lam, props.nu)
        Ra_star = calc_Ra_star(Gr_star, props.Pr)

        if Ra_star < RA_STAR_CRIT:
            Nu = 0.60 * Ra_star ** 0.2
            regime = 'laminar'
        else:
            Nu = 0.568 * Ra_star ** 0.22
            regime = 'turbulent'

        alpha_x = Nu * props.lam / x
        T_wall = T_inf_C + q_w / alpha_x

    return {
        'T_wall_C': T_wall,
        'props': props,
        'Nu': Nu,
        'alpha': alpha_x,
        'Ra_star': Ra_star,
        'Gr_star': Gr_star,
        'Ra': None,
        'Gr': None,
        'Pr': props.Pr,
        'beta': beta,
        'regime': regime,
        'n_iter': 0,
        'converged': True,
    }
