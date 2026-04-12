"""
Методика 4: Fujii & Fujii (1976).

Источник:
  Fujii T., Fujii M. The Dependence of Local Nusselt Number on Prandtl
  Number in the Case of Free Convection Along a Vertical Surface with
  Uniform Heat Flux // Int. J. Heat Mass Transfer, 1976, Vol. 19, pp. 121-122.

Безытерационная формула через Gr*_x. Только ламинарный режим.
Физически обоснована: корректные асимптотики Pr -> 0 и Pr -> inf.
"""

import math
from ..air_properties import get_props, calc_beta, calc_alpha_thermal
from ..criteria import calc_Gr_star, calc_Ra_star


def calc_fujii(x, q_w, T_inf_C, P_Pa=101325.0, g=9.80665, **kwargs):
    """
    Расчёт локального Nu по методике Fujii & Fujii (1976).

    Nu_x = ((Gr*_x * Pr^2) / (4 + 9*sqrt(Pr) + 10*Pr))^(1/5)

    Только ламинарный режим.

    Returns:
        dict с ключами аналогично другим методикам.
    """
    T_wall = T_inf_C + 30.0

    for iteration in range(3):
        T_film = (T_wall + T_inf_C) / 2.0
        props = get_props(T_film, P_Pa)
        beta = calc_beta(T_film)

        Gr_star = calc_Gr_star(g, beta, q_w, x, props.lam, props.nu)
        Pr = props.Pr

        denom = 4.0 + 9.0 * math.sqrt(Pr) + 10.0 * Pr
        Nu = (Gr_star * Pr ** 2 / denom) ** 0.2

        alpha_x = Nu * props.lam / x
        T_wall = T_inf_C + q_w / alpha_x

    Ra_star = calc_Ra_star(Gr_star, Pr)

    return {
        'T_wall_C': T_wall,
        'props': props,
        'Nu': Nu,
        'alpha': alpha_x,
        'Ra_star': Ra_star,
        'Gr_star': Gr_star,
        'Ra': None,
        'Gr': None,
        'Pr': Pr,
        'beta': beta,
        'regime': 'laminar',
        'n_iter': 0,
        'converged': True,
    }
