"""
Методика 2: Churchill & Ozoe (1973).

Источник:
  Churchill S.W., Ozoe H. A Correlation for Laminar Free Convection
  From a Vertical Plate // J. Heat Transfer, 1973, Vol. 95(4), pp. 540-541.

Ламинарный режим. Итерационная формула через стандартный Ra_x.
Формула (11) из статьи — для UHF (q_w = const).
"""

from ..iteration import iterate_T_wall
from ..air_properties import get_props, calc_beta
from ..criteria import calc_Gr_star, calc_Ra_star

# Табличные точные значения Nu_x / Ra_x^(1/4) при UHF
EXACT_TABLE = {
    0.10: 0.336,
    1.0: 0.456,
    10.0: 0.524,
    100.0: 0.550,
    1000.0: 0.559,
}

RA_MAX_LAMINAR = 1e9


def _calc_Nu_churchill_ozoe_uhf(Ra, Pr, **extra):
    """
    Формула (11) Churchill & Ozoe (1973) для UHF:

    Nu_x = 0.563 * Ra_x^(1/4) / [1 + (0.437/Pr)^(9/16)]^(4/9)

    Только ламинарный режим.
    """
    if Ra <= 0:
        return 0.01, 'laminar'

    Nu = 0.563 * Ra ** 0.25 / (1.0 + (0.437 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)
    regime = 'laminar'

    if Ra > RA_MAX_LAMINAR:
        regime = 'out_of_range'

    return Nu, regime


def calc_churchill_ozoe(x, q_w, T_inf_C, P_Pa=101325.0, g=9.80665,
                        max_iter=50, tol=1e-3, **kwargs):
    """
    Расчёт локального Nu по методике Churchill & Ozoe (1973).

    Только ламинарный режим. Итерация через стандартный Ra.
    Свойства при плёночной температуре.

    Returns:
        dict с результатами.
    """
    result = iterate_T_wall(
        x, q_w, T_inf_C, _calc_Nu_churchill_ozoe_uhf,
        ref_temp_mode='film',
        P_Pa=P_Pa, g=g, max_iter=max_iter, tol=tol,
    )

    # Пересчитаем Gr* и Ra* для полноты
    T_film = (result['T_wall_C'] + T_inf_C) / 2.0
    props = get_props(T_film, P_Pa)
    beta = calc_beta(T_film)
    Gr_star = calc_Gr_star(g, beta, q_w, x, props.lam, props.nu)
    Ra_star = calc_Ra_star(Gr_star, props.Pr)
    result['Ra_star'] = Ra_star
    result['Gr_star'] = Gr_star

    return result
