"""
Методика 1: Леонтьев А.И. Теория тепломассообмена. Глава VII.

Ламинарный режим — приближённое решение Брдлика (безытерационное, через Gr*_x).
Турбулентный режим — формула Эккерта—Джексона (итерационная, через стандартный Ra).

Точное решение через таблицу Theta_0(Pr) для верификации.
"""

import math
import numpy as np
from scipy.interpolate import interp1d

from ..air_properties import get_props, calc_beta, calc_alpha_thermal
from ..criteria import calc_Gr_star, calc_Ra_star, calc_Gr, calc_Ra
from ..iteration import iterate_T_wall

# Таблица Theta_0(Pr) — из точного автомодельного решения (Леонтьев)
# Theta_0 отрицательный; знак минус учитывается в формуле
_THETA0_TABLE_PR = [0.01, 0.1, 0.7, 1.0, 10.0, 100.0, 1000.0]
_THETA0_TABLE_VAL = [-4.528, -2.7507, -1.5108, -1.3574, -0.76746, -0.46566, -0.2952]

_theta0_interp = interp1d(
    np.log(_THETA0_TABLE_PR), _THETA0_TABLE_VAL,
    kind='linear', fill_value='extrapolate',
)

# Граница ламинарного -> турбулентного (по Леонтьеву, через обычный Ra)
RA_CRIT_LEONTIEV = 2e7


def _theta0(Pr):
    """Интерполяция Theta_0(Pr) по таблице (линейно по ln(Pr))."""
    return float(_theta0_interp(math.log(Pr)))


def _nu_brdlik(Gr_star, Pr):
    """Приближённое решение Брдлика (ламинарный).

    Nu_x = 0.616 * (Pr / (Pr + 0.8))^(1/5) * (Gr*_x * Pr)^(1/5)
    """
    Ra_star = Gr_star * Pr
    return 0.616 * (Pr / (Pr + 0.8)) ** 0.2 * Ra_star ** 0.2


def _nu_exact(Gr_star, Pr):
    """Точное решение через Theta_0 (ламинарный).

    Nu_x = -1 / (5^(1/5) * Theta_0(Pr)) * Gr*_x^(1/5)
    """
    theta0 = _theta0(Pr)
    return -1.0 / (5.0 ** 0.2 * theta0) * Gr_star ** 0.2


def _nu_eckert_jackson(Ra, Pr):
    """Формула Эккерта—Джексона (турбулентный).

    Nu_x = 0.0295 * Ra^(2/5) * Pr^(1/15) * (1 + 0.494 * Pr^(2/3))^(-2/5)
    """
    return 0.0295 * Ra ** 0.4 * Pr ** (1.0 / 15.0) * (1.0 + 0.494 * Pr ** (2.0 / 3.0)) ** (-0.4)


def calc_leontiev(x, q_w, T_inf_C, P_Pa=101325.0, g=9.80665,
                  max_iter=50, tol=1e-3, use_exact=False, **kwargs):
    """
    Расчёт локального Nu по методике Леонтьева.

    Автоматически определяет режим:
      - Сначала рассчитывает Nu из ламинарной формулы
      - Определяет dT, Ra
      - Если Ra > Ra_crit — переключается на турбулентную (итерационную)

    Args:
        use_exact: если True, использует точное решение через Theta_0
                   вместо приближения Брдлика (для верификации).

    Returns:
        dict с результатами.
    """
    # Шаг 1: ламинарная формула (безытерационная)
    T_wall = T_inf_C + 30.0

    for iteration in range(3):
        T_film = (T_wall + T_inf_C) / 2.0
        props = get_props(T_film, P_Pa)
        beta = calc_beta(T_film)

        Gr_star = calc_Gr_star(g, beta, q_w, x, props.lam, props.nu)

        if use_exact:
            Nu_lam = _nu_exact(Gr_star, props.Pr)
        else:
            Nu_lam = _nu_brdlik(Gr_star, props.Pr)

        alpha_x = Nu_lam * props.lam / x
        T_wall = T_inf_C + q_w / alpha_x

    # Проверка режима через стандартный Ra
    dT = T_wall - T_inf_C
    Gr = calc_Gr(g, beta, dT, x, props.nu)
    Ra = calc_Ra(Gr, props.Pr)
    Ra_star = calc_Ra_star(Gr_star, props.Pr)

    if Ra <= RA_CRIT_LEONTIEV:
        return {
            'T_wall_C': T_wall,
            'props': props,
            'Nu': Nu_lam,
            'alpha': alpha_x,
            'Ra_star': Ra_star,
            'Gr_star': Gr_star,
            'Ra': Ra,
            'Gr': Gr,
            'Pr': props.Pr,
            'beta': beta,
            'regime': 'laminar',
            'n_iter': 0,
            'converged': True,
        }

    # Шаг 2: турбулентный режим — итерация через Эккерта-Джексона
    def _nu_fn(Ra_val, Pr_val, **extra):
        Nu = _nu_eckert_jackson(Ra_val, Pr_val)
        return Nu, 'turbulent'

    result = iterate_T_wall(
        x, q_w, T_inf_C, _nu_fn, ref_temp_mode='film',
        P_Pa=P_Pa, g=g, max_iter=max_iter, tol=tol,
    )

    # Пересчитаем Gr* и Ra* для полноты
    T_film = (result['T_wall_C'] + T_inf_C) / 2.0
    props_f = get_props(T_film, P_Pa)
    beta_f = calc_beta(T_film)
    Gr_star = calc_Gr_star(g, beta_f, q_w, x, props_f.lam, props_f.nu)
    Ra_star = calc_Ra_star(Gr_star, props_f.Pr)

    result['Ra_star'] = Ra_star
    result['Gr_star'] = Gr_star
    return result
