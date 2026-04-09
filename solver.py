"""
Решатель теплового баланса вертикальной нагреваемой пластины.

Для каждой координаты x решается нелинейное уравнение F(t_c) = 0:
    F(t_c) = q_эл(t_c) - q_конв(t_c) - q_рад(t_c)

Метод: scipy.optimize.brentq (гарантированная сходимость на отрезке).
Поддерживаются две корреляции: кусочная (по методичке) и Черчилль–Чу (полная).
"""

from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np
from scipy.optimize import brentq

from properties import get_air_properties, AirProperties
from correlations import (
    calc_beta, calc_Gr, calc_Ra, calc_eps_t, calc_Nu,
    calc_Nu_churchill_chu,
    calc_alpha, calc_q_el, calc_q_conv, calc_q_rad,
)
from config import BRENTQ_LOW_OFFSET, BRENTQ_HIGH_OFFSET


# Режимы корреляции
CORR_PIECEWISE = 'piecewise'       # По методичке (кусочная)
CORR_CHURCHILL_CHU = 'churchill_chu'  # Черчилль–Чу (полная)


@dataclass
class PointResult:
    """Полный результат расчёта в одном сечении x."""
    x_m: float

    # Решение
    t_c: float
    t_film: float

    # Свойства воздуха при t_film
    air: AirProperties

    # Pr при температуре стенки и среды (для ε_t)
    Pr_inf: float
    Pr_c: float

    # Критерии подобия
    beta: float
    Gr: float
    Ra: float
    GrPr: float
    eps_t: float
    Nu: float            # Nu по выбранной корреляции (используется для решения)
    regime: str          # 'lam', 'turb' (кусочная) или 'full' (Черчилль–Чу)

    # Сравнительные значения (по другой корреляции)
    Nu_alt: float        # Nu по альтернативной корреляции
    alpha_alt: float     # alpha по альтернативной корреляции

    # Теплообмен (по выбранной корреляции)
    alpha: float
    q_conv: float
    q_rad: float
    q_el: float

    # Баланс
    residual: float
    converged: bool
    error_msg: str


@dataclass
class CalculationResult:
    """Полный результат расчёта для всей пластины."""
    points: List[PointResult]
    I: float
    R20: float
    alpha_R: float
    b_m: float
    t_fluid_C: float
    g: float
    C_pr: float
    P_Pa: float
    L_m: float
    x_min_m: float
    N: int
    correlation: str     # 'piecewise' или 'churchill_chu'


def _calc_nu_by_mode(correlation: str, Ra: float, Pr: float,
                     GrPr: float, eps_t: float):
    """Вычислить Nu по выбранной корреляции. Возвращает (Nu, regime)."""
    if correlation == CORR_CHURCHILL_CHU:
        Nu = calc_Nu_churchill_chu(Ra, Pr)
        return Nu, 'full'
    else:
        return calc_Nu(GrPr, eps_t)


def heat_balance_residual(t_c: float, x: float, t_fluid_C: float, P_Pa: float,
                          g: float, I: float, R20: float, b_m: float,
                          alpha_R: float, C_pr: float,
                          correlation: str = CORR_PIECEWISE) -> float:
    """Функция невязки F(t_c) = q_эл - q_конв - q_рад.

    Публичная функция — используется также в turbulence_solver.
    """
    t_film = (t_c + t_fluid_C) / 2.0
    air = get_air_properties(t_film, P_Pa)
    beta = calc_beta(t_film)

    delta_T = t_c - t_fluid_C
    Gr = calc_Gr(g, beta, delta_T, x, air.nu)
    Ra = calc_Ra(Gr, air.Pr)
    GrPr = Gr * air.Pr

    # ε_t для кусочной корреляции
    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_c = get_air_properties(t_c, P_Pa)
    eps_t = calc_eps_t(air_inf.Pr, air_c.Pr)

    Nu, _ = _calc_nu_by_mode(correlation, Ra, air.Pr, GrPr, eps_t)
    alpha = calc_alpha(Nu, air.lam, x)

    q_e = calc_q_el(I, R20, b_m, t_c, alpha_R)
    q_c = calc_q_conv(alpha, t_c, t_fluid_C)
    q_r = calc_q_rad(C_pr, t_c + 273.15, t_fluid_C + 273.15)

    return q_e - q_c - q_r


def _compute_point(t_c: float, x: float, t_fluid_C: float, P_Pa: float,
                   g: float, I: float, R20: float, b_m: float,
                   alpha_R: float, C_pr: float,
                   correlation: str = CORR_PIECEWISE) -> PointResult:
    """Пересчёт ВСЕХ промежуточных величин для найденного t_c."""
    t_film = (t_c + t_fluid_C) / 2.0
    air = get_air_properties(t_film, P_Pa)
    beta = calc_beta(t_film)

    delta_T = t_c - t_fluid_C
    Gr = calc_Gr(g, beta, delta_T, x, air.nu)
    Ra = calc_Ra(Gr, air.Pr)
    GrPr = Gr * air.Pr

    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_c = get_air_properties(t_c, P_Pa)
    eps_t = calc_eps_t(air_inf.Pr, air_c.Pr)

    # Основная корреляция (использована для решения)
    Nu, regime = _calc_nu_by_mode(correlation, Ra, air.Pr, GrPr, eps_t)
    alpha = calc_alpha(Nu, air.lam, x)

    # Альтернативная корреляция (для сравнения)
    if correlation == CORR_CHURCHILL_CHU:
        Nu_alt, _ = calc_Nu(GrPr, eps_t)
    else:
        Nu_alt = calc_Nu_churchill_chu(Ra, air.Pr)
    alpha_alt = calc_alpha(Nu_alt, air.lam, x)

    q_e = calc_q_el(I, R20, b_m, t_c, alpha_R)
    q_c = calc_q_conv(alpha, t_c, t_fluid_C)
    q_r = calc_q_rad(C_pr, t_c + 273.15, t_fluid_C + 273.15)

    return PointResult(
        x_m=x, t_c=t_c, t_film=t_film, air=air,
        Pr_inf=air_inf.Pr if correlation == CORR_PIECEWISE else 0,
        Pr_c=air_c.Pr if correlation == CORR_PIECEWISE else 0,
        beta=beta, Gr=Gr, Ra=Ra, GrPr=GrPr,
        eps_t=eps_t if correlation == CORR_PIECEWISE else 0,
        Nu=Nu, regime=regime,
        Nu_alt=Nu_alt, alpha_alt=alpha_alt,
        alpha=alpha,
        q_conv=q_c, q_rad=q_r, q_el=q_e,
        residual=q_e - q_c - q_r,
        converged=True, error_msg='',
    )


def _make_error_point(x: float, msg: str) -> PointResult:
    """Заглушка для несошедшейся точки."""
    return PointResult(
        x_m=x, t_c=float('nan'), t_film=float('nan'),
        air=AirProperties(0, 0, 0, 0, 0, 0),
        Pr_inf=0, Pr_c=0,
        beta=0, Gr=0, Ra=0, GrPr=0, eps_t=0, Nu=0,
        regime='', Nu_alt=0, alpha_alt=0,
        alpha=0, q_conv=0, q_rad=0, q_el=0,
        residual=float('nan'), converged=False, error_msg=msg,
    )


def solve_plate(
    I: float, R20: float, alpha_R: float,
    b_mm: float, L_mm: float,
    t_fluid_C: float, g: float, C_pr: float, P_Pa: float,
    x_min_mm: float, N: int,
    correlation: str = CORR_PIECEWISE,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> CalculationResult:
    """Решение теплового баланса для всех сечений по высоте пластины."""
    b_m = b_mm / 1000.0
    L_m = L_mm / 1000.0
    x_min_m = x_min_mm / 1000.0

    x_array = np.linspace(x_min_m, L_m, N)
    points: List[PointResult] = []

    args = (t_fluid_C, P_Pa, g, I, R20, b_m, alpha_R, C_pr, correlation)

    for i, x in enumerate(x_array):
        t_lo = t_fluid_C + BRENTQ_LOW_OFFSET
        t_hi = t_fluid_C + BRENTQ_HIGH_OFFSET

        try:
            f_lo = heat_balance_residual(t_lo, x, *args)
            f_hi = heat_balance_residual(t_hi, x, *args)

            if f_lo * f_hi > 0:
                t_hi = t_fluid_C + 1000.0
                f_hi = heat_balance_residual(t_hi, x, *args)
                if f_lo * f_hi > 0:
                    points.append(_make_error_point(x,
                        f'F не меняет знак на [{t_lo:.1f}, {t_hi:.1f}]'))
                    if progress_callback:
                        progress_callback((i + 1) / N)
                    continue

            t_c = brentq(heat_balance_residual, t_lo, t_hi,
                         args=(x, *args), xtol=1e-8)
            pt = _compute_point(t_c, x, *args)

        except Exception as e:
            pt = _make_error_point(x, str(e))

        points.append(pt)
        if progress_callback:
            progress_callback((i + 1) / N)

    return CalculationResult(
        points=points,
        I=I, R20=R20, alpha_R=alpha_R,
        b_m=b_m, t_fluid_C=t_fluid_C,
        g=g, C_pr=C_pr, P_Pa=P_Pa,
        L_m=L_m, x_min_m=x_min_m, N=N,
        correlation=correlation,
    )
