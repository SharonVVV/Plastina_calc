"""
Решатель теплового баланса вертикальной нагреваемой пластины.

Для каждой координаты x решается нелинейное уравнение F(t_c) = 0:
    F(t_c) = q_эл(t_c) - q_конв(t_c) - q_рад(t_c)

Метод: scipy.optimize.brentq (гарантированная сходимость на отрезке).

Корреляции:
  kerimov       — методичка Керимова (свойства при t_ж, ε_t)
  kuznetov      — задачник Кузнецова (свойства при t_плёнки, Φ(Pr), q_c=const)
  churchill_chu — обобщённая Черчилля–Чу (средний Nu, свойства при t_плёнки)
"""

from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np
from scipy.optimize import brentq

from properties import get_air_properties, AirProperties
from correlations import (
    calc_beta, calc_Gr, calc_Ra, calc_eps_t,
    calc_Nu_kerimov, calc_Nu_kuznetov, calc_Nu_churchill_chu,
    calc_Nu_leontiev, calc_Nu_churchill_ozoe, calc_Nu_vliet,
    calc_Nu_fujii, calc_Nu_isachenko,
    calc_Nu,  # обратная совместимость
    calc_alpha, calc_q_el, calc_q_conv, calc_q_rad,
)
from config import (
    BRENTQ_LOW_OFFSET, BRENTQ_HIGH_OFFSET,
    GR_PR_CRIT_1, GR_PR_CRIT_2, T_REF_FLUID, T_REF_FILM,
)


# Режимы корреляции
CORR_KERIMOV = 'kerimov'            # Методичка Керимова
CORR_KUZNETOV = 'kuznetov'          # Задачник Кузнецова
CORR_CHURCHILL_CHU = 'churchill_chu' # Черчилль–Чу (обобщённая)
CORR_LEONTIEV = 'leontiev'          # Леонтьев (Брдлик / Эккерт—Джексон)
CORR_CHURCHILL_OZOE = 'churchill_ozoe'  # Churchill & Ozoe (1973)
CORR_VLIET = 'vliet'                # Vliet (1969/1975)
CORR_FUJII = 'fujii'                # Fujii & Fujii (1976)
CORR_ISACHENKO = 'isachenko'        # Исаченко и др. (1981)
# Обратная совместимость
CORR_PIECEWISE = 'kerimov'


@dataclass
class PointResult:
    """Полный результат расчёта в одном сечении x."""
    x_m: float
    t_c: float
    t_film: float

    # Свойства воздуха при определяющей температуре
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
    Nu: float
    regime: str          # 'lam', 'trans', 'turb', 'full'

    # Сравнительные значения
    Nu_alt: float
    alpha_alt: float

    # Теплообмен
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
    correlation: str
    t_ref_mode: str      # 'fluid' или 'film'
    crit_1: float
    crit_2: float


def _get_t_ref_mode(correlation: str, t_ref_mode: str) -> str:
    """Определяющая температура: для Керимова/Исаченко — t_ж, для остальных — t_плёнки.
    Если пользователь явно указал — использовать его выбор."""
    if t_ref_mode != 'auto':
        return t_ref_mode
    if correlation in (CORR_KERIMOV, CORR_ISACHENKO):
        return T_REF_FLUID
    return T_REF_FILM


def _get_ref_properties(t_c, t_fluid_C, P_Pa, t_ref_mode):
    """Получить свойства воздуха при определяющей температуре."""
    t_film = (t_c + t_fluid_C) / 2.0
    if t_ref_mode == T_REF_FLUID:
        air_ref = get_air_properties(t_fluid_C, P_Pa)
        beta = calc_beta(t_fluid_C)
    else:
        air_ref = get_air_properties(t_film, P_Pa)
        beta = calc_beta(t_film)
    return air_ref, beta, t_film


def _calc_nu_by_mode(correlation, Ra, Pr, GrPr, eps_t, crit_1, crit_2):
    """Вычислить Nu по выбранной корреляции. Возвращает (Nu, regime)."""
    if correlation == CORR_CHURCHILL_CHU:
        return calc_Nu_churchill_chu(Ra, Pr), 'full'
    elif correlation == CORR_KUZNETOV:
        return calc_Nu_kuznetov(Ra, Pr, crit_1, crit_2)
    elif correlation == CORR_LEONTIEV:
        return calc_Nu_leontiev(Ra, Pr, crit_1, crit_2)
    elif correlation == CORR_CHURCHILL_OZOE:
        return calc_Nu_churchill_ozoe(Ra, Pr)
    elif correlation == CORR_VLIET:
        return calc_Nu_vliet(Ra, Pr, crit_1, crit_2)
    elif correlation == CORR_FUJII:
        return calc_Nu_fujii(Ra, Pr)
    elif correlation == CORR_ISACHENKO:
        return calc_Nu_isachenko(GrPr, eps_t, crit_1, crit_2)
    else:  # kerimov
        return calc_Nu_kerimov(GrPr, eps_t, crit_1, crit_2)


def heat_balance_residual(t_c: float, x: float, t_fluid_C: float, P_Pa: float,
                          g: float, I: float, R20: float, b_m: float,
                          alpha_R: float, C_pr: float,
                          correlation: str = CORR_KERIMOV,
                          t_ref_mode: str = T_REF_FLUID,
                          crit_1: float = GR_PR_CRIT_1,
                          crit_2: float = GR_PR_CRIT_2) -> float:
    """Функция невязки F(t_c) = q_эл − q_конв − q_рад."""
    air_ref, beta, _ = _get_ref_properties(t_c, t_fluid_C, P_Pa, t_ref_mode)

    delta_T = t_c - t_fluid_C
    Gr = calc_Gr(g, beta, delta_T, x, air_ref.nu)
    Ra = calc_Ra(Gr, air_ref.Pr)
    GrPr = Gr * air_ref.Pr

    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_c = get_air_properties(t_c, P_Pa)
    eps_t = calc_eps_t(air_inf.Pr, air_c.Pr)

    Nu, _ = _calc_nu_by_mode(correlation, Ra, air_ref.Pr, GrPr, eps_t,
                             crit_1, crit_2)
    alpha = calc_alpha(Nu, air_ref.lam, x)

    q_e = calc_q_el(I, R20, b_m, t_c, alpha_R)
    q_c = calc_q_conv(alpha, t_c, t_fluid_C)
    q_r = calc_q_rad(C_pr, t_c + 273.15, t_fluid_C + 273.15)

    return q_e - q_c - q_r


def _compute_point(t_c, x, t_fluid_C, P_Pa, g, I, R20, b_m,
                   alpha_R, C_pr, correlation, t_ref_mode,
                   crit_1, crit_2) -> PointResult:
    """Пересчёт ВСЕХ промежуточных величин для найденного t_c."""
    air_ref, beta, t_film = _get_ref_properties(t_c, t_fluid_C, P_Pa, t_ref_mode)

    delta_T = t_c - t_fluid_C
    Gr = calc_Gr(g, beta, delta_T, x, air_ref.nu)
    Ra = calc_Ra(Gr, air_ref.Pr)
    GrPr = Gr * air_ref.Pr

    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_c = get_air_properties(t_c, P_Pa)
    eps_t = calc_eps_t(air_inf.Pr, air_c.Pr)

    Nu, regime = _calc_nu_by_mode(correlation, Ra, air_ref.Pr, GrPr, eps_t,
                                  crit_1, crit_2)
    alpha = calc_alpha(Nu, air_ref.lam, x)

    # Альтернативная корреляция (Черчилль–Чу для сравнения)
    Nu_alt = calc_Nu_churchill_chu(Ra, air_ref.Pr)
    alpha_alt = calc_alpha(Nu_alt, air_ref.lam, x)

    q_e = calc_q_el(I, R20, b_m, t_c, alpha_R)
    q_c = calc_q_conv(alpha, t_c, t_fluid_C)
    q_r = calc_q_rad(C_pr, t_c + 273.15, t_fluid_C + 273.15)

    return PointResult(
        x_m=x, t_c=t_c, t_film=t_film, air=air_ref,
        Pr_inf=air_inf.Pr, Pr_c=air_c.Pr,
        beta=beta, Gr=Gr, Ra=Ra, GrPr=GrPr,
        eps_t=eps_t,
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
    correlation: str = CORR_KERIMOV,
    t_ref_mode: str = 'auto',
    crit_1: float = GR_PR_CRIT_1,
    crit_2: float = GR_PR_CRIT_2,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> CalculationResult:
    """Решение теплового баланса для всех сечений по высоте пластины."""
    b_m = b_mm / 1000.0
    L_m = L_mm / 1000.0
    x_min_m = x_min_mm / 1000.0

    actual_t_ref = _get_t_ref_mode(correlation, t_ref_mode)

    x_array = np.linspace(x_min_m, L_m, N)
    points: List[PointResult] = []

    args = (t_fluid_C, P_Pa, g, I, R20, b_m, alpha_R, C_pr,
            correlation, actual_t_ref, crit_1, crit_2)

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
        t_ref_mode=actual_t_ref,
        crit_1=crit_1,
        crit_2=crit_2,
    )
