"""
Расчёт критической координаты перехода к турбулентному режиму.

Прямая задача:  найти x_crit, где GrPr = 10⁹ (при заданных I, R20, b и т.д.)
Обратная задача: найти минимальный ток I для турбулентности при заданной высоте L.
"""

from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq

from properties import get_air_properties
from correlations import calc_beta, calc_Gr, calc_eps_t, calc_Nu, calc_alpha
from solver import heat_balance_residual, CORR_PIECEWISE
from config import GR_PR_CRIT, BRENTQ_LOW_OFFSET, BRENTQ_HIGH_OFFSET


@dataclass
class TurbulenceResult:
    """Результат анализа турбулентного режима."""
    found: bool          # True если x_crit найден
    x_crit_m: float      # критическая координата, м
    t_c_crit: float      # температура стенки при x_crit, °C
    GrPr_crit: float     # GrPr при x_crit (должно быть ≈ 10⁹)
    L_sufficient: bool   # True если L >= x_crit
    margin_m: float      # запас: L - x_crit, м
    message: str         # текстовое пояснение


def _solve_tc_at_x(x: float, t_fluid_C: float, P_Pa: float, g: float,
                   I: float, R20: float, b_m: float, alpha_R: float,
                   C_pr: float) -> float:
    """Решить тепловой баланс при заданном x, вернуть t_c."""
    t_lo = t_fluid_C + BRENTQ_LOW_OFFSET
    t_hi = t_fluid_C + BRENTQ_HIGH_OFFSET

    args = (t_fluid_C, P_Pa, g, I, R20, b_m, alpha_R, C_pr, CORR_PIECEWISE)

    f_lo = heat_balance_residual(t_lo, x, *args)
    f_hi = heat_balance_residual(t_hi, x, *args)
    if f_lo * f_hi > 0:
        t_hi = t_fluid_C + 1000.0
        f_hi = heat_balance_residual(t_hi, x, *args)
        if f_lo * f_hi > 0:
            raise ValueError(f'F не меняет знак на [{t_lo:.1f}, {t_hi:.1f}]')

    return brentq(heat_balance_residual, t_lo, t_hi, args=(x, *args), xtol=1e-8)


def _grpr_at_x(x: float, t_fluid_C: float, P_Pa: float, g: float,
               I: float, R20: float, b_m: float, alpha_R: float,
               C_pr: float) -> float:
    """Вычислить GrPr при заданном x (сначала решив баланс для t_c)."""
    t_c = _solve_tc_at_x(x, t_fluid_C, P_Pa, g, I, R20, b_m, alpha_R, C_pr)

    t_film = (t_c + t_fluid_C) / 2.0
    air = get_air_properties(t_film, P_Pa)
    beta = calc_beta(t_film)

    Gr = calc_Gr(g, beta, t_c - t_fluid_C, x, air.nu)
    return Gr * air.Pr


def find_x_crit(
    I: float, R20: float, alpha_R: float, b_m: float,
    t_fluid_C: float, g: float, C_pr: float, P_Pa: float,
    L_m: float,
    x_min: float = 0.001,
    x_max: float = 5.0,
) -> TurbulenceResult:
    """
    Найти координату x_crit, где GrPr достигает 10⁹.

    Args:
        L_m: фактическая высота пластины (для проверки достаточности)
        x_min, x_max: диапазон поиска, м
    """
    common_args = (t_fluid_C, P_Pa, g, I, R20, b_m, alpha_R, C_pr)

    try:
        grpr_min = _grpr_at_x(x_min, *common_args)
        grpr_max = _grpr_at_x(x_max, *common_args)
    except Exception as e:
        return TurbulenceResult(
            found=False, x_crit_m=0, t_c_crit=0, GrPr_crit=0,
            L_sufficient=False, margin_m=0,
            message=f'Ошибка расчёта: {e}',
        )

    if grpr_min >= GR_PR_CRIT:
        # Уже турбулентный у основания
        t_c = _solve_tc_at_x(x_min, *common_args)
        return TurbulenceResult(
            found=True, x_crit_m=x_min, t_c_crit=t_c, GrPr_crit=grpr_min,
            L_sufficient=True, margin_m=L_m - x_min,
            message='Турбулентный режим уже при x_min.',
        )

    if grpr_max < GR_PR_CRIT:
        # Адаптивное расширение
        for x_try in [10.0, 20.0]:
            try:
                if _grpr_at_x(x_try, *common_args) >= GR_PR_CRIT:
                    x_max = x_try
                    break
            except Exception:
                pass
        else:
            return TurbulenceResult(
                found=False, x_crit_m=0, t_c_crit=0, GrPr_crit=grpr_max,
                L_sufficient=False, margin_m=0,
                message=f'Турбулентный режим не достигается даже при x = {x_max*1000:.0f} мм. '
                        f'Максимальный GrPr = {grpr_max:.2e}.',
            )

    # Найти корень: GrPr(x) - 10⁹ = 0
    def residual(x):
        return _grpr_at_x(x, *common_args) - GR_PR_CRIT

    x_crit = brentq(residual, x_min, x_max, xtol=1e-6)
    t_c_crit = _solve_tc_at_x(x_crit, *common_args)
    grpr_crit = _grpr_at_x(x_crit, *common_args)

    sufficient = L_m >= x_crit
    margin = L_m - x_crit

    return TurbulenceResult(
        found=True,
        x_crit_m=x_crit,
        t_c_crit=t_c_crit,
        GrPr_crit=grpr_crit,
        L_sufficient=sufficient,
        margin_m=margin,
        message='Критическая координата найдена.' if sufficient
                else f'Пластина L = {L_m*1000:.0f} мм короче x_крит = {x_crit*1000:.0f} мм.',
    )


def find_I_for_turbulence(
    L_m: float, R20: float, alpha_R: float, b_m: float,
    t_fluid_C: float, g: float, C_pr: float, P_Pa: float,
    I_min: float = 10.0,
    I_max: float = 2000.0,
    n_scan: int = 40,
) -> Optional[float]:
    """
    Найти минимальный ток I, при котором x_crit <= L.

    x_crit(I) не монотонна (при больших I свойства воздуха ухудшают Gr),
    поэтому сначала сканируем диапазон, находим минимум x_crit,
    затем ищем корень на нужном участке.

    Returns:
        I_crit в амперах, или None если невозможно.
    """
    import numpy as np

    common = (R20, alpha_R, b_m, t_fluid_C, g, C_pr, P_Pa, L_m)

    # Сканируем x_crit(I) по диапазону токов
    I_arr = np.linspace(I_min, I_max, n_scan)
    xcrit_arr = []
    for I in I_arr:
        try:
            res = find_x_crit(I, *common)
            xcrit_arr.append(res.x_crit_m if res.found else float('inf'))
        except Exception:
            xcrit_arr.append(float('inf'))

    # Найти минимальный x_crit
    min_xcrit = min(xcrit_arr)

    if min_xcrit > L_m:
        return None  # турбулентность недостижима при любом токе

    # Найти первый I, где x_crit <= L_m
    for i, xc in enumerate(xcrit_arr):
        if xc <= L_m:
            # Уточнить brentq между I_arr[i-1] и I_arr[i]
            if i == 0:
                return float(I_arr[0])
            try:
                def residual(I):
                    r = find_x_crit(I, *common)
                    return (r.x_crit_m if r.found else 1e6) - L_m

                I_crit = brentq(residual, float(I_arr[i - 1]), float(I_arr[i]), xtol=0.5)
                return I_crit
            except Exception:
                return float(I_arr[i])

    return None
