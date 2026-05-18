"""
Расчётчик Задачи №1 — высота достижения турбулентного режима естественной
конвекции на вертикальной плите.

Оба варианта нагрева — Джоулев и силиконовые маты — это UHF (q_w = const):
  • Джоулев: q_w(T_s) = (I²·R₂₀ / b)·[1 + (T_s − 20)·α_R]
  • Маты:    q_w     = N_total / (b·L)

Для каждой методики (корреляции из correlations_uhf) считаем профиль T_s(x)
итерацией баланса:
    F(T_s) = q_w(T_s) − α(T_s, x)·(T_s − T_∞) − ε·σ·(T_s⁴ − T_∞⁴) = 0
где α(T_s, x) = Nu_x·λ/x и Nu_x определяется выбранной корреляцией.

Свойства воздуха берутся при опорной температуре, заданной методикой
(см. `UHFCorrelation.t_ref` в correlations_uhf):
  • 'film'  — T_f = (T_s + T_∞)/2 — стандарт у Bejan, Holman, Jiji, Vliet,
    Леонтьев, ЦКВ;
  • 'fluid' — T_∞ — Керимов 1992 (МЭИ).
Дополнительно вычисляется ε_t = (Pr_ж/Pr_c)^0.25; передаётся в compute_nu
и используется только Керимовым 1992 — у остальных методик игнорируется.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from scipy.optimize import brentq

from properties import get_air_properties
from correlations_uhf import compute_nu, calc_Gr, calc_Gr_star, t_ref_for
from correlations_meta import META, CorrelationMeta


SIGMA = 5.670374419e-8   # постоянная Стефана-Больцмана, Вт/(м²·К⁴)


# ── Связь входных параметров и q_w ─────────────────────────────────────────

def q_w_from_N(N_total_W: float, b_mm: float, L_mm: float) -> float:
    """Поверхностная плотность теплового потока для силиконовых матов, Вт/м²."""
    return N_total_W / ((b_mm / 1000.0) * (L_mm / 1000.0))


def q_w_from_I_at_T(I: float, R20: float, b_mm: float,
                    t_s_C: float, alpha_R: float) -> float:
    """Поверхностная плотность теплового потока для Джоулева нагрева, Вт/м²,
    при заданной температуре стенки t_s_C (учёт ТКС)."""
    return (I ** 2 * R20 / (b_mm / 1000.0)) * (1.0 + (t_s_C - 20.0) * alpha_R)


# ── Точечный результат ─────────────────────────────────────────────────────

@dataclass
class PointUHF:
    x_m: float
    t_s_C: float
    t_film_C: float
    q_w: float          # фактический q_w на стенке (W/m²)
    Gr_x: float
    Ra_x: float
    Gr_x_star: float
    GrPr_star: float
    Nu_x: float
    alpha: float
    regime: str         # 'lam' / 'trans' / 'turb' / 'out'
    converged: bool
    error_msg: str = ''


@dataclass
class MethodologyResult:
    key: str
    meta: CorrelationMeta
    points: list[PointUHF]
    x_lam_to_trans_m: Optional[float] = None
    x_trans_to_turb_m: Optional[float] = None
    # удобный синоним «координата, где режим стал турбулентным»:
    x_turb_start_m: Optional[float] = None

    @property
    def achieves_turb(self) -> bool:
        return self.x_turb_start_m is not None


# ── Решение в одной точке x ────────────────────────────────────────────────

def _q_provider_joule(I: float, R20: float, b_mm: float,
                      alpha_R: float) -> Callable[[float], float]:
    """Возвращает q_w(T_s) для Джоулева нагрева."""
    def _q(t_s_C: float) -> float:
        return q_w_from_I_at_T(I, R20, b_mm, t_s_C, alpha_R)
    return _q


def _q_provider_qconst(q_w: float) -> Callable[[float], float]:
    """Возвращает константный q_w (силиконовые маты)."""
    def _q(_t_s_C: float) -> float:
        return q_w
    return _q


def _reference_air(t_s_C: float, t_fluid_C: float, P_Pa: float,
                   t_ref: str) -> tuple:
    """Возвращает (air_ref, beta_ref, t_ref_C) при опорной температуре,
    которую запрашивает методика ('film' либо 'fluid')."""
    if t_ref == 'fluid':
        t_ref_C = t_fluid_C
    else:  # 'film'
        t_ref_C = 0.5 * (t_s_C + t_fluid_C)
    air = get_air_properties(t_ref_C, P_Pa)
    beta = 1.0 / (t_ref_C + 273.15)
    return air, beta, t_ref_C


def _balance_residual(t_s_C: float, *, x_m: float, key: str,
                      q_provider: Callable[[float], float],
                      t_fluid_C: float, P_Pa: float, g: float,
                      eps_surface: float) -> float:
    """F(T_s) = q_w(T_s) − α·(T_s − T_∞) − ε·σ·(T⁴ − T_∞⁴).

    Свойства воздуха берутся при опорной температуре, заданной методикой
    (плёночная либо t_ж). ε_t = (Pr_ж/Pr_c)^0.25 — поправка Керимова 1992
    на переменность свойств, вычисляется всегда (для остальных методик она
    мульпипликативно «1» и игнорируется в compute_nu).
    """
    t_ref = t_ref_for(key)
    air, beta, _ = _reference_air(t_s_C, t_fluid_C, P_Pa, t_ref)

    # Pr при t_ж и t_c — для ε_t.
    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_wall = get_air_properties(t_s_C, P_Pa)
    eps_t = (air_inf.Pr / air_wall.Pr) ** 0.25 if air_wall.Pr > 0 else 1.0

    delta_T = t_s_C - t_fluid_C
    q_w = q_provider(t_s_C)
    Gr_star = calc_Gr_star(g, beta, q_w, x_m, air.lam, air.nu)
    Gr = calc_Gr(g, beta, delta_T, x_m, air.nu)
    Ra = Gr * air.Pr

    nu_res = compute_nu(key, Ra=Ra, Gr_star=Gr_star, Pr=air.Pr, eps_t=eps_t)
    alpha = nu_res.Nu * air.lam / x_m

    q_conv = alpha * delta_T
    T_s_K = t_s_C + 273.15
    T_inf_K = t_fluid_C + 273.15
    q_rad = eps_surface * SIGMA * (T_s_K ** 4 - T_inf_K ** 4)

    return q_w - q_conv - q_rad


# Верхний потолок T_s при поиске Brent. CoolProp Air надёжен до T ≈ 2000 K
# (≈1700 °C); ставим 1500 °C с запасом. Для реального стенда (q_w ≤ 50 кВт/м²)
# T_s не превышает 800–1000 °C.
_T_HI_PRIMARY = 1500.0
_T_HI_FALLBACK = 1700.0
_T_LO_OFFSET = 0.01   # T_∞ + 0.01 °C — нижняя точка, при которой q_conv ≈ 0


def _solve_point(*, x_m: float, key: str,
                 q_provider: Callable[[float], float],
                 t_fluid_C: float, P_Pa: float, g: float,
                 eps_surface: float) -> PointUHF:
    """Решает баланс для одной x; возвращает PointUHF."""
    args = dict(x_m=x_m, key=key, q_provider=q_provider,
                t_fluid_C=t_fluid_C, P_Pa=P_Pa, g=g,
                eps_surface=eps_surface)
    lo = t_fluid_C + _T_LO_OFFSET
    hi = t_fluid_C + _T_HI_PRIMARY

    def _f(t):
        return _balance_residual(t, **args)

    try:
        f_lo = _f(lo)
        f_hi = _f(hi)
        if f_lo * f_hi > 0:
            hi = t_fluid_C + _T_HI_FALLBACK
            f_hi = _f(hi)
            if f_lo * f_hi > 0:
                return PointUHF(
                    x_m=x_m, t_s_C=float('nan'), t_film_C=float('nan'),
                    q_w=float('nan'), Gr_x=0, Ra_x=0, Gr_x_star=0, GrPr_star=0,
                    Nu_x=0, alpha=0, regime='', converged=False,
                    error_msg=f'F не меняет знак на [{lo:.2f}, {hi:.0f}] °C',
                )
        t_s = brentq(_f, lo, hi, xtol=1e-6)
    except Exception as e:
        return PointUHF(
            x_m=x_m, t_s_C=float('nan'), t_film_C=float('nan'),
            q_w=float('nan'), Gr_x=0, Ra_x=0, Gr_x_star=0, GrPr_star=0,
            Nu_x=0, alpha=0, regime='', converged=False, error_msg=str(e),
        )

    # Пересчёт всех величин для найденного T_s — при той же опорной T,
    # которую запрашивает методика.
    t_ref = t_ref_for(key)
    air, beta, t_ref_C = _reference_air(t_s, t_fluid_C, P_Pa, t_ref)
    # Для отчётности удобнее всегда хранить плёночную:
    t_film = 0.5 * (t_s + t_fluid_C)
    delta_T = t_s - t_fluid_C
    q_w = q_provider(t_s)
    Gr_star = calc_Gr_star(g, beta, q_w, x_m, air.lam, air.nu)
    Gr = calc_Gr(g, beta, delta_T, x_m, air.nu)
    Ra = Gr * air.Pr
    # ε_t для Керимова 1992; для остальных методик не используется.
    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_wall = get_air_properties(t_s, P_Pa)
    eps_t = (air_inf.Pr / air_wall.Pr) ** 0.25 if air_wall.Pr > 0 else 1.0
    nu_res = compute_nu(key, Ra=Ra, Gr_star=Gr_star, Pr=air.Pr, eps_t=eps_t)
    alpha = nu_res.Nu * air.lam / x_m

    return PointUHF(
        x_m=x_m, t_s_C=t_s, t_film_C=t_film, q_w=q_w,
        Gr_x=Gr, Ra_x=Ra, Gr_x_star=Gr_star, GrPr_star=Gr_star * air.Pr,
        Nu_x=nu_res.Nu, alpha=alpha,
        regime=nu_res.regime, converged=True,
    )


# ── Прогон по высоте для одной методики ────────────────────────────────────

def _wrap_methodology(key: str, points: list[PointUHF]) -> MethodologyResult:
    """Считает границы режимов по точкам и заворачивает в MethodologyResult.

    Учитывает три типа границ:
      • lam → trans  — координата первой transition-точки;
      • trans → turb — координата первой turbulent-точки после transition;
      • lam → turb   — прямой стык (методики без trans-зоны: Bejan/Vliet-Liu,
                       Bejan/air, Леонтьев). В этом случае x_lam_to_trans_m
                       проставляется равным координате стыка для отображения
                       в UI «здесь произошёл переход».
    """
    converged = [p for p in points if p.converged]
    x_trans = None
    x_turb = None
    prev_regime = None
    for p in converged:
        if x_trans is None and p.regime == 'trans':
            x_trans = p.x_m
        if x_turb is None and p.regime == 'turb':
            x_turb = p.x_m
            # Стыковая граница lam→turb без transition-зоны.
            if x_trans is None and prev_regime == 'lam':
                x_trans = p.x_m
            break
        prev_regime = p.regime
    return MethodologyResult(
        key=key, meta=META[key], points=points,
        x_lam_to_trans_m=x_trans,
        x_trans_to_turb_m=x_turb,
        x_turb_start_m=x_turb,
    )


def run_methodology_joule(
    key: str, *,
    I: float, R20: float, alpha_R: float,
    b_mm: float, L_mm: float,
    t_fluid_C: float, g: float, P_Pa: float,
    eps_surface: float,
    x_min_mm: float, N: int,
) -> MethodologyResult:
    """Полный прогон одной методики при Джоулевом нагреве."""
    L_m = L_mm / 1000.0
    x_min_m = x_min_mm / 1000.0

    q_provider = _q_provider_joule(I, R20, b_mm, alpha_R)
    xs = np.linspace(x_min_m, L_m, N)
    pts = [_solve_point(
        x_m=float(x), key=key, q_provider=q_provider,
        t_fluid_C=t_fluid_C, P_Pa=P_Pa, g=g, eps_surface=eps_surface,
    ) for x in xs]
    return _wrap_methodology(key, pts)


def run_methodology_qconst(
    key: str, *,
    N_total_W: float,
    b_mm: float, L_mm: float,
    t_fluid_C: float, g: float, P_Pa: float,
    eps_surface: float,
    x_min_mm: float, N: int,
) -> MethodologyResult:
    """Полный прогон одной методики при нагреве матами (q_w = N/(b·L))."""
    L_m = L_mm / 1000.0
    x_min_m = x_min_mm / 1000.0
    q_w = q_w_from_N(N_total_W, b_mm, L_mm)

    q_provider = _q_provider_qconst(q_w)
    xs = np.linspace(x_min_m, L_m, N)
    pts = [_solve_point(
        x_m=float(x), key=key, q_provider=q_provider,
        t_fluid_C=t_fluid_C, P_Pa=P_Pa, g=g, eps_surface=eps_surface,
    ) for x in xs]
    return _wrap_methodology(key, pts)


# ── Обратная задача: минимальный I / N для турбулента на target·L ──────────

@dataclass
class InverseSearchResult:
    achievable: bool
    value: Optional[float]      # I (А) либо N (Вт)
    x_turb_m: Optional[float]
    n_iter: int
    message: str = ''


def find_min_input_for_turbulence(
    *, mode: str, key: str, target_x_m: float,
    b_mm: float, L_mm: float,
    t_fluid_C: float, g: float, P_Pa: float,
    eps_surface: float,
    x_min_mm: float, N: int,
    # для Джоуля:
    R20: float = 0.0, alpha_R: float = 0.0,
    # диапазон поиска:
    search_min: float, search_max: float,
    n_scan: int = 12, bisect_iter: int = 12, bisect_tol: float = 0.5,
) -> InverseSearchResult:
    """
    Грубое сканирование + бисекция минимального I (либо N) при котором
    координата начала турбулентного режима ≤ target_x_m.

    Для методик без верхней турбулентной формулы (Fujii-Fujii) возвращает
    achievable=False с пояснением.
    """
    meta = META[key]
    if meta.turb_min is None and meta.nu_turb_tex is None:
        return InverseSearchResult(
            achievable=False, value=None, x_turb_m=None, n_iter=0,
            message='Методика не описывает турбулентный режим.',
        )

    def x_turb_at(val: float) -> Optional[float]:
        if mode == 'joule':
            res = run_methodology_joule(
                key, I=val, R20=R20, alpha_R=alpha_R,
                b_mm=b_mm, L_mm=L_mm,
                t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
                eps_surface=eps_surface, x_min_mm=x_min_mm, N=N,
            )
        else:
            res = run_methodology_qconst(
                key, N_total_W=val,
                b_mm=b_mm, L_mm=L_mm,
                t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
                eps_surface=eps_surface, x_min_mm=x_min_mm, N=N,
            )
        return res.x_turb_start_m

    grid = np.linspace(search_min, search_max, n_scan)
    n_iter = 0
    first_fail = None
    first_ok = None
    first_ok_x = None
    for v in grid:
        n_iter += 1
        try:
            xt = x_turb_at(float(v))
        except Exception:
            xt = None
        if xt is not None and xt <= target_x_m:
            first_ok = float(v)
            first_ok_x = xt
            break
        first_fail = float(v)

    if first_ok is None:
        return InverseSearchResult(
            achievable=False, value=None, x_turb_m=None, n_iter=n_iter,
            message=f'Турбулент не достигается на ≤ {target_x_m*1000:.0f} мм '
                    f'в диапазоне {search_min:g}…{search_max:g}.',
        )
    if first_fail is None:
        return InverseSearchResult(
            achievable=True, value=first_ok, x_turb_m=first_ok_x, n_iter=n_iter,
            message='Турбулент достигается уже на нижней границе диапазона.',
        )

    lo, hi = first_fail, first_ok
    best_v, best_x = first_ok, first_ok_x
    for _ in range(bisect_iter):
        n_iter += 1
        mid = 0.5 * (lo + hi)
        try:
            xt = x_turb_at(mid)
        except Exception:
            xt = None
        if xt is not None and xt <= target_x_m:
            hi, best_v, best_x = mid, mid, xt
        else:
            lo = mid
        if (hi - lo) <= bisect_tol:
            break

    return InverseSearchResult(
        achievable=True, value=best_v, x_turb_m=best_x, n_iter=n_iter,
        message='Найдено бисекцией.',
    )
