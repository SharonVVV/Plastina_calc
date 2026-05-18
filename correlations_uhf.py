"""
Локальные корреляции Nu_x для свободной конвекции у вертикальной плиты
при постоянном тепловом потоке (UHF, q_w = const).

Каждая формула записана В НОТАЦИИ ПЕРВОИСТОЧНИКА. Источники сверены
по PDF/сканам страниц — см. /Users/SharonovVV/Downloads/convection_methods.md.

Свойства воздуха берутся при опорной температуре, заданной самой методикой
через поле `UHFCorrelation.t_ref`:
  • 'film'  — T_f = (T_s + T_∞)/2  — consensus монографий (ЦКВ, Vliet, Holman,
    Bejan, Jiji, Леонтьев);
  • 'fluid' — T_∞ — Керимов 1992 (МЭИ): «свойства при t_ж».

Обозначения и связи:
    Ra_x   = Gr_x · Pr              — через ΔT = T_s − T_∞
    Gr_x*  = g·β·q_w·x⁴ / (λ·ν²)    — модифицированное число Грасгофа (через q_w)
    Ra_*x  = Gr_x* · Pr             — модифицированное число Рэлея (тождественно)

В UHF-задаче параметр через q_w (Gr_x* и Ra_*x) известен сразу — он не
зависит от T_s; параметр через ΔT (Ra_x и Gr_x) зависит от T_s и
получает физический смысл только после сходимости теплового баланса.
Поэтому каждая методика классифицирует режим по своему характерному
параметру — точно так же, как это делается в её первоисточнике.

Каждая Nu-функция возвращает NuResult с явным флагом регима:
    REG_LAM / REG_TRANS / REG_TURB     — внутри декларированного коридора методики
    REG_OUT_OF_RANGE                    — параметр вне опубликованного диапазона
Это позволяет визуализатору пометить «вне применимости» снизу и сверху.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


# Константы режимов.
REG_LAM = 'lam'
REG_TURB = 'turb'
REG_TRANS = 'trans'
REG_OUT_OF_RANGE = 'out'   # параметр вне диапазона применимости


@dataclass
class NuResult:
    Nu: float
    regime: str       # REG_LAM / REG_TURB / REG_TRANS / REG_OUT_OF_RANGE
    param_name: str   # имя параметра, по которому проведена классификация
    param_value: float


# ── Вспомогательные ────────────────────────────────────────────────────────

def calc_Phi(Pr: float) -> float:
    """Φ(Pr) для UHF (ЦКВ 2008, формула 3.6).

        Φ(Pr) = [1 + (0.437/Pr)^(9/16)]^(−16/9)
        Для воздуха Pr≈0.7: Φ ≈ 0.363.
    """
    return (1.0 + (0.437 / Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)


def calc_Gr(g: float, beta: float, delta_T: float, x: float, nu: float) -> float:
    """Gr_x = g·β·ΔT·x³/ν²."""
    return g * beta * delta_T * x ** 3 / nu ** 2


def calc_Gr_star(g: float, beta: float, q_w: float, x: float,
                 k: float, nu: float) -> float:
    """Gr_x* = g·β·q_w·x⁴ / (k·ν²).  k — теплопроводность."""
    return g * beta * q_w * x ** 4 / (k * nu ** 2)


# ── 1. ЦКВ 2008 — Цветков-Керимов-Величко, UHF локально ─────────────────────
# Источник: «Задачник по тепломассообмену», 2-е изд., 2008, стр. 34-35.
# (3.4) ламинар (Ra 10⁴…10⁹), (3.8) турбулент (Ra > 10¹²).

_CKV_LAM_MIN = 1.0e4
_CKV_LAM_MAX = 1.0e9
_CKV_TURB_MIN = 1.0e12

def nu_ckv(Ra: float, Pr: float) -> NuResult:
    """Цветков-Керимов-Величко 2008, UHF локально.
    (3.4) Nu_x = 0.563·[Ra·Φ(Pr)]^(1/4),  10⁴ ≤ Ra ≤ 10⁹
    (3.8) Nu_x = 0.15·[Ra·Φ(Pr)]^(1/3),   Ra ≥ 10¹²
    Транзитная зона 10⁹…10¹² в источнике как локальная корреляция
    не описана; здесь — log-интерполяция между значениями на границах.
    Вне опубликованного диапазона возвращается REG_OUT_OF_RANGE.
    """
    Phi = calc_Phi(Pr)
    if Ra < _CKV_LAM_MIN:
        return NuResult(0.563 * (Ra * Phi) ** 0.25, REG_OUT_OF_RANGE, 'Ra', Ra)
    if Ra <= _CKV_LAM_MAX:
        return NuResult(0.563 * (Ra * Phi) ** 0.25, REG_LAM, 'Ra', Ra)
    if Ra >= _CKV_TURB_MIN:
        return NuResult(0.15 * (Ra * Phi) ** (1.0 / 3.0), REG_TURB, 'Ra', Ra)
    Nu_lo = 0.563 * (_CKV_LAM_MAX * Phi) ** 0.25
    Nu_hi = 0.15 * (_CKV_TURB_MIN * Phi) ** (1.0 / 3.0)
    t = (math.log10(Ra) - math.log10(_CKV_LAM_MAX)) / \
        (math.log10(_CKV_TURB_MIN) - math.log10(_CKV_LAM_MAX))
    return NuResult(Nu_lo + t * (Nu_hi - Nu_lo), REG_TRANS, 'Ra', Ra)


# ── 2. Vliet 1969 — ASME J. Heat Transfer ──────────────────────────────────
# (1) ламинар (Gr*Pr 10⁸…1.3·10¹³), (2) турбулент (10¹⁴…10¹⁶).

_VLIET_LAM_MIN = 1.0e8
_VLIET_LAM_MAX = 1.3e13
_VLIET_TURB_MIN = 1.0e14
_VLIET_TURB_MAX = 1.0e16

def nu_vliet(GrPr_star: float, Pr: float) -> NuResult:
    """Vliet 1969, UHF локально.
    (1) Nu_x = 0.60·(Gr_x*·Pr)^(1/5),  10⁸ ≤ Gr*Pr ≤ 1.3·10¹³
    (2) Nu_x = 0.30·(Gr_x*·Pr)^(0.24), 10¹⁴ ≤ Gr*Pr ≤ 10¹⁶
    Между 1.3·10¹³ и 10¹⁴ — переход (log-интерполяция, в источнике не описан).
    """
    if GrPr_star < _VLIET_LAM_MIN:
        return NuResult(0.60 * GrPr_star ** 0.2, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)
    if GrPr_star <= _VLIET_LAM_MAX:
        return NuResult(0.60 * GrPr_star ** 0.2, REG_LAM, 'Gr*Pr', GrPr_star)
    if GrPr_star > _VLIET_TURB_MAX:
        return NuResult(0.30 * GrPr_star ** 0.24, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)
    if GrPr_star >= _VLIET_TURB_MIN:
        return NuResult(0.30 * GrPr_star ** 0.24, REG_TURB, 'Gr*Pr', GrPr_star)
    Nu_lo = 0.60 * _VLIET_LAM_MAX ** 0.2
    Nu_hi = 0.30 * _VLIET_TURB_MIN ** 0.24
    t = (math.log10(GrPr_star) - math.log10(_VLIET_LAM_MAX)) / \
        (math.log10(_VLIET_TURB_MIN) - math.log10(_VLIET_LAM_MAX))
    return NuResult(Nu_lo + t * (Nu_hi - Nu_lo), REG_TRANS, 'Gr*Pr', GrPr_star)


# ── 3. Леонтьев 2018 — Брдлик + Эккерт-Джексон, UHF ────────────────────────
# Брдлик: «Теория тепломассообмена» под ред. Леонтьева, 3-е изд., 2018, стр. 307.
# Эккерт-Джексон: стр. 310.  Граница перехода (стр. 310): Ra > 0.7·10⁹.
# Верхняя граница локальной формулы Эккерта-Джексона в источнике не указана;
# здесь не контролируется (Ra > 0.7·10⁹ → REG_TURB всегда).

_LEONTIEV_RA_TRANS = 0.7e9

def nu_leontiev(Ra: float, GrPr_star: float, Pr: float) -> NuResult:
    """Леонтьев 2018, UHF локально.
    Ламинар (Брдлик): Nu_x = 0.616·[Pr/(Pr+0.8)]^(1/5)·(Gr_x*·Pr)^(1/5)
    Турбулент (Эккерт-Джексон):
        Nu_x = 0.0295·Ra_x^(2/5)·Pr^(1/15)·(1 + 0.494·Pr^(2/3))^(−2/5)
    Граница (стр. 310): Ra > 0.7·10⁹ → переход в турбулент.
    """
    if Ra <= _LEONTIEV_RA_TRANS:
        Nu = 0.616 * (Pr / (Pr + 0.8)) ** 0.2 * GrPr_star ** 0.2
        return NuResult(Nu, REG_LAM, 'Ra', Ra)
    Nu = (0.0295 * Ra ** 0.4 * Pr ** (1.0 / 15.0)
          * (1.0 + 0.494 * Pr ** (2.0 / 3.0)) ** (-0.4))
    return NuResult(Nu, REG_TURB, 'Ra', Ra)


# ── 4. Holman 2010, UHF локально (7-31)+(7-32) ─────────────────────────────

_HOLMAN_LAM_MIN = 1.0e5
_HOLMAN_LAM_MAX = 1.0e11
_HOLMAN_TURB_MIN = 2.0e13
_HOLMAN_TURB_MAX = 1.0e16

def nu_holman(GrPr_star: float, Pr: float) -> NuResult:
    """Holman 2010, UHF локально, стр. 336.
    (7-31) Nu_x = 0.60·(Gr_x*·Pr)^(1/5),  10⁵ ≤ Gr*Pr ≤ 10¹¹
    (7-32) Nu_x = 0.17·(Gr_x*·Pr)^(1/4),  2·10¹³ ≤ Gr*Pr ≤ 10¹⁶
    """
    if GrPr_star < _HOLMAN_LAM_MIN:
        return NuResult(0.60 * GrPr_star ** 0.2, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)
    if GrPr_star <= _HOLMAN_LAM_MAX:
        return NuResult(0.60 * GrPr_star ** 0.2, REG_LAM, 'Gr*Pr', GrPr_star)
    if GrPr_star > _HOLMAN_TURB_MAX:
        return NuResult(0.17 * GrPr_star ** 0.25, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)
    if GrPr_star >= _HOLMAN_TURB_MIN:
        return NuResult(0.17 * GrPr_star ** 0.25, REG_TURB, 'Gr*Pr', GrPr_star)
    Nu_lo = 0.60 * _HOLMAN_LAM_MAX ** 0.2
    Nu_hi = 0.17 * _HOLMAN_TURB_MIN ** 0.25
    t = (math.log10(GrPr_star) - math.log10(_HOLMAN_LAM_MAX)) / \
        (math.log10(_HOLMAN_TURB_MIN) - math.log10(_HOLMAN_LAM_MAX))
    return NuResult(Nu_lo + t * (Nu_hi - Nu_lo), REG_TRANS, 'Gr*Pr', GrPr_star)


# ── 5. Bejan 2013 — Vliet & Liu (4.108)+(4.109) ────────────────────────────

_BEJAN_VL_LAM_MIN = 1.0e5
_BEJAN_VL_STITCH = 1.0e13
_BEJAN_VL_TURB_MAX = 1.0e16

def nu_bejan_vliet_liu(Ra_star: float, Pr: float) -> NuResult:
    """Bejan 2013, стр. 204, через Vliet & Liu.
    (4.108) Nu = 0.6·Ra_*^(1/5),     10⁵ ≤ Ra_* ≤ 10¹³
    (4.109) Nu = 0.568·Ra_*^(0.22),  10¹³ ≤ Ra_* ≤ 10¹⁶
    """
    if Ra_star < _BEJAN_VL_LAM_MIN:
        return NuResult(0.6 * Ra_star ** 0.2, REG_OUT_OF_RANGE, 'Ra*', Ra_star)
    if Ra_star <= _BEJAN_VL_STITCH:
        return NuResult(0.6 * Ra_star ** 0.2, REG_LAM, 'Ra*', Ra_star)
    if Ra_star > _BEJAN_VL_TURB_MAX:
        return NuResult(0.568 * Ra_star ** 0.22, REG_OUT_OF_RANGE, 'Ra*', Ra_star)
    return NuResult(0.568 * Ra_star ** 0.22, REG_TURB, 'Ra*', Ra_star)


# ── 6. Bejan 2013 — для воздуха (4.110)+(4.111) ────────────────────────────
# Граница в источнике не дана — заимствуем стык 10¹³ из раздела 4 у Bejan
# (тот же диапазон, что и для Vliet-Liu).

_BEJAN_AIR_LAM_MIN = 1.0e5
_BEJAN_AIR_STITCH = 1.0e13
_BEJAN_AIR_TURB_MAX = 1.0e16

def nu_bejan_air(Ra_star: float, Pr: float) -> NuResult:
    """Bejan 2013, стр. 204, специальные формулы для воздуха.
    (4.110) Nu = 0.55·Ra_*^(1/5)   — ламинар
    (4.111) Nu = 0.17·Ra_*^(1/4)   — турбулент
    """
    if Ra_star < _BEJAN_AIR_LAM_MIN:
        return NuResult(0.55 * Ra_star ** 0.2, REG_OUT_OF_RANGE, 'Ra*', Ra_star)
    if Ra_star <= _BEJAN_AIR_STITCH:
        return NuResult(0.55 * Ra_star ** 0.2, REG_LAM, 'Ra*', Ra_star)
    if Ra_star > _BEJAN_AIR_TURB_MAX:
        return NuResult(0.17 * Ra_star ** 0.25, REG_OUT_OF_RANGE, 'Ra*', Ra_star)
    return NuResult(0.17 * Ra_star ** 0.25, REG_TURB, 'Ra*', Ra_star)


# ── 7. Керимов Р.В., МЭИ 1992 — лабораторная методичка ─────────────────────
# Источник: Керимов Р.В. Лабораторная работа № 9 и 9а по курсу «Тепломассо-
# обмен». Местная теплоотдача при свободном движении воздуха около вертикаль-
# ной пластины. — М.: Изд-во МЭИ, 1992. — 10 с.  Формула (2).
# Свойства при t_ж (НЕ при плёночной); ε_t = (Pr_ж/Pr_c)^0.25 — опытная
# поправка на переменность свойств жидкости.

_KER92_LAM_MIN = 1.0e3
_KER92_LAM_MAX = 1.0e9
_KER92_TURB_MIN = 6.0e10

def nu_kerimov_1992(Ra: float, Pr: float, eps_t: float) -> NuResult:
    """Керимов Р.В., МЭИ 1992, формула (2).
        Nu_ж,x = c · (Gr·Pr)^n · ε_t
    Ламинарный  (Gr·Pr)_ж,x = 10³ — 10⁹:  c = 0.60, n = 0.25
    Турбулентный (Gr·Pr)_ж,x > 6·10¹⁰:    c = 0.15, n = 1/3
    Транзитная зона 10⁹ … 6·10¹⁰ в источнике как отдельная корреляция не
    описана — здесь log-интерполяция между значениями на границах.
    """
    if Ra < _KER92_LAM_MIN:
        return NuResult(0.60 * Ra ** 0.25 * eps_t, REG_OUT_OF_RANGE, 'Gr·Pr', Ra)
    if Ra <= _KER92_LAM_MAX:
        return NuResult(0.60 * Ra ** 0.25 * eps_t, REG_LAM, 'Gr·Pr', Ra)
    if Ra >= _KER92_TURB_MIN:
        return NuResult(0.15 * Ra ** (1.0 / 3.0) * eps_t, REG_TURB, 'Gr·Pr', Ra)
    Nu_lo = 0.60 * _KER92_LAM_MAX ** 0.25 * eps_t
    Nu_hi = 0.15 * _KER92_TURB_MIN ** (1.0 / 3.0) * eps_t
    t = (math.log10(Ra) - math.log10(_KER92_LAM_MAX)) / \
        (math.log10(_KER92_TURB_MIN) - math.log10(_KER92_LAM_MAX))
    return NuResult(Nu_lo + t * (Nu_hi - Nu_lo), REG_TRANS, 'Gr·Pr', Ra)


# ── 8. Fujii & Fujii 1976 — Jiji (7.32)+(7.33), UHF, только ламинар ───────
# Pr-диапазон формулы (7.33): 0.001 < Pr < 1000. Верхняя граница режима —
# перевод критерия Jiji (Example 7.2, стр. 277: «Ra_x < 10⁹») в форму Gr_x*·Pr
# для UHF: при Ra ≈ 10⁹ и Nu ~ 10⁴ имеем Gr_x*·Pr ≈ Ra·Nu ≈ 10¹³.

_FUJII_GRPR_MAX = 1.0e13
_FUJII_PR_MIN = 1.0e-3
_FUJII_PR_MAX = 1.0e3

def nu_fujii_fujii(Gr_star: float, Pr: float) -> NuResult:
    """Fujii & Fujii 1976 — через θ(0) (Jiji 2009, стр. 275-277).

    Развёрнутая форма (Jiji указывает её в комментарии к Example 7.2):
        Nu_x = [Pr/(4 + 9·Pr^(1/2) + 10·Pr)]^(1/5) · (Pr·Gr_x*)^(1/5)

    Применима для 0.001 < Pr < 1000.  Только ламинарный режим.
    """
    coef = (Pr / (4.0 + 9.0 * Pr ** 0.5 + 10.0 * Pr)) ** 0.2
    Nu = coef * (Pr * Gr_star) ** 0.2
    GrPr_star = Gr_star * Pr
    if Pr < _FUJII_PR_MIN or Pr > _FUJII_PR_MAX:
        return NuResult(Nu, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)
    if GrPr_star <= _FUJII_GRPR_MAX:
        return NuResult(Nu, REG_LAM, 'Gr*Pr', GrPr_star)
    return NuResult(Nu, REG_OUT_OF_RANGE, 'Gr*Pr', GrPr_star)


# ── Реестр методик ─────────────────────────────────────────────────────────

@dataclass
class UHFCorrelation:
    key: str
    needs: tuple[str, ...]   # имена kwargs из {'Ra','GrPr_star','Ra_star','Gr_star','Pr','eps_t'}
    fn: Callable
    # Опорная температура физических свойств воздуха:
    #   'film'  — T_f = (T_s + T_∞)/2  (стандарт у Bejan, Holman, Jiji, Incropera…)
    #   'fluid' — T_∞                  (Керимов 1992: «свойства при t_ж»)
    t_ref: str = 'film'


REGISTRY: dict[str, UHFCorrelation] = {
    'kerimov_1992':    UHFCorrelation('kerimov_1992',    ('Ra', 'Pr', 'eps_t'),      nu_kerimov_1992, t_ref='fluid'),
    'ckv':             UHFCorrelation('ckv',             ('Ra', 'Pr'),               nu_ckv),
    'vliet':           UHFCorrelation('vliet',           ('GrPr_star', 'Pr'),        nu_vliet),
    'leontiev':        UHFCorrelation('leontiev',        ('Ra', 'GrPr_star', 'Pr'),  nu_leontiev),
    'holman':          UHFCorrelation('holman',          ('GrPr_star', 'Pr'),        nu_holman),
    'bejan_vliet_liu': UHFCorrelation('bejan_vliet_liu', ('Ra_star', 'Pr'),          nu_bejan_vliet_liu),
    'bejan_air':       UHFCorrelation('bejan_air',       ('Ra_star', 'Pr'),          nu_bejan_air),
    'fujii_fujii':     UHFCorrelation('fujii_fujii',     ('Gr_star', 'Pr'),          nu_fujii_fujii),
}


def compute_nu(key: str, *, Ra: float, Gr_star: float, Pr: float,
               eps_t: float = 1.0) -> NuResult:
    """Универсальный вызов: знает что подставлять каждой корреляции.

    eps_t — опытная поправка (Pr_ж/Pr_c)^0.25; нужна Керимову 1992.
    Для остальных методик игнорируется.
    """
    corr = REGISTRY[key]
    GrPr_star = Gr_star * Pr
    Ra_star = GrPr_star
    args = {
        'Ra': Ra,
        'GrPr_star': GrPr_star,
        'Ra_star': Ra_star,
        'Gr_star': Gr_star,
        'Pr': Pr,
        'eps_t': eps_t,
    }
    needed = {n: args[n] for n in corr.needs}
    return corr.fn(**needed)


def t_ref_for(key: str) -> str:
    """Опорная температура для свойств воздуха в выбранной методике."""
    return REGISTRY[key].t_ref
