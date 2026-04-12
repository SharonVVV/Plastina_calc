"""
Критериальные уравнения и расчёт тепловых потоков.
Свободная конвекция у вертикальной пластины.

Три варианта корреляций:
  1. Методичка Керимова — Nu_ж,x = c·(GrPr)^n·ε_t, свойства при t_ж
  2. Задачник Кузнецова — Nu_x = c·[Ra·Φ(Pr)]^n, свойства при t_плёнки, q_c=const
  3. Черчилль–Чу (обобщённая) — средний Nu, весь диапазон Ra
"""


def calc_beta(t_C: float) -> float:
    """Коэффициент объёмного расширения (модель идеального газа), К⁻¹."""
    return 1.0 / (t_C + 273.15)


def calc_Gr(g: float, beta: float, delta_T: float, x: float, nu: float) -> float:
    """Число Грасгофа.  Gr_x = g·β·ΔT·x³/ν²"""
    return g * beta * delta_T * x ** 3 / nu ** 2


def calc_Ra(Gr: float, Pr: float) -> float:
    """Число Рэлея.  Ra_x = Gr_x·Pr"""
    return Gr * Pr


def calc_eps_t(Pr_inf: float, Pr_c: float) -> float:
    """Поправка на переменность свойств (методичка).  ε_t = (Pr_ж/Pr_c)^0.25"""
    return (Pr_inf / Pr_c) ** 0.25


def calc_Phi(Pr: float) -> float:
    """Функция Φ(Pr) для q_c = const (формула 3.6 Кузнецова).

    Φ(Pr) = [1 + (0.437/Pr)^(9/16)]^(-16/9)
    Для воздуха (Pr ≈ 0.7): Φ ≈ 0.363
    """
    return (1.0 + (0.437 / Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)


def calc_Psi(Pr: float) -> float:
    """Функция Ψ(Pr) для t_c = const (формула 3.3 Кузнецова).

    Ψ(Pr) = [1 + (0.492/Pr)^(9/16)]^(-16/9)
    Для воздуха (Pr ≈ 0.7): Ψ ≈ 0.344
    """
    return (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)


# ── Методичка Керимова ──────────────────────────────────────────────────────

def calc_Nu_kerimov(GrPr: float, eps_t: float,
                    crit_1: float = 1e9, crit_2: float = 6e10) -> tuple:
    """Местный Nu — кусочная корреляция по методичке Керимова.

    Свойства при t_ж.  Двухзонная модель + промежуток.

    Ламинарный (GrPr ≤ crit_1):  Nu = 0.60·(GrPr)^0.25·ε_t
    Переходный (crit_1 < GrPr < crit_2):  интерполяция по lg
    Турбулентный (GrPr ≥ crit_2):  Nu = 0.15·(GrPr)^(1/3)·ε_t
    """
    if GrPr <= crit_1:
        return 0.60 * GrPr ** 0.25 * eps_t, 'lam'
    elif GrPr >= crit_2:
        return 0.15 * GrPr ** (1.0 / 3.0) * eps_t, 'turb'
    else:
        # Интерполяция по lg между значениями на границах
        import math
        Nu_lam = 0.60 * crit_1 ** 0.25 * eps_t
        Nu_turb = 0.15 * crit_2 ** (1.0 / 3.0) * eps_t
        t = (math.log10(GrPr) - math.log10(crit_1)) / (math.log10(crit_2) - math.log10(crit_1))
        Nu = Nu_lam + t * (Nu_turb - Nu_lam)
        return Nu, 'trans'


# ── Задачник Кузнецова (q_c = const) ────────────────────────────────────────

def calc_Nu_kuznetov(Ra: float, Pr: float,
                     crit_1: float = 1e9, crit_2: float = 1e12) -> tuple:
    """Местный Nu — формулы (3.4)/(3.8) Кузнецова, q_c = const.

    Свойства при t_плёнки.

    Ламинарный (Ra ≤ crit_1):  Nu_x = 0.563·[Ra·Φ(Pr)]^0.25
    Переходный (crit_1 < Ra < crit_2):  интерполяция по lg
    Турбулентный (Ra ≥ crit_2):  Nu_x = 0.15·[Ra·Φ(Pr)]^(1/3)
    """
    Phi = calc_Phi(Pr)
    if Ra <= crit_1:
        return 0.563 * (Ra * Phi) ** 0.25, 'lam'
    elif Ra >= crit_2:
        return 0.15 * (Ra * Phi) ** (1.0 / 3.0), 'turb'
    else:
        import math
        Nu_lam = 0.563 * (crit_1 * Phi) ** 0.25
        Nu_turb = 0.15 * (crit_2 * Phi) ** (1.0 / 3.0)
        t = (math.log10(Ra) - math.log10(crit_1)) / (math.log10(crit_2) - math.log10(crit_1))
        Nu = Nu_lam + t * (Nu_turb - Nu_lam)
        return Nu, 'trans'


# ── Черчилль–Чу (обобщённая, средний Nu) ────────────────────────────────────

def calc_Nu_churchill_chu(Ra: float, Pr: float) -> float:
    """Средний Nu — корреляция Черчилля–Чу (3.9), t_c = const, весь диапазон Ra.

    Nu_ср = [0.825 + 0.387·Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)]²
    """
    psi = calc_Psi(Pr)
    return (0.825 + 0.387 * (Ra * psi) ** (1.0 / 6.0)) ** 2


# ── Леонтьев (Брдлик + Эккерт—Джексон) ────────────────────────────────────

def calc_Nu_leontiev(Ra: float, Pr: float,
                     crit_1: float = 2e7, crit_2: float = 2e7) -> tuple:
    """Местный Nu — методика Леонтьева (q_w = const).

    Ламинарный (Брдлик, через пересчёт Gr* → Ra):
        Nu = [0.616 · (Pr/(Pr+0.8))^0.2]^1.25 · Ra^0.25

    Турбулентный (Эккерт—Джексон):
        Nu = 0.0295 · Ra^0.4 · Pr^(1/15) · (1 + 0.494·Pr^(2/3))^(-0.4)

    Граница: Ra = 2·10⁷.
    """
    if Ra <= crit_1:
        C_lam = (0.616 * (Pr / (Pr + 0.8)) ** 0.2) ** 1.25
        return C_lam * Ra ** 0.25, 'lam'
    else:
        Nu = 0.0295 * Ra ** 0.4 * Pr ** (1.0/15.0) * (1.0 + 0.494 * Pr ** (2.0/3.0)) ** (-0.4)
        return Nu, 'turb'


# ── Churchill & Ozoe (1973), UHF ──────────────────────────────────────────

def calc_Nu_churchill_ozoe(Ra: float, Pr: float) -> tuple:
    """Местный Nu — Churchill & Ozoe (1973) для q_w = const.

    Nu = 0.563 · Ra^(1/4) / [1 + (0.437/Pr)^(9/16)]^(4/9)

    Только ламинарный режим (Ra < 10⁹).
    """
    if Ra <= 0:
        return 0.01, 'lam'
    Nu = 0.563 * Ra ** 0.25 / (1.0 + (0.437 / Pr) ** (9.0/16.0)) ** (4.0/9.0)
    regime = 'lam' if Ra < 1e9 else 'turb'
    return Nu, regime


# ── Vliet (1969) / Vliet & Ross (1975) ────────────────────────────────────

def calc_Nu_vliet(Ra: float, Pr: float,
                  crit_1: float = 3.4e9, crit_2: float = 3.4e9) -> tuple:
    """Местный Nu — Vliet (q_w = const, воздух).

    Пересчёт из Ra* = Ra · Nu в эквивалентные формулы через Ra:
      Ламинарный:  Nu = 0.60^1.25 · Ra^0.25
      Турбулентный: Nu = 0.568^(1/0.78) · Ra^(0.22/0.78)

    Граница (Ra*_crit=10¹³ ↔ Ra_crit ≈ 3.4·10⁹ для воздуха).
    """
    if Ra <= crit_1:
        Nu = 0.60 ** 1.25 * Ra ** 0.25
        return Nu, 'lam'
    else:
        Nu = 0.568 ** (1.0/0.78) * Ra ** (0.22/0.78)
        return Nu, 'turb'


# ── Fujii & Fujii (1976) ──────────────────────────────────────────────────

def calc_Nu_fujii(Ra: float, Pr: float) -> tuple:
    """Местный Nu — Fujii & Fujii (1976) для q_w = const.

    Пересчёт через Gr* = Gr · Nu:
        Nu = (Ra · Pr / (4 + 9·√Pr + 10·Pr))^0.25

    Только ламинарный.
    """
    import math
    denom = 4.0 + 9.0 * math.sqrt(Pr) + 10.0 * Pr
    Nu = (Ra * Pr / denom) ** 0.25
    return max(Nu, 0.01), 'lam'


# ── Исаченко, Осипова, Сукомел (1981) ────────────────────────────────────

def calc_Nu_isachenko(GrPr: float, eps_t: float,
                      crit_1: float = 1e9, crit_2: float = 6e10) -> tuple:
    """Местный Nu — Исаченко и др. (1981), q_w = const.

    Свойства при t_ж (НЕ плёночная).
    Ламинарный: Nu = 0.60·(GrPr)^0.25·ε_t
    Турбулентный: Nu = 0.15·(GrPr)^(1/3)·ε_t
    Переходный: турбулентная формула.
    """
    if GrPr <= crit_1:
        return 0.60 * GrPr ** 0.25 * eps_t, 'lam'
    elif GrPr >= crit_2:
        return 0.15 * GrPr ** (1.0/3.0) * eps_t, 'turb'
    else:
        return 0.15 * GrPr ** (1.0/3.0) * eps_t, 'trans'


# ── Обратная совместимость ──────────────────────────────────────────────────

def calc_Nu(GrPr: float, eps_t: float,
            Ra: float = 0.0, Pr: float = 0.7,
            crit_1: float = 1e9, crit_2: float = 6e10) -> tuple:
    """Обёртка: по умолчанию вызывает методичку Керимова."""
    return calc_Nu_kerimov(GrPr, eps_t, crit_1, crit_2)


# ── Тепловые потоки ─────────────────────────────────────────────────────────

def calc_alpha(Nu: float, lam: float, x: float) -> float:
    """Коэффициент теплоотдачи.  α_x = Nu_x·λ/x"""
    return Nu * lam / x


T_REF = 20.0


def calc_q_el(I: float, R20: float, b: float, t_c: float, alpha_R: float) -> float:
    """Электрический тепловой поток.  q_эл = (I²·R₂₀/b)·[1 + (t_c−20)·α_R]"""
    return (I ** 2 * R20 / b) * (1.0 + (t_c - T_REF) * alpha_R)


def calc_q_conv(alpha: float, t_c: float, t_fluid: float) -> float:
    """Конвективный тепловой поток.  q_конв = α·(t_c − t_ж)"""
    return alpha * (t_c - t_fluid)


def calc_q_rad(C_pr: float, T_c_K: float, T_fluid_K: float) -> float:
    """Радиационный тепловой поток.  q_рад = C_пр·[(T_c/100)⁴ − (T_ж/100)⁴]"""
    return C_pr * ((T_c_K / 100.0) ** 4 - (T_fluid_K / 100.0) ** 4)
