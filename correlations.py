"""
Критериальные уравнения и расчёт тепловых потоков.
Свободная конвекция у вертикальной пластины.
Кусочная корреляция по методичке: ламинарный / турбулентный режим + поправка ε_t.
"""


def calc_beta(t_film_C: float) -> float:
    """Коэффициент объёмного расширения (модель идеального газа), К⁻¹.

    beta = 1 / (t_опр + 273.15)
    """
    return 1.0 / (t_film_C + 273.15)


def calc_Gr(g: float, beta: float, delta_T: float, x: float, nu: float) -> float:
    """Число Грасгофа.

    Gr_x = g * beta * (t_c - t_ж) * x³ / nu²
    """
    return g * beta * delta_T * x ** 3 / nu ** 2


def calc_Ra(Gr: float, Pr: float) -> float:
    """Число Рэлея.

    Ra_x = Gr_x * Pr
    """
    return Gr * Pr


def calc_eps_t(Pr_inf: float, Pr_c: float) -> float:
    """Поправка на переменность свойств по методичке.

    ε_t = (Pr_∞ / Pr_c)^0.25
    """
    return (Pr_inf / Pr_c) ** 0.25


def calc_Nu(GrPr: float, eps_t: float) -> tuple:
    """Число Нуссельта — кусочная корреляция по методичке.

    Ламинарный (GrPr <= 10⁹):
        Nu = 0.60 * (GrPr)^0.25 * ε_t

    Турбулентный (GrPr > 10⁹):
        Nu = 0.15 * (GrPr)^(1/3) * ε_t

    Returns:
        (Nu, regime) — число Нуссельта и строка режима ('lam', 'turb')
    """
    if GrPr <= 1e9:
        Nu = 0.60 * GrPr ** 0.25 * eps_t
        regime = 'lam'
    else:
        Nu = 0.15 * GrPr ** (1.0 / 3.0) * eps_t
        regime = 'turb'

    return Nu, regime


def calc_Nu_churchill_chu(Ra: float, Pr: float) -> float:
    """Число Нуссельта — полная корреляция Черчилля–Чу (весь диапазон Ra).

    Nu_x = [0.825 + 0.387 * (Ra * ψ)^(1/6)]²

    ψ = [1 + (0.492/Pr)^(9/16)]^(-16/9)
    """
    psi = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    return (0.825 + 0.387 * (Ra * psi) ** (1.0 / 6.0)) ** 2


def calc_alpha(Nu: float, lam: float, x: float) -> float:
    """Коэффициент теплоотдачи, Вт/(м²·К).

    alpha_x = Nu_x * lambda / x
    """
    return Nu * lam / x


T_REF = 20.0  # Базовая температура для поправки сопротивления, °C


def calc_q_el(I: float, R20: float, b: float, t_c: float, alpha_R: float) -> float:
    """Электрический тепловой поток (закон Джоуля–Ленца с ТКС), Вт/м².

    q_эл = (I² * R₂₀ / b) * [1 + (t_c - 20) * α_R]

    Поправка по сопротивлению считается от базовой температуры 20 °C.
    """
    return (I ** 2 * R20 / b) * (1.0 + (t_c - T_REF) * alpha_R)


def calc_q_conv(alpha: float, t_c: float, t_fluid: float) -> float:
    """Конвективный тепловой поток, Вт/м².

    q_конв = alpha * (t_c - t_ж)
    """
    return alpha * (t_c - t_fluid)


def calc_q_rad(C_pr: float, T_c_K: float, T_fluid_K: float) -> float:
    """Радиационный тепловой поток, Вт/м².

    q_рад = C_пр * [(T_c/100)⁴ - (T_ж/100)⁴]

    Формула в стандарте Михеева: C_пр в Вт/(м²·К⁴), температуры в Кельвинах,
    делённых на 100.
    """
    return C_pr * ((T_c_K / 100.0) ** 4 - (T_fluid_K / 100.0) ** 4)
