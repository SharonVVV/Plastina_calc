"""
Безразмерные числа подобия для свободной конвекции.
"""


def calc_Gr(g: float, beta: float, dT: float, x: float, nu: float) -> float:
    """Стандартное число Грасгофа. Gr_x = g * beta * dT * x^3 / nu^2."""
    return g * beta * dT * x ** 3 / nu ** 2


def calc_Gr_star(g: float, beta: float, q_w: float, x: float,
                 lam: float, nu: float) -> float:
    """Модифицированное число Грасгофа. Gr*_x = g * beta * q_w * x^4 / (lam * nu^2)."""
    return g * beta * q_w * x ** 4 / (lam * nu ** 2)


def calc_Ra(Gr: float, Pr: float) -> float:
    """Число Рэлея. Ra_x = Gr_x * Pr."""
    return Gr * Pr


def calc_Ra_star(Gr_star: float, Pr: float) -> float:
    """Модифицированное число Рэлея. Ra*_x = Gr*_x * Pr."""
    return Gr_star * Pr
