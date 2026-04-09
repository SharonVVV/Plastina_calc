"""
Общие функции форматирования чисел для отображения в UI и LaTeX.
Десятичная запятая, научный формат, защита от NaN/inf.
"""

import math


def _fc(value: float, precision: int = 4) -> str:
    """Форматирование числа с десятичной запятой (для текста/таблиц)."""
    if not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('∞' if value > 0 else '−∞')
    return f'{value:.{precision}f}'.replace('.', ',')


def _fe(value: float, digits: int = 3) -> str:
    """Научный формат: 3,14·10⁻³."""
    if value == 0:
        return '0'
    if not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('∞' if value > 0 else '−∞')
    s = f'{value:.{digits}e}'
    mantissa, exp_part = s.split('e')
    mantissa = mantissa.replace('.', ',')
    exp = int(exp_part)
    if exp == 0:
        return mantissa
    sup = str(exp).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{mantissa}·10{sup}'


def _fcl(value: float, precision: int = 4) -> str:
    r"""Форматирование числа с десятичной запятой для LaTeX: 3{,}14."""
    if not math.isfinite(value):
        if math.isnan(value):
            return r'\mathrm{NaN}'
        return r'\infty' if value > 0 else r'-\infty'
    return f'{value:.{precision}f}'.replace('.', '{,}')


def _fl(value: float, digits: int = 3) -> str:
    r"""Форматирование в научном виде для LaTeX: 3{,}14 \cdot 10^{-3}."""
    if value == 0:
        return '0'
    if not math.isfinite(value):
        if math.isnan(value):
            return r'\mathrm{NaN}'
        return r'\infty' if value > 0 else r'-\infty'
    s = f'{value:.{digits}e}'
    mantissa, exp_part = s.split('e')
    mantissa = mantissa.replace('.', '{,}')
    exp = int(exp_part)
    if exp == 0:
        return mantissa
    return rf'{mantissa} \cdot 10^{{{exp}}}'
