"""
Модуль расчёта свободной конвекции у вертикальной пластины.

Граничное условие: постоянный тепловой поток q_w = const (UHF).
Среда: воздух (Pr ~ 0.7).

Реализованы 5 независимых методик:
  1. Леонтьев (Брдлик + Эккерт—Джексон)
  2. Churchill & Ozoe (1973)
  3. Vliet (1969) / Vliet & Ross (1975)
  4. Fujii & Fujii (1976)
  5. Исаченко, Осипова, Сукомел (1981)
"""

from .solver import ConvectionResult, solve_convection, compare_methods

__all__ = ['ConvectionResult', 'solve_convection', 'compare_methods']
