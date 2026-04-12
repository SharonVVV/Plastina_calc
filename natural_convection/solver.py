"""
Решатель и функции сравнения методик.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .methods import ALL_METHODS


@dataclass
class ConvectionResult:
    """Результат расчёта в одной точке x."""
    x: float
    T_wall_C: float
    T_inf_C: float
    q_w: float
    Nu_x: float
    alpha_x: float
    Ra_x: Optional[float]
    Ra_x_star: Optional[float]
    Gr_x_star: Optional[float]
    Pr: float
    beta: float
    regime: str
    method: str
    n_iter: int
    converged: bool


def solve_convection(method, q_w, T_inf_C, L, N=100, x_min=0.001,
                     P_Pa=101325.0, g=9.80665, **kwargs):
    """
    Расчёт локального коэффициента теплоотдачи вдоль пластины.

    Args:
        method: название методики ('leontiev', 'churchill_ozoe', 'vliet',
                'fujii', 'isachenko')
        q_w: тепловой поток, Вт/м²
        T_inf_C: температура среды, °C
        L: высота пластины, м
        N: количество точек
        x_min: минимальная координата, м
        P_Pa: давление, Па
        g: ускорение свободного падения, м/с²

    Returns:
        list[ConvectionResult]
    """
    if method not in ALL_METHODS:
        raise ValueError(
            f"Неизвестная методика: '{method}'. "
            f"Доступные: {list(ALL_METHODS.keys())}"
        )

    calc_fn = ALL_METHODS[method]
    x_array = np.linspace(x_min, L, N)
    results = []

    for x in x_array:
        raw = calc_fn(x, q_w, T_inf_C, P_Pa=P_Pa, g=g, **kwargs)

        results.append(ConvectionResult(
            x=x,
            T_wall_C=raw['T_wall_C'],
            T_inf_C=T_inf_C,
            q_w=q_w,
            Nu_x=raw['Nu'],
            alpha_x=raw['alpha'],
            Ra_x=raw.get('Ra'),
            Ra_x_star=raw.get('Ra_star'),
            Gr_x_star=raw.get('Gr_star'),
            Pr=raw['Pr'],
            beta=raw['beta'],
            regime=raw['regime'],
            method=method,
            n_iter=raw['n_iter'],
            converged=raw['converged'],
        ))

    return results


def compare_methods(q_w, T_inf_C, L, N=100, x_min=0.001,
                    methods=None, P_Pa=101325.0, g=9.80665, **kwargs):
    """
    Сравнение методик в одной таблице.

    Args:
        methods: список названий методик (None = все 5)

    Returns:
        pd.DataFrame со столбцами:
            x, method, Nu_x, alpha_x, T_wall_C, Ra_x, Ra_x_star, regime
    """
    if methods is None:
        methods = list(ALL_METHODS.keys())

    rows = []
    for method_name in methods:
        results = solve_convection(
            method_name, q_w, T_inf_C, L, N, x_min,
            P_Pa=P_Pa, g=g, **kwargs,
        )
        for r in results:
            rows.append({
                'x': r.x,
                'method': r.method,
                'Nu_x': r.Nu_x,
                'alpha_x': r.alpha_x,
                'T_wall_C': r.T_wall_C,
                'Ra_x': r.Ra_x,
                'Ra_x_star': r.Ra_x_star,
                'Pr': r.Pr,
                'regime': r.regime,
                'n_iter': r.n_iter,
                'converged': r.converged,
            })

    return pd.DataFrame(rows)
