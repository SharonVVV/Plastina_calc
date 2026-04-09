"""
Обёртка над CoolProp для получения теплофизических свойств воздуха.
"""

from dataclasses import dataclass
from CoolProp.CoolProp import PropsSI


@dataclass
class AirProperties:
    """Теплофизические свойства воздуха при заданной температуре и давлении."""
    nu: float      # кинематическая вязкость, м²/с
    lam: float     # теплопроводность, Вт/(м·К)
    Pr: float      # число Прандтля
    mu: float      # динамическая вязкость, Па·с
    rho: float     # плотность, кг/м³
    cp: float      # теплоёмкость, Дж/(кг·К)


def get_air_properties(t_film_C: float, P_Pa: float = 101325.0) -> AirProperties:
    """
    Возвращает теплофизические свойства воздуха при заданной температуре.

    Args:
        t_film_C: определяющая (плёночная) температура, °C
        P_Pa: давление, Па

    Returns:
        AirProperties с полями: nu, lam, Pr, mu, rho, cp
    """
    T_K = t_film_C + 273.15

    mu = PropsSI('V', 'T', T_K, 'P', P_Pa, 'Air')       # Па·с
    rho = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Air')       # кг/м³
    lam = PropsSI('L', 'T', T_K, 'P', P_Pa, 'Air')       # Вт/(м·К)
    Pr = PropsSI('Prandtl', 'T', T_K, 'P', P_Pa, 'Air')  # —
    cp = PropsSI('C', 'T', T_K, 'P', P_Pa, 'Air')        # Дж/(кг·К)
    nu = mu / rho                                          # м²/с

    return AirProperties(nu=nu, lam=lam, Pr=Pr, mu=mu, rho=rho, cp=cp)
