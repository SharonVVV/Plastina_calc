"""
Обёртка над CoolProp для свойств воздуха.

Переиспользует существующий properties.py из корня проекта,
добавляя удобные функции и расчёт температуропроводности.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from properties import get_air_properties, AirProperties


def get_props(T_C: float, P_Pa: float = 101325.0) -> AirProperties:
    """Свойства воздуха при температуре в °C."""
    return get_air_properties(T_C, P_Pa)


def get_props_K(T_K: float, P_Pa: float = 101325.0) -> AirProperties:
    """Свойства воздуха при температуре в K."""
    return get_air_properties(T_K - 273.15, P_Pa)


def calc_alpha_thermal(props: AirProperties) -> float:
    """Температуропроводность a = lambda / (rho * cp), м²/с."""
    return props.lam / (props.rho * props.cp)


def calc_beta(T_C: float) -> float:
    """Коэффициент объёмного расширения (идеальный газ), 1/К."""
    return 1.0 / (T_C + 273.15)
