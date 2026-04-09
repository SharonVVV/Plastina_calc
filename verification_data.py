"""
Справочные данные ГСССД для свойств воздуха при атмосферном давлении.
Используются для верификации CoolProp.

Источник: ГСССД 8-79 / Михеев М.А., Михеева И.М. Основы теплопередачи.
"""

import pandas as pd
from properties import get_air_properties
from formatting import _fe as _fmt_sci

# Табличные данные ГСССД: свойства сухого воздуха при P = 101325 Па
# T_C    — температура, °C
# rho    — плотность, кг/м³
# cp     — теплоёмкость, Дж/(кг·К)
# lam    — теплопроводность, Вт/(м·К)
# mu_e6  — динамическая вязкость × 10⁶, Па·с
# nu_e6  — кинематическая вязкость × 10⁶, м²/с
# Pr     — число Прандтля

GSSSD_DATA = {
    'T_C':   [0,      20,     40,     60,     80,     100,    150,    200,    300,    400],
    'rho':   [1.293,  1.205,  1.127,  1.067,  1.000,  0.946,  0.834,  0.746,  0.615,  0.524],
    'cp':    [1005,   1005,   1005,   1005,   1009,   1009,   1017,   1026,   1047,   1068],
    'lam':   [0.0244, 0.0257, 0.0271, 0.0285, 0.0299, 0.0314, 0.0349, 0.0386, 0.0454, 0.0515],
    'mu_e6': [17.2,   18.1,   19.1,   20.1,   21.1,   21.8,   23.7,   25.5,   29.3,   33.0],
    'nu_e6': [13.3,   15.1,   16.9,   18.9,   21.1,   23.1,   28.5,   34.2,   47.7,   63.0],
    'Pr':    [0.707,  0.713,  0.711,  0.709,  0.708,  0.703,  0.690,  0.677,  0.674,  0.678],
}


def get_gsssd_table() -> pd.DataFrame:
    """Вернуть справочные данные ГСССД как DataFrame."""
    df = pd.DataFrame(GSSSD_DATA)
    df.columns = ['T, °C', 'ρ, кг/м³', 'cₚ, Дж/(кг·К)', 'λ, Вт/(м·К)',
                   'μ×10⁶, Па·с', 'ν×10⁶, м²/с', 'Pr']
    return df


def compare_coolprop_vs_gsssd(P_Pa: float = 101325.0) -> pd.DataFrame:
    """
    Сравнить CoolProp с ГСССД при заданном давлении.

    Returns:
        DataFrame с колонками: T, свойство, ГСССД, CoolProp, δ (%)
    """
    rows = []
    for i, T_C in enumerate(GSSSD_DATA['T_C']):
        air = get_air_properties(T_C, P_Pa)

        pairs = [
            ('ρ, кг/м³',      GSSSD_DATA['rho'][i],    air.rho),
            ('λ, Вт/(м·К)',   GSSSD_DATA['lam'][i],    air.lam),
            ('μ×10⁶, Па·с',   GSSSD_DATA['mu_e6'][i],  air.mu * 1e6),
            ('ν×10⁶, м²/с',   GSSSD_DATA['nu_e6'][i],  air.nu * 1e6),
            ('Pr',            GSSSD_DATA['Pr'][i],      air.Pr),
            ('cₚ, Дж/(кг·К)', GSSSD_DATA['cp'][i],     air.cp),
        ]

        for name, ref_val, cp_val in pairs:
            delta_pct = abs(cp_val - ref_val) / ref_val * 100 if ref_val != 0 else 0
            rows.append({
                'T, °C': T_C,
                'Свойство': name,
                'ГСССД': _fmt_sci(ref_val, 4),
                'CoolProp': _fmt_sci(cp_val, 4),
                'δ, %': f'{delta_pct:.2f}'.replace('.', ','),
            })

    return pd.DataFrame(rows)
