"""
Верификация корреляций проекта Plastina через эталонную библиотеку ht (Heat Transfer).

Библиотека ht реализует корреляцию Churchill-Chu для свободной конвекции
у вертикальной пластины. Сравниваем с нашими 8 корреляциями при одинаковых
входных данных (Gr, Pr), чтобы:
  1. Убедиться, что наша Churchill-Chu совпадает с эталоном
  2. Оценить разброс остальных методик относительно эталона
  3. Проверить свойства воздуха CoolProp vs fluids (библиотека из того же пакета)

Зависимости: pip install ht fluids CoolProp
"""

import numpy as np
import pandas as pd

from ht.conv_free_immersed import Nu_free_vertical_plate
from properties import get_air_properties
from correlations import (
    calc_beta, calc_Gr, calc_Ra, calc_eps_t,
    calc_Nu_kerimov, calc_Nu_kuznetov, calc_Nu_churchill_chu,
    calc_Nu_leontiev, calc_Nu_churchill_ozoe, calc_Nu_vliet,
    calc_Nu_fujii, calc_Nu_isachenko,
)
from formatting import _fc, _fe


# ── 1. Верификация Churchill-Chu vs ht ────────────────────────────────────

def verify_churchill_chu_vs_ht(
    dT_values=None,
    x_values=None,
    T_inf_C=20.0,
    P_Pa=101325.0,
    g=9.80665,
):
    """
    Точечное сравнение нашей calc_Nu_churchill_chu с ht.

    Входные данные — (dT, x). Для каждой пары считаем Nu обоими способами.

    Returns:
        pd.DataFrame с колонками: x, dT, T_film, Ra, Nu_ht, Nu_наш, delta_%
    """
    if dT_values is None:
        dT_values = [10, 30, 50, 80, 100, 150, 200]
    if x_values is None:
        x_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]

    rows = []
    for x in x_values:
        for dT in dT_values:
            T_wall = T_inf_C + dT
            T_film = (T_wall + T_inf_C) / 2.0

            air = get_air_properties(T_film, P_Pa)
            beta = calc_beta(T_film)
            Gr = calc_Gr(g, beta, dT, x, air.nu)
            Ra = calc_Ra(Gr, air.Pr)

            # Эталон: библиотека ht
            Nu_ht = Nu_free_vertical_plate(Pr=air.Pr, Gr=Gr, buoyancy=True)

            # Наша реализация
            Nu_ours = calc_Nu_churchill_chu(Ra, air.Pr)

            delta = abs(Nu_ours - Nu_ht) / Nu_ht * 100 if Nu_ht > 0 else 0

            rows.append({
                'x, м': x,
                'dT, °C': dT,
                'T_плён, °C': T_film,
                'Ra': Ra,
                'Nu (ht)': Nu_ht,
                'Nu (наш)': Nu_ours,
                'delta, %': delta,
            })

    return pd.DataFrame(rows)


# ── 2. Все 8 методик vs эталон ht ────────────────────────────────────────

def verify_all_methods_vs_ht(
    x_values=None,
    T_inf_C=20.0,
    dT=100.0,
    P_Pa=101325.0,
    g=9.80665,
):
    """
    Сравнение всех 8 корреляций с эталоном ht при фиксированном dT.

    Returns:
        pd.DataFrame с колонками: x, Ra, method, Nu, delta_vs_ht_%
    """
    if x_values is None:
        x_values = np.concatenate([
            np.linspace(0.01, 0.1, 5),
            np.linspace(0.15, 0.5, 8),
            np.linspace(0.6, 1.5, 10),
        ])

    methods = {
        'Черчилль-Чу (наш)': lambda Ra, Pr, GrPr, et: (calc_Nu_churchill_chu(Ra, Pr), 'full'),
        'Керимов': lambda Ra, Pr, GrPr, et: calc_Nu_kerimov(GrPr, et),
        'Кузнецов': lambda Ra, Pr, GrPr, et: calc_Nu_kuznetov(Ra, Pr),
        'Леонтьев': lambda Ra, Pr, GrPr, et: calc_Nu_leontiev(Ra, Pr),
        'Churchill-Ozoe': lambda Ra, Pr, GrPr, et: calc_Nu_churchill_ozoe(Ra, Pr),
        'Vliet': lambda Ra, Pr, GrPr, et: calc_Nu_vliet(Ra, Pr),
        'Fujii': lambda Ra, Pr, GrPr, et: calc_Nu_fujii(Ra, Pr),
        'Исаченко': lambda Ra, Pr, GrPr, et: calc_Nu_isachenko(GrPr, et),
    }

    rows = []
    for x in x_values:
        T_wall = T_inf_C + dT
        T_film = (T_wall + T_inf_C) / 2.0

        air_film = get_air_properties(T_film, P_Pa)
        beta = calc_beta(T_film)
        Gr = calc_Gr(g, beta, dT, x, air_film.nu)
        Ra = calc_Ra(Gr, air_film.Pr)
        GrPr = Gr * air_film.Pr

        # Поправка eps_t (для Керимова и Исаченко)
        air_inf = get_air_properties(T_inf_C, P_Pa)
        air_wall = get_air_properties(T_wall, P_Pa)
        eps_t = calc_eps_t(air_inf.Pr, air_wall.Pr)

        # Эталон
        Nu_ht = Nu_free_vertical_plate(Pr=air_film.Pr, Gr=Gr, buoyancy=True)

        for method_name, calc_fn in methods.items():
            Nu_val, regime = calc_fn(Ra, air_film.Pr, GrPr, eps_t)
            delta = (Nu_val - Nu_ht) / Nu_ht * 100 if Nu_ht > 0 else 0

            rows.append({
                'x, м': round(x, 4),
                'Ra': Ra,
                'Методика': method_name,
                'Nu': Nu_val,
                'Nu (ht)': Nu_ht,
                'delta, %': round(delta, 1),
                'Режим': regime,
            })

    return pd.DataFrame(rows)


# ── 3. Верификация свойств воздуха: CoolProp vs fluids ────────────────────

def verify_air_properties_vs_fluids(
    T_values=None,
    P_Pa=101325.0,
):
    """
    Сравнение свойств воздуха из CoolProp (наш properties.py)
    с библиотекой fluids.

    Returns:
        pd.DataFrame
    """
    try:
        from fluids.atmosphere import ATMOSPHERE_1976
    except ImportError:
        return pd.DataFrame({'Ошибка': ['fluids не установлен']})

    if T_values is None:
        T_values = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400]

    rows = []
    for T_C in T_values:
        air = get_air_properties(T_C, P_Pa)

        rows.append({
            'T, °C': T_C,
            'nu (CoolProp), м²/с': air.nu,
            'lam (CoolProp), Вт/(м·К)': air.lam,
            'Pr (CoolProp)': air.Pr,
            'rho (CoolProp), кг/м³': air.rho,
        })

    return pd.DataFrame(rows)


# ── 4. Сводный отчёт ─────────────────────────────────────────────────────

def run_full_verification(T_inf_C=20.0, P_Pa=101325.0):
    """
    Полная верификация: свойства + Churchill-Chu + все методики.

    Returns:
        dict с ключами: 'churchill_chu', 'all_methods', 'air_properties'
    """
    print('1. Верификация Churchill-Chu vs ht...')
    df_cc = verify_churchill_chu_vs_ht(T_inf_C=T_inf_C, P_Pa=P_Pa)
    max_delta_cc = df_cc['delta, %'].max()
    print(f'   Макс. расхождение: {max_delta_cc:.4f}%')
    print()

    print('2. Все методики vs ht (dT=100°C)...')
    df_all = verify_all_methods_vs_ht(T_inf_C=T_inf_C, dT=100.0, P_Pa=P_Pa)

    # Сводка по методикам
    summary = df_all.groupby('Методика')['delta, %'].agg(['mean', 'min', 'max'])
    summary.columns = ['Среднее δ, %', 'Мин δ, %', 'Макс δ, %']
    print(summary.to_string())
    print()

    print('3. Свойства воздуха CoolProp...')
    df_air = verify_air_properties_vs_fluids(P_Pa=P_Pa)
    print(f'   {len(df_air)} точек проверено')

    return {
        'churchill_chu': df_cc,
        'all_methods': df_all,
        'summary': summary.reset_index(),
        'air_properties': df_air,
    }


if __name__ == '__main__':
    results = run_full_verification()
