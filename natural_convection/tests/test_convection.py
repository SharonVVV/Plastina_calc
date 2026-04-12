"""
Тесты модуля natural_convection.

Запуск: cd /Users/SharonovVV/Plastina && python -m pytest natural_convection/tests/ -v
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from natural_convection.air_properties import get_props, calc_beta, calc_alpha_thermal
from natural_convection.criteria import calc_Gr, calc_Gr_star, calc_Ra, calc_Ra_star
from natural_convection.methods.vliet import calc_vliet
from natural_convection.methods.fujii import calc_fujii
from natural_convection.methods.leontiev import calc_leontiev, _nu_brdlik, _nu_exact
from natural_convection.methods.churchill_ozoe import calc_churchill_ozoe
from natural_convection.methods.isachenko import calc_isachenko
from natural_convection.solver import solve_convection, compare_methods


# ─── Свойства воздуха ────────────────────────────────────────────────────────

class TestAirProperties:
    def test_props_at_20C(self):
        p = get_props(20.0)
        assert 1.1 < p.rho < 1.3
        assert 0.69 < p.Pr < 0.72
        assert 14e-6 < p.nu < 16e-6
        assert 0.025 < p.lam < 0.027

    def test_beta_ideal_gas(self):
        beta = calc_beta(20.0)
        assert abs(beta - 1.0 / 293.15) < 1e-6

    def test_alpha_thermal(self):
        p = get_props(20.0)
        a = calc_alpha_thermal(p)
        assert 1.5e-5 < a < 2.5e-5


# ─── Безразмерные числа ──────────────────────────────────────────────────────

class TestCriteria:
    def test_Gr_positive(self):
        Gr = calc_Gr(9.81, 1.0 / 300, 50, 0.5, 15e-6)
        assert Gr > 0

    def test_Gr_star_formula(self):
        g, beta, q_w, x, lam, nu = 9.81, 1.0 / 300, 500, 0.3, 0.026, 15e-6
        Gr_star = calc_Gr_star(g, beta, q_w, x, lam, nu)
        expected = g * beta * q_w * x ** 4 / (lam * nu ** 2)
        assert abs(Gr_star - expected) / expected < 1e-10

    def test_Ra_star_relation(self):
        Gr_star = 1e10
        Pr = 0.71
        assert abs(calc_Ra_star(Gr_star, Pr) - Gr_star * Pr) < 1


# ─── Тест Vliet ──────────────────────────────────────────────────────────────

class TestVliet:
    def test_laminar_regime(self):
        """q_w=500, x=0.3, T_inf=27°C (300K). Ламинарный режим."""
        r = calc_vliet(0.3, 500, 27.0)
        assert r['regime'] == 'laminar'
        assert r['Ra_star'] < 1e13
        assert r['Nu'] > 0
        assert r['converged']

    def test_no_iteration(self):
        r = calc_vliet(0.3, 500, 27.0)
        assert r['n_iter'] == 0

    def test_Nu_positive_at_small_x(self):
        r = calc_vliet(0.001, 500, 20.0)
        assert r['Nu'] > 0
        assert r['alpha'] > 0

    def test_T_wall_above_T_inf(self):
        r = calc_vliet(0.5, 1000, 20.0)
        assert r['T_wall_C'] > 20.0


# ─── Тест Fujii ──────────────────────────────────────────────────────────────

class TestFujii:
    def test_laminar_only(self):
        r = calc_fujii(0.3, 500, 27.0)
        assert r['regime'] == 'laminar'
        assert r['Nu'] > 0

    def test_converged(self):
        r = calc_fujii(0.5, 1000, 20.0)
        assert r['converged']
        assert r['T_wall_C'] > 20.0


# ─── Тест Леонтьев ───────────────────────────────────────────────────────────

class TestLeontiev:
    def test_laminar_non_iterative(self):
        # x=0.05 гарантирует Ra < 2e7 (порог Леонтьева)
        r = calc_leontiev(0.05, 500, 27.0)
        assert r['regime'] == 'laminar'
        assert r['n_iter'] == 0
        assert r['Nu'] > 0

    def test_brdlik_vs_exact(self):
        """Брдлик и точное решение должны давать близкие Nu."""
        props = get_props(30.0)
        Gr_star = 1e8
        Nu_brdlik = _nu_brdlik(Gr_star, props.Pr)
        Nu_exact = _nu_exact(Gr_star, props.Pr)
        diff = abs(Nu_brdlik - Nu_exact) / Nu_exact
        assert diff < 0.20  # < 20% расхождение

    def test_exact_solution_flag(self):
        r1 = calc_leontiev(0.3, 500, 27.0, use_exact=False)
        r2 = calc_leontiev(0.3, 500, 27.0, use_exact=True)
        assert r1['Nu'] > 0
        assert r2['Nu'] > 0


# ─── Тест Churchill & Ozoe ───────────────────────────────────────────────────

class TestChurchillOzoe:
    def test_converges(self):
        r = calc_churchill_ozoe(0.3, 500, 27.0)
        assert r['converged']
        assert r['n_iter'] <= 20
        assert r['Nu'] > 0

    def test_table_value_Pr1(self):
        """
        Для Pr=1.0, UHF: Nu/Ra^(1/4) должно быть ≈ 0.456.
        Проверяем формулу напрямую.
        """
        Pr = 1.0
        Ra = 1e6  # произвольное Ra в ламинарном диапазоне
        Nu = 0.563 * Ra ** 0.25 / (1.0 + (0.437 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)
        ratio = Nu / Ra ** 0.25
        assert abs(ratio - 0.456) < 0.01

    def test_iterations_count(self):
        r = calc_churchill_ozoe(0.3, 500, 27.0)
        assert r['n_iter'] <= 20


# ─── Тест Исаченко ───────────────────────────────────────────────────────────

class TestIsachenko:
    def test_converges(self):
        r = calc_isachenko(0.3, 500, 27.0)
        assert r['converged']
        assert r['n_iter'] <= 20

    def test_uses_fluid_temperature(self):
        """Исаченко использует T_inf для свойств, не T_film."""
        r = calc_isachenko(0.3, 500, 27.0)
        # Pr должен быть ≈ Pr при 27°C
        props_inf = get_props(27.0)
        assert abs(r['Pr'] - props_inf.Pr) < 0.01

    def test_laminar_regime(self):
        r = calc_isachenko(0.1, 200, 20.0)
        assert r['regime'] in ('laminar', 'transitional')


# ─── Согласованность методик ─────────────────────────────────────────────────

class TestConsistency:
    def test_laminar_agreement(self):
        """В ламинарном режиме разница Nu между Vliet, Fujii, Леонтьев < 15%."""
        # x=0.05 — все методики в ламинарном режиме
        x, q_w, T_inf = 0.05, 300, 20.0

        r_vliet = calc_vliet(x, q_w, T_inf)
        r_fujii = calc_fujii(x, q_w, T_inf)
        r_leon = calc_leontiev(x, q_w, T_inf)

        assert r_vliet['regime'] == 'laminar'
        assert r_fujii['regime'] == 'laminar'
        assert r_leon['regime'] == 'laminar'

        nus = [r_vliet['Nu'], r_fujii['Nu'], r_leon['Nu']]
        nu_mean = sum(nus) / len(nus)

        for nu in nus:
            assert abs(nu - nu_mean) / nu_mean < 0.15, \
                f"Nu={nus}, отклонение > 15% от среднего {nu_mean:.2f}"

    def test_asymptotic_small_x(self):
        """При x → 0 все формулы дают Nu → 0."""
        x = 1e-4
        for name, calc_fn in [('vliet', calc_vliet), ('fujii', calc_fujii),
                               ('leontiev', calc_leontiev)]:
            r = calc_fn(x, 500, 20.0)
            assert r['Nu'] < 5.0, f"{name}: Nu={r['Nu']} при x={x}"


# ─── Решатель ────────────────────────────────────────────────────────────────

class TestSolver:
    def test_solve_convection_length(self):
        results = solve_convection('vliet', 500, 20.0, L=1.0, N=50)
        assert len(results) == 50

    def test_solve_all_methods(self):
        for method in ['vliet', 'fujii', 'leontiev', 'churchill_ozoe', 'isachenko']:
            results = solve_convection(method, 500, 20.0, L=0.5, N=10)
            assert len(results) == 10
            assert all(r.Nu_x > 0 for r in results)

    def test_compare_methods_df(self):
        df = compare_methods(500, 20.0, L=0.5, N=10)
        assert isinstance(df, type(df))
        assert 'method' in df.columns
        assert 'Nu_x' in df.columns
        assert len(df['method'].unique()) == 5

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Неизвестная методика"):
            solve_convection('nonexistent', 500, 20.0, L=1.0)
