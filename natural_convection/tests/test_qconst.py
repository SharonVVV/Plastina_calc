"""
Тесты для решателя q_w = const (силиконовые нагреватели).

Запуск:
    cd /Users/SharonovVV/Plastina && python -m pytest natural_convection/tests/test_qconst.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from solver import solve_plate, solve_plate_qconst, CalculationResult


# ── Общие параметры стенда (близкие к ВКР) ──────────────────────────────────

ENV = dict(t_fluid_C=20.0, g=9.80665, P_Pa=101325.0)
GEOM_NEW = dict(b_mm=100.0, L_mm=2000.0, x_min_mm=5.0, N=40)
GEOM_KERIMOV = dict(b_mm=205.0, L_mm=1540.0, x_min_mm=10.0, N=40)


def _eps_to_C_pr(eps: float) -> float:
    return eps * 5.67


# ── 1. q_el в каждой точке равно q_w ────────────────────────────────────────

def test_qconst_q_el_constant():
    q_w = 3000.0
    r = solve_plate_qconst(
        q_w=q_w, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.95),
        correlation='kuznetov', eps_surface=0.95, N_total_W=q_w * 0.2,
    )
    converged = [p for p in r.points if p.converged]
    assert len(converged) > 0
    for p in converged:
        assert p.q_el == pytest.approx(q_w, abs=1e-9)


# ── 2. Невязка ниже допустимой точности ─────────────────────────────────────

def test_qconst_residual_below_tol():
    r = solve_plate_qconst(
        q_w=2000.0, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.9),
        correlation='kerimov', eps_surface=0.9, N_total_W=400.0,
    )
    for p in r.points:
        if p.converged:
            assert abs(p.residual) < 1e-5


# ── 3. Профиль t_c(x) монотонно возрастает ──────────────────────────────────

def test_qconst_t_c_monotonic():
    # Берём гладкую корреляцию Черчилля–Чу — без скачков на границе режимов.
    r = solve_plate_qconst(
        q_w=1500.0, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.9),
        correlation='churchill_chu', eps_surface=0.9, N_total_W=300.0,
    )
    pts = [p for p in r.points if p.converged]
    assert len(pts) >= 10
    for p_prev, p_next in zip(pts, pts[1:]):
        assert p_next.t_c >= p_prev.t_c - 0.05


# ── 4. Воспроизводимость: фикс. параметры → фикс. t_max ─────────────────────

def test_qconst_reproducibility():
    kwargs = dict(
        q_w=5000.0, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.95),
        correlation='kuznetov', eps_surface=0.95, N_total_W=1000.0,
    )
    r1 = solve_plate_qconst(**kwargs)
    r2 = solve_plate_qconst(**kwargs)
    t1 = max(p.t_c for p in r1.points if p.converged)
    t2 = max(p.t_c for p in r2.points if p.converged)
    assert t1 == pytest.approx(t2, abs=1e-6)


# ── 5. Эквивалентность с Керимовским решателем ──────────────────────────────

def test_qconst_equivalence_with_kerimov():
    # Малое C_pr → пренебрежимая радиация → q_конв ≈ q_эл по всей высоте.
    r_k = solve_plate(
        I=200.0, R20=3.34e-3, alpha_R=1.088e-3,
        **GEOM_KERIMOV, **ENV, C_pr=0.05,
        correlation='kerimov', t_ref_mode='auto',
    )
    ok = [p for p in r_k.points if p.converged]
    assert len(ok) > 0
    q_w_avg = sum(p.q_el for p in ok) / len(ok)

    r_q = solve_plate_qconst(
        q_w=q_w_avg, **GEOM_KERIMOV, **ENV, C_pr=0.05,
        correlation='kerimov', t_ref_mode='auto',
        eps_surface=0.05 / 5.67, N_total_W=q_w_avg * 0.205 * 1.540,
    )

    # Сравниваем в средней зоне (избегаем краевых эффектов).
    L_m = r_k.L_m
    x_lo, x_hi = 0.2 * L_m, 0.8 * L_m
    diffs = []
    for pk, pq in zip(r_k.points, r_q.points):
        if pk.converged and pq.converged and x_lo <= pk.x_m <= x_hi:
            diffs.append(abs(pk.t_c - pq.t_c))
    assert diffs, 'нет точек в средней зоне'
    # При близкой к константе q_эл (ТКС вносит ~5% разброс) разница t_c должна
    # быть невелика — границу 8 K выбираем с запасом.
    assert max(diffs) < 8.0


# ── 6. Малая мощность — небольшой перегрев ──────────────────────────────────

def test_qconst_low_q_w():
    r = solve_plate_qconst(
        q_w=200.0, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.9),
        correlation='kuznetov', eps_surface=0.9, N_total_W=40.0,
    )
    pts = [p for p in r.points if p.converged]
    assert len(pts) == r.N
    t_max = max(p.t_c for p in pts)
    assert t_max < ENV['t_fluid_C'] + 50.0


# ── 7. Высокая мощность — расширенный интервал brentq ───────────────────────

def test_qconst_high_q_w():
    r = solve_plate_qconst(
        q_w=15000.0, **GEOM_NEW, **ENV, C_pr=_eps_to_C_pr(0.95),
        correlation='kuznetov', eps_surface=0.95, N_total_W=3000.0,
    )
    failed = [p for p in r.points if not p.converged]
    assert not failed, f'Не сошлось точек: {len(failed)} (ожидалось 0)'


# ── 8. Обратная совместимость solve_plate ───────────────────────────────────

def test_qconst_backward_compat_kerimov():
    r = solve_plate(
        I=300.0, R20=3.34e-3, alpha_R=1.088e-3,
        **GEOM_KERIMOV, **ENV, C_pr=2.0,
        correlation='kerimov', t_ref_mode='auto',
    )
    assert isinstance(r, CalculationResult)
    assert r.mode == 'kerimov'
    assert r.q_w == 0.0
    assert r.N_total_W == 0.0
    assert r.eps_surface == 0.0
    assert r.A_m2 == 0.0
    # Первая сходящаяся точка должна давать положительный q_эл (ток ≠ 0).
    ok = [p for p in r.points if p.converged]
    assert ok and ok[0].q_el > 0
