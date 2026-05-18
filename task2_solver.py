"""
Расчётчик Задачи №2 — реальная пластина с толщиной δ, баланс тепла по всем
торцам и сторонам.

Геометрия:
  b × L × δ     (ширина × высота × толщина), типично 100 × 2000 × 12 мм
  • тыльная грань (b × L)         — подвод q_w от силиконовых матов
  • лицевая грань (b × L)         — конвекция (UHF-методика) + радиация
  • 2 боковых торца (δ × L)       — конвекция (та же UHF) + радиация
  • верхний торец (b × δ, z=L)    — конвекция (Леонтьев 1979, ↑) + радиация
  • нижний торец «днище» (b × δ)  — ТОЛЬКО радиация

Физическая модель:
  • Bi = h·δ/λ_метал ≪ 1 → плита изотермична поперёк толщины ⇒ задача 1D.
  • Учитывается продольная теплопроводность вдоль z (важна у краёв).
  • Уравнение фина:
        λ·A_cr·d²T/dz² + q_w·b
          − [α_v(T,z)·ΔT + ε·σ·(T⁴−T_∞⁴)]·(b + 2δ) = 0
    с ГУ:
        z = 0:  −λ·A_cr·dT/dz = ε·σ·(T⁴−T_∞⁴)·b·δ
        z = L:  +λ·A_cr·dT/dz = (α_top·ΔT + ε·σ·(T⁴−T_∞⁴))·b·δ
  • A_cr = b·δ — площадь поперечного сечения металла.

Численный метод: дискретизация конечными разностями + Picard-итерация:
зафиксировать α(T_k, z), линеаризовать радиацию через h_rad(T_k),
решить тридиаг. систему, обновить T, повторить.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from properties import get_air_properties
from correlations_uhf import compute_nu, calc_Gr, calc_Gr_star, t_ref_for
from correlations_horizontal import alpha_horizontal_up
from correlations_meta import META, CorrelationMeta


SIGMA = 5.670374419e-8
G = 9.80665


# ── Расщепление T_mean → T_back, T_front через толщину δ (Задача 3) ────────

def compute_back_front_temps(result) -> tuple:
    """Из распределения T_mean(z) — полученного 1D-решателем для тонкой
    плиты (Bi ≪ 1) — восстанавливает температуры на тыльной и лицевой
    гранях с учётом небольшой 1D-теплопроводности поперёк толщины δ.

    Модель: предполагаем линейный профиль температуры поперёк δ.
    Поток через толщину q_through равен удельному тепловому потоку,
    уходящему с лицевой грани в воздух (конвекция + излучение):
        q_through(z) = α(z,T_mean)·(T_mean - T_∞) + ε·σ·(T_mean⁴ - T_∞⁴)
    Тогда перепад температур поперёк толщины:
        ΔT_through(z) = q_through(z) · δ / λ_м
    и расщепление вокруг T_mean (она же — средняя по толщине):
        T_back(z)  = T_mean(z) + ΔT_through(z) / 2
        T_front(z) = T_mean(z) − ΔT_through(z) / 2

    Возвращает кортеж массивов (T_back, T_front, dT_through),
    в °C / °C / К соответственно. Длина — как у result.nodes.
    """
    import numpy as np
    delta_m = result.delta_mm / 1000.0
    lambda_m = result.lambda_metal
    T_back = []
    T_front = []
    dT_through = []
    for n in result.nodes:
        # Удельный поток в воздух с вертикальной (= лицевой) грани:
        q_face = n.q_conv_per_m2 + n.q_rad_per_m2
        # Перепад поперёк толщины (линейная модель):
        dT = q_face * delta_m / lambda_m if lambda_m > 0 else 0.0
        dT_through.append(dT)
        T_back.append(n.T_s_C + dT / 2.0)
        T_front.append(n.T_s_C - dT / 2.0)
    return (np.array(T_back), np.array(T_front), np.array(dT_through))


# ── Материалы плиты ─────────────────────────────────────────────────────────

# ── Изоляционные материалы (для Задачи 3) ─────────────────────────────────
# Типовые значения теплопроводности при ~50…100 °C; в реальности зависят
# от температуры и плотности изоляции.

INSULATION_MATERIALS: dict[str, dict] = {
    'Аэрогель (Pyrogel XT)':       {'lambda': 0.015, 'note': 'минимальная λ среди гибких изоляций, ~15 мВт/(м·К)'},
    'Пенополиуретан (PUR)':         {'lambda': 0.025, 'note': 'жёсткие плиты, ~25 мВт/(м·К)'},
    'Пенополистирол (EPS)':         {'lambda': 0.035, 'note': 'белый пенопласт, ~35 мВт/(м·К)'},
    'Каменная вата (базальт)':      {'lambda': 0.040, 'note': 'плотные плиты, ~40 мВт/(м·К)'},
    'Минеральная вата':             {'lambda': 0.045, 'note': 'универсальная, ~45 мВт/(м·К)'},
    'Каолиновая вата':              {'lambda': 0.060, 'note': 'для высоких температур до 1200 °C'},
    'Силикат кальция':              {'lambda': 0.060, 'note': 'жёсткие плиты для индустриальных приложений'},
    'Перлит вспученный':            {'lambda': 0.060, 'note': 'сыпучая засыпка, ~60 мВт/(м·К)'},
    'Шамот лёгкий':                 {'lambda': 0.200, 'note': 'огнеупорный, относительно высокая λ'},
}


MATERIALS: dict[str, dict] = {
    'Д16-Т (алюминий)': {
        'lambda': 130.0, 'rho': 2780.0, 'cp': 920.0,
        'note': 'Дюралюминий Д16-Т (≈AA 2024-T4), λ при 20 °C',
    },
    'АМг3 (алюминий)': {
        'lambda': 160.0, 'rho': 2670.0, 'cp': 920.0,
        'note': 'Деформируемый алюминий АМг3, более теплопроводный',
    },
    'AISI 304 (нерж. сталь)': {
        'lambda': 16.0, 'rho': 7900.0, 'cp': 500.0,
        'note': 'Аустенитная нержавеющая сталь — низкая λ',
    },
    'Медь М1': {
        'lambda': 400.0, 'rho': 8960.0, 'cp': 385.0,
        'note': 'Чистая медь — практически изотермична',
    },
}


# ── Точечный и интегральный результат ───────────────────────────────────────

@dataclass
class NodeResult:
    z_m: float
    T_s_C: float
    alpha_front: float          # коэф. теплоотдачи лицевой грани, Вт/(м²·К)
    q_conv_per_m2: float        # конвективный тепловой поток на вертикальной грани
    q_rad_per_m2: float         # радиационный тепловой поток
    Ra_x: float
    Gr_x_star: float
    GrPr_star: float
    regime: str                 # из NuResult


@dataclass
class Task2Result:
    # Входные данные (для отчётности).
    key: str                    # корреляция конвекции для вертикальных граней
    meta: CorrelationMeta
    material_name: str
    lambda_metal: float
    b_mm: float
    L_mm: float
    delta_mm: float
    N_total_W: float
    q_w: float
    t_fluid_C: float
    eps_surface: float

    # Профиль.
    nodes: list[NodeResult]

    # Интегральные потери (Вт).
    Q_input_W: float                 # подведённое от матов
    Q_front_conv_W: float
    Q_front_rad_W: float
    Q_sides_conv_W: float            # суммарно для 2 боковых торцов
    Q_sides_rad_W: float
    Q_top_conv_W: float
    Q_top_rad_W: float
    Q_bottom_rad_W: float

    # Итеративная сходимость.
    iterations: int
    max_delta_K: float
    converged: bool

    @property
    def Q_front_W(self) -> float:    return self.Q_front_conv_W + self.Q_front_rad_W
    @property
    def Q_sides_W(self) -> float:    return self.Q_sides_conv_W + self.Q_sides_rad_W
    @property
    def Q_top_W(self) -> float:      return self.Q_top_conv_W + self.Q_top_rad_W
    @property
    def Q_bottom_W(self) -> float:   return self.Q_bottom_rad_W

    @property
    def Q_output_W(self) -> float:
        return self.Q_front_W + self.Q_sides_W + self.Q_top_W + self.Q_bottom_W

    @property
    def residual_W(self) -> float:   return self.Q_input_W - self.Q_output_W

    @property
    def residual_pct(self) -> float:
        return 100.0 * self.residual_W / self.Q_input_W if self.Q_input_W > 0 else 0.0

    @property
    def T_max_C(self) -> float:      return max(n.T_s_C for n in self.nodes)
    @property
    def T_min_C(self) -> float:      return min(n.T_s_C for n in self.nodes)
    @property
    def T_avg_C(self) -> float:      return float(np.mean([n.T_s_C for n in self.nodes]))


# ── Локальный коэф. теплоотдачи (вертикальная грань) ───────────────────────

def _alpha_vertical(key: str, t_s_C: float, t_fluid_C: float, P_Pa: float,
                    q_w: float, x_m: float, g: float) -> tuple[float, dict]:
    """α на вертикальной поверхности при заданной T_s и координате x вдоль высоты.

    Возвращает (α, info) где info содержит Ra, Gr*·Pr, regime, T_film и т.п.
    """
    t_ref = t_ref_for(key)
    if t_ref == 'fluid':
        t_eval = t_fluid_C
    else:
        t_eval = 0.5 * (t_s_C + t_fluid_C)
    air = get_air_properties(t_eval, P_Pa)
    beta = 1.0 / (t_eval + 273.15)
    delta_T = max(t_s_C - t_fluid_C, 0.001)   # защита от деления на ноль

    air_inf = get_air_properties(t_fluid_C, P_Pa)
    air_wall = get_air_properties(t_s_C, P_Pa)
    eps_t = (air_inf.Pr / air_wall.Pr) ** 0.25 if air_wall.Pr > 0 else 1.0

    Gr_star = calc_Gr_star(g, beta, q_w, x_m, air.lam, air.nu)
    Gr = calc_Gr(g, beta, delta_T, x_m, air.nu)
    Ra = Gr * air.Pr

    nu_res = compute_nu(key, Ra=Ra, Gr_star=Gr_star, Pr=air.Pr, eps_t=eps_t)
    alpha = nu_res.Nu * air.lam / x_m if x_m > 0 else 0.0
    return alpha, {
        'Ra': Ra, 'Gr_star': Gr_star, 'GrPr_star': Gr_star * air.Pr,
        'regime': nu_res.regime,
    }


def _h_rad(t_s_C: float, t_fluid_C: float, eps: float) -> float:
    """Линеаризованный коэф. радиации:
        q_rad = ε·σ·(T_s⁴ − T_∞⁴) = h_rad(T_s)·(T_s − T_∞),
        h_rad = ε·σ·(T_s + T_∞)·(T_s² + T_∞²).
    """
    Ts = t_s_C + 273.15
    Ti = t_fluid_C + 273.15
    return eps * SIGMA * (Ts + Ti) * (Ts * Ts + Ti * Ti)


# ── Основной солвер ────────────────────────────────────────────────────────

def _h_eff_with_insulation(
    h_native_total: float,
    h_outer_conv: float,
    iso_thickness_m: float,
    iso_lambda: float,
) -> float:
    """Эффективный коэффициент теплоотдачи на торце при наличии изоляции.

    Тепло идёт последовательно: металл → теплопроводность через слой
    изоляции δ_изо/λ_изо → свободная конвекция с внешней поверхности
    изоляции (коэф. α_внеш ≈ нативной конвекции этой грани).
    Радиация через изоляцию не учитывается (низкая ε лицевой стороны
    изоляции + непрозрачность).

        1/h_eff = δ_изо/λ_изо + 1/α_внеш

    При iso_thickness_m=0 (нет изоляции) возвращает h_native_total,
    т.е. исходный коэффициент (конвекция + излучение голого металла).
    """
    if iso_thickness_m <= 0:
        return h_native_total
    R_iso = iso_thickness_m / iso_lambda if iso_lambda > 0 else 1e9
    R_conv = 1.0 / h_outer_conv if h_outer_conv > 0 else 1e6
    return 1.0 / (R_iso + R_conv)


def solve_task2(
    *,
    key: str,
    N_total_W: float,
    b_mm: float, L_mm: float, delta_mm: float,
    lambda_metal: float,
    material_name: str = '',
    t_fluid_C: float = 20.0,
    P_Pa: float = 101325.0,
    eps_surface: float = 0.95,
    N_nodes: int = 121,
    max_iter: int = 40,
    tol_K: float = 0.01,
    g: float = G,
    relax: float = 0.7,
    x_min_mm: float = 10.0,       # защита от сингулярности корреляций при x→0
                                  # (ниже физический ПС не успевает развиться)
    axial_conduction: bool = True,  # если False — кондуктив λ_м·d²T/dz² зануляется,
                                    # каждое сечение решается как локальный баланс
    # Изоляция торцов (Задача 3). Если толщина = 0 — без изоляции.
    # Внешняя поверхность изоляции имеет ту же ориентацию, что и торец;
    # коэф. конвекции на ней приближённо равен нативной конвекции этой грани.
    sides_iso_thickness_mm: float = 0.0,
    sides_iso_lambda: float = 0.045,
    top_iso_thickness_mm: float = 0.0,
    top_iso_lambda: float = 0.045,
    bottom_iso_thickness_mm: float = 0.0,
    bottom_iso_lambda: float = 0.045,
) -> Task2Result:
    """Решает 1D-фин-уравнение для пластины с толщиной δ.

    Алгоритм:
      1. Дискретизация: z_i = i·dz, i = 0..N-1, dz = L/(N-1).
         Для α на вертикальной грани используется max(z, x_min).
      2. Picard-итерация:
         (a) При текущем T_k(z) вычисляем α_v(z), h_rad(z) — поточечно.
         (b) Строим тридиаг. систему для нового T_{k+1}(z):
             −λ·A_cr·(T_{i-1} − 2T_i + T_{i+1})/dz²
                 + h_total_i·(b+2δ)·(T_i − T_∞)
                 = q_w·b
             где h_total_i = α_v_i + h_rad_i.
             ГУ через ghost-узлы:
               z=0: h_b·b·δ·ΔT = (T_1 − T_{-1})·λ·A_cr/(2dz)·(−1)
                    → T_{-1} = T_1 + 2dz·h_b·b·δ/(λ·A_cr)·ΔT_0
                    (где h_b — только h_rad на днище; конвекции нет)
               z=L: симметрично с h_top = α_top + h_rad
         (c) Решаем; обновляем T_{k+1} = T_k + relax·(T_new − T_k).
      3. Останов когда max|ΔT| < tol_K.
    """
    b_m = b_mm / 1000.0
    L_m = L_mm / 1000.0
    delta_m = delta_mm / 1000.0
    x_min_m = x_min_mm / 1000.0

    A_cr = b_m * delta_m              # поперечное сечение металла
    A_end = b_m * delta_m             # площадь верхнего/нижнего торца
    perim_v = b_m + 2 * delta_m       # «эффективная ширина» вертикального периметра
    perim_top_m = 2.0 * (b_m + delta_m)
    q_w = N_total_W / (b_m * L_m)     # удельный тепловой поток матов

    # Толщины изоляции (в м); 0 — без изоляции на этом торце.
    sides_iso_m = sides_iso_thickness_mm / 1000.0
    top_iso_m   = top_iso_thickness_mm   / 1000.0
    bottom_iso_m = bottom_iso_thickness_mm / 1000.0
    iso_sides  = sides_iso_m > 0
    iso_top    = top_iso_m   > 0
    iso_bottom = bottom_iso_m > 0

    dz = L_m / (N_nodes - 1)
    z = np.linspace(0.0, L_m, N_nodes)
    x_eff = np.maximum(z, x_min_m)    # координата для UHF-корреляции

    # Стартовое приближение: ΔT, выводимая из «грубого» баланса
    #   q_w·b·L ≈ h_eff·(b+2δ)·L·ΔT + h_rad·…  ⇒  ΔT ≈ q_w·b/(h_eff·(b+2δ))
    h_guess = 8.0      # Вт/(м²·К) — порядок величины для воздуха
    dT_guess = q_w * b_m / (h_guess * perim_v)
    T = np.full(N_nodes, t_fluid_C + max(dT_guess, 10.0))

    converged = False
    last_dT = float('inf')
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        # 1. Локальные коэффициенты при T_k(z).
        alpha_v = np.zeros(N_nodes)
        h_rad_v = np.zeros(N_nodes)
        for i in range(N_nodes):
            alpha_v[i], _ = _alpha_vertical(
                key, T[i], t_fluid_C, P_Pa, q_w, x_eff[i], g,
            )
            h_rad_v[i] = _h_rad(T[i], t_fluid_C, eps_surface)
        h_v_native = alpha_v + h_rad_v   # голый металл вертик. грани

        # Front (b) — никогда не изолируется; lateral (2δ) — может быть.
        if iso_sides:
            h_lat = np.array([
                _h_eff_with_insulation(h_v_native[i], alpha_v[i],
                                       sides_iso_m, sides_iso_lambda)
                for i in range(N_nodes)
            ])
        else:
            h_lat = h_v_native.copy()
        # Эффективная «h × ширина» вертикальных потерь на единицу dz:
        h_total_perim_v = h_v_native * b_m + h_lat * 2.0 * delta_m

        # ГУ-коэффициенты для торцов z=0 и z=L.
        # z=L: Леонтьев 1979 (7.30), горячая грань ↑ + радиация.
        alpha_top_native, _ = alpha_horizontal_up(
            T[-1], t_fluid_C, P_Pa, A_end, perim_top_m,
        )
        h_top_native_total = alpha_top_native + _h_rad(T[-1], t_fluid_C, eps_surface)
        if iso_top:
            h_top = _h_eff_with_insulation(h_top_native_total,
                                           alpha_top_native,
                                           top_iso_m, top_iso_lambda)
        else:
            h_top = h_top_native_total

        # z=0: только радиация на голом днище. На изолированном днище
        # конвекция на внешней (нижней, обращённой вниз) поверхности
        # пренебрежимо мала (по постановке) → h_outer_conv = 0 → h_eff ≈ 0
        # (изолированное днище фактически перекрывает теплоотвод полностью).
        h_bot_native_total = _h_rad(T[0], t_fluid_C, eps_surface)
        if iso_bottom:
            h_bot = _h_eff_with_insulation(h_bot_native_total, 0.0,
                                           bottom_iso_m, bottom_iso_lambda)
        else:
            h_bot = h_bot_native_total

        # 2. Построение тридиаг. системы A·T_new = rhs.
        # Узел i (1..N-2) — стандартный fin:
        #   (-λ·A_cr/dz²)·T_{i-1} + (2λ·A_cr/dz² + h_total_i·perim_v)·T_i
        #     + (-λ·A_cr/dz²)·T_{i+1} = q_w·b + h_total_i·perim_v·T_∞
        a = np.zeros(N_nodes)   # subdiag
        b_d = np.zeros(N_nodes)  # main diag
        c = np.zeros(N_nodes)   # superdiag
        rhs = np.zeros(N_nodes)
        kappa = (lambda_metal * A_cr / (dz * dz)) if axial_conduction else 0.0

        for i in range(1, N_nodes - 1):
            a[i] = -kappa
            c[i] = -kappa
            b_d[i] = 2 * kappa + h_total_perim_v[i]
            rhs[i] = q_w * b_m + h_total_perim_v[i] * t_fluid_C

        # ГУ через ghost-узел.
        # Вклад торцов (днище / верх) в баланс крайних узлов через
        # технику «полу-элемента». Локальные потери на единственной
        # площади торца A_end = b·δ дают «эквивалентную плотность»
        # 2·h_edge·A_end/dz (множитель 2 — узел представляет полу-ячейку
        # длиной dz/2).  Эти члены работают вместе с кондуктивом и
        # без него дают «1/dz»-артефакт на крайних узлах — поэтому
        # включаются только когда axial_conduction=True. В режиме
        # «локального баланса» (без кондуктива) Q_верх и Q_дн всё равно
        # учитываются — но через постпроцесс при готовом T(0), T(L).
        if axial_conduction:
            edge_b_coeff = 2.0 * h_bot * A_end / dz
            edge_t_coeff = 2.0 * h_top * A_end / dz
        else:
            edge_b_coeff = 0.0
            edge_t_coeff = 0.0

        # На z=0:  диагональ + edge_b_coeff, побочный элемент −2κ
        # (зеркальный ghost-узел используется только когда κ ≠ 0).
        b_d[0] = 2 * kappa + h_total_perim_v[0] + edge_b_coeff
        c[0] = -2 * kappa
        rhs[0] = (q_w * b_m
                  + (h_total_perim_v[0] + edge_b_coeff) * t_fluid_C)

        # На z=L:  симметрично с edge_t_coeff
        a[-1] = -2 * kappa
        b_d[-1] = 2 * kappa + h_total_perim_v[-1] + edge_t_coeff
        rhs[-1] = (q_w * b_m
                   + (h_total_perim_v[-1] + edge_t_coeff) * t_fluid_C)

        # 3. Решаем тридиаг. систему методом прогонки.
        T_new = _thomas(a, b_d, c, rhs)

        # 4. Релаксация и проверка сходимости.
        T_upd = (1 - relax) * T + relax * T_new
        last_dT = float(np.max(np.abs(T_upd - T)))
        T = T_upd
        if last_dT < tol_K:
            converged = True
            break

    # ── Постпроцесс: точечная и интегральная статистика ─────────────────
    nodes: list[NodeResult] = []
    for i in range(N_nodes):
        alpha_i, info = _alpha_vertical(
            key, T[i], t_fluid_C, P_Pa, q_w, x_eff[i], g,
        )
        delta_T = T[i] - t_fluid_C
        q_conv = alpha_i * delta_T
        Ts_K = T[i] + 273.15
        Ti_K = t_fluid_C + 273.15
        q_rad = eps_surface * SIGMA * (Ts_K ** 4 - Ti_K ** 4)
        nodes.append(NodeResult(
            z_m=z[i], T_s_C=T[i],
            alpha_front=alpha_i,
            q_conv_per_m2=q_conv,
            q_rad_per_m2=q_rad,
            Ra_x=info['Ra'], Gr_x_star=info['Gr_star'],
            GrPr_star=info['GrPr_star'], regime=info['regime'],
        ))

    # Интегралы — трапеция.
    # Лицевая грань всегда без изоляции — потери разделяются на конвекцию и
    # излучение голого металла.
    q_conv_v = np.array([n.q_conv_per_m2 for n in nodes])
    q_rad_v = np.array([n.q_rad_per_m2 for n in nodes])
    Q_front_conv = float(np.trapezoid(q_conv_v, z) * b_m)
    Q_front_rad  = float(np.trapezoid(q_rad_v,  z) * b_m)

    # Боковые торцы (2δ): без изоляции — q_conv + q_rad как у голого металла.
    # С изоляцией — потери через эффективный коэф. h_lat (нет радиации
    # через изоляцию, перенос только теплопроводностью + конвекция на
    # внешней поверхности изоляции).
    if iso_sides:
        # h_lat[i] вычислен в финальной итерации. Используем его.
        delta_T_per_z = np.array([T[i] - t_fluid_C for i in range(N_nodes)])
        q_side_iso = np.array([
            _h_eff_with_insulation(
                alpha_v[i] + h_rad_v[i], alpha_v[i],
                sides_iso_m, sides_iso_lambda
            ) * delta_T_per_z[i]
            for i in range(N_nodes)
        ])
        Q_sides_conv = float(np.trapezoid(q_side_iso, z) * 2 * delta_m)
        Q_sides_rad = 0.0
    else:
        Q_sides_conv = float(np.trapezoid(q_conv_v, z) * 2 * delta_m)
        Q_sides_rad  = float(np.trapezoid(q_rad_v,  z) * 2 * delta_m)

    # Верхний торец.
    alpha_top_final, _ = alpha_horizontal_up(
        T[-1], t_fluid_C, P_Pa, A_end, perim_top_m,
    )
    Ts_top_K = T[-1] + 273.15
    dT_top = T[-1] - t_fluid_C
    if iso_top:
        h_top_native = alpha_top_final + eps_surface * SIGMA * (Ts_top_K + Ti_K) * (Ts_top_K**2 + Ti_K**2)
        h_top_iso_eff = _h_eff_with_insulation(
            h_top_native, alpha_top_final, top_iso_m, top_iso_lambda
        )
        Q_top_conv = float(h_top_iso_eff * dT_top * A_end)
        Q_top_rad = 0.0
    else:
        Q_top_conv = float(alpha_top_final * dT_top * A_end)
        Q_top_rad  = float(eps_surface * SIGMA * (Ts_top_K**4 - Ti_K**4) * A_end)

    # Днище. Голое — только радиация. Изолированное — потерь практически
    # нет (нет ни конвекции на внешней грани, ни радиации через изоляцию).
    Ts_bot_K = T[0] + 273.15
    if iso_bottom:
        Q_bot_rad = 0.0
    else:
        Q_bot_rad = float(eps_surface * SIGMA * (Ts_bot_K**4 - Ti_K**4) * A_end)

    return Task2Result(
        key=key, meta=META[key],
        material_name=material_name, lambda_metal=lambda_metal,
        b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
        N_total_W=N_total_W, q_w=q_w,
        t_fluid_C=t_fluid_C, eps_surface=eps_surface,
        nodes=nodes,
        Q_input_W=q_w * b_m * L_m,
        Q_front_conv_W=Q_front_conv, Q_front_rad_W=Q_front_rad,
        Q_sides_conv_W=Q_sides_conv, Q_sides_rad_W=Q_sides_rad,
        Q_top_conv_W=Q_top_conv, Q_top_rad_W=Q_top_rad,
        Q_bottom_rad_W=Q_bot_rad,
        iterations=n_iter, max_delta_K=last_dT, converged=converged,
    )


# ── Тридиагональный решатель Томаса ────────────────────────────────────────

def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
            d: np.ndarray) -> np.ndarray:
    """Решение системы (a — sub, b — main, c — sup; d — rhs).
    a[0] и c[-1] игнорируются. Стандартный алгоритм прогонки."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
