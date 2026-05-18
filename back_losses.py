"""
back_losses.py — расчёт тепловых потерь сэндвич-конструкции лабораторного
стенда свободной конвекции:
    • тыльная поверхность (через ПИР + стальной щит → конвекция + излучение);
    • торцы (4 слоя × периметр панели → конвекция + излучение).

Сэндвич (от лицевой стороны вглубь):
    1. АМг3,                  12 мм,   λ = 130 Вт/(м·К)    — теплораспределитель
    2. Силиконовый нагреватель, 1.5 мм, λ = 0.2 Вт/(м·К)    — источник тепла
    3. ПИР,                   30 мм,   λ = 0.022 Вт/(м·К)
    4. Стальной щит,            3 мм,  λ = 50  Вт/(м·К)

В тыльной кондуктивной цепочке учитываются только ПИР и сталь
(сопротивлениями АМг3 и нагревателя в тыл пренебрегаем — они на пути
полезного теплового потока).

Использует CoolProp (через `properties.get_air_properties`) и корреляцию
Черчилля–Чу (через `correlations.calc_Nu_churchill_chu`) — те же, что в
основном проекте Plastina_calc.

Запуск как CLI:
    python3 back_losses.py
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scipy.optimize import brentq

from properties import get_air_properties
from correlations import calc_Nu_churchill_chu


# ── Константы ──────────────────────────────────────────────────────────────

SIGMA0 = 5.67          # Вт/(м²·К⁴), σ·10⁸ — постоянная Стефана–Больцмана
                       # в форме «деление на 100⁴» (как в correlations.calc_q_rad)


# ── Описание конструкции ───────────────────────────────────────────────────


@dataclass
class Layer:
    """Слой сэндвича."""
    name: str
    thickness: float       # м
    lam: float             # Вт/(м·К) — теплопроводность поперёк слоя
    eps_edge: float = 0.9  # степень черноты торцевой поверхности (для радиации)
    color: str = '#cccccc' # цвет для визуализации поперечного сечения


@dataclass
class EdgeProfile:
    """Алюминиевый рамочный профиль с термическим разрывом по периметру.

    Конструкция (от торца пакета наружу):
      торец пакета → прокладка (МБОР-5Ф) → профиль (AД31) → воздух.

    Прокладка обеспечивает термический разрыв (λ ≈ 0.045 при 80–100 °C),
    профиль почти изотермичен (Al, λ ≈ 200) и излучает наружу со своей ε.
    """
    enabled: bool = False

    # Прокладка (МБОР-5Ф фольгированный или аналог)
    delta_break: float = 5e-3        # м, толщина
    lambda_break: float = 0.045      # Вт/(м·К), консервативно при ~90 °C
    eps_break_outer: float = 0.05    # ε фольгированной стороны (информ.)

    # Профиль (AД31, типовой Al-сплав)
    delta_profile: float = 2e-3      # м, толщина стенки
    lambda_profile: float = 200.0    # Вт/(м·К)
    eps_profile: float = 0.25        # анодированный натуральный

    @property
    def R_cond(self) -> float:
        """Кондуктивное сопротивление цепочки прокладка + профиль (м²·К/Вт)."""
        return (self.delta_break / self.lambda_break
                + self.delta_profile / self.lambda_profile)


@dataclass
class SandwichConfig:
    """Полная конструкция и условия среды."""
    layers: List[Layer]                           # ВСЕ слои, лицо → тыл
    back_path_indices: Tuple[int, ...] = (2, 3)   # индексы слоёв тыльного пути
    L_height: float = 2.0                         # м — высота панели
    b_width: float = 0.1                          # м — ширина панели
    eps_back: float = 0.3                         # ε внешней поверхности тыла
    t_inf_C: float = 20.0
    P_Pa: float = 101325.0
    g: float = 9.80665
    include_edges: bool = True
    profile: EdgeProfile = field(default_factory=EdgeProfile)

    @property
    def A_face(self) -> float:
        return self.L_height * self.b_width

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.L_height + self.b_width)

    @property
    def total_thickness(self) -> float:
        return sum(L.thickness for L in self.layers)

    def R_cond_back(self) -> float:
        """Сопротивление кондуктивной цепочки тыльных потерь."""
        return sum(self.layers[i].thickness / self.layers[i].lam
                   for i in self.back_path_indices)


# ── Дефолтная конструкция (по ТЗ) ──────────────────────────────────────────


def default_config() -> SandwichConfig:
    return SandwichConfig(
        layers=[
            Layer('АМг3',                  0.012,  130.0,  0.95, '#a8b2c0'),
            Layer('Силикон. нагреватель',  0.0015, 0.2,    0.85, '#e8a87c'),
            Layer('ПИР',                   0.030,  0.022,  0.90, '#f1c40f'),
            Layer('Стальной щит',          0.003,  50.0,   0.30, '#7f8c8d'),
        ],
        back_path_indices=(2, 3),  # PIR + сталь
    )


DEFAULT_REGIMES: List[Tuple[float, float]] = [
    (600.0,  71.0),
    (1200.0, 106.0),
    (1800.0, 134.0),
    (2400.0, 158.0),
]


# ── Свободная конвекция у вертикальной стенки ──────────────────────────────


def alpha_free_conv_vertical(T_surface_C: float, t_inf_C: float, P_Pa: float,
                             g: float, L_char: float
                             ) -> Tuple[float, float, float]:
    """Средний коэффициент теплоотдачи свободной конвекции у вертикальной
    поверхности. Возвращает (alpha, Ra, Nu).

    Используется единая корреляция Черчилля–Чу по всему диапазону Ra."""
    delta_T = T_surface_C - t_inf_C
    if delta_T <= 0 or L_char <= 0:
        return 0.0, 0.0, 0.0
    t_film = (T_surface_C + t_inf_C) / 2.0
    air = get_air_properties(t_film, P_Pa)
    beta = 1.0 / (t_film + 273.15)
    Gr = g * beta * delta_T * L_char ** 3 / air.nu ** 2
    Ra = Gr * air.Pr
    Nu = calc_Nu_churchill_chu(Ra, air.Pr)
    alpha = Nu * air.lam / L_char
    return alpha, Ra, Nu


def q_outside(T_surface_C: float, eps: float, cfg: SandwichConfig,
              L_char: float) -> Tuple[float, float, float, float]:
    """Удельные потоки на внешнюю среду: (q_conv, q_rad, q_total, alpha)."""
    alpha, _, _ = alpha_free_conv_vertical(
        T_surface_C, cfg.t_inf_C, cfg.P_Pa, cfg.g, L_char)
    q_conv = alpha * (T_surface_C - cfg.t_inf_C)
    T_K = T_surface_C + 273.15
    T_inf_K = cfg.t_inf_C + 273.15
    q_rad = eps * SIGMA0 * ((T_K / 100.0) ** 4 - (T_inf_K / 100.0) ** 4)
    return q_conv, q_rad, q_conv + q_rad, alpha


# ── Решение тыльного баланса (T_steel_outer) ───────────────────────────────


def _back_residual(T_outer_C: float, t_face_C: float,
                   cfg: SandwichConfig) -> float:
    """F(T_outer) = q_cond − [q_conv + q_rad]."""
    q_cond = (t_face_C - T_outer_C) / cfg.R_cond_back()
    _, _, q_out, _ = q_outside(T_outer_C, cfg.eps_back, cfg, cfg.L_height)
    return q_cond - q_out


def solve_T_steel_outer(t_face_C: float, cfg: SandwichConfig) -> float:
    """Поиск температуры внешней поверхности тыла методом Брента."""
    T_lo = cfg.t_inf_C + 0.05
    T_hi = t_face_C - 0.05
    return brentq(_back_residual, T_lo, T_hi,
                  args=(t_face_C, cfg), xtol=1e-7)


# ── Расчёт температур интерфейсов внутри сэндвича ──────────────────────────


def layer_interface_temps(t_face_C: float, T_steel_outer_C: float,
                          q_w: float, q_loss: float,
                          cfg: SandwichConfig) -> List[float]:
    """Температуры на границах слоёв (от лицевой стороны вглубь).

    Стандартный сценарий 4 слоя: АМг3 → нагреватель → ПИР → сталь.
    Источник тепла — на задней поверхности нагревателя (≈ передней ПИР).
    Назад идёт q_loss, вперёд — q_w. Полагаем нагреватель тонким (без R).
    Поле температур:
      0   = t_face
      x_1 = t_face + q_w·δ_АМг3/λ_АМг3 (тыльная грань АМг3 = передняя нагревателя)
      x_2 = x_1 (по нагревателю без R)
      x_3 = T_steel_outer + q_loss·δ_сталь/λ_сталь (передняя грань стали)
      x_4 = T_steel_outer
    """
    n = len(cfg.layers)
    T = [t_face_C]
    if n == 0:
        return T

    # Если стандартная 4-слойная конструкция:
    if n == 4 and cfg.back_path_indices == (2, 3):
        d_amg, d_h, d_pir, d_st = (L.thickness for L in cfg.layers)
        l_amg, l_h, l_pir, l_st = (L.lam for L in cfg.layers)
        T1 = t_face_C + q_w * d_amg / l_amg            # тыл АМг3
        T2 = T1                                         # нагреватель — без R
        T3 = T_steel_outer_C + q_loss * d_st / l_st    # перед стали (≈T_steel)
        T4 = T_steel_outer_C
        T = [t_face_C, T1, T2, T3, T4]
        return T

    # Обобщённый случай: линейный градиент по back-path слоям.
    # До первого back-path-слоя — t_face (АМг3, нагреватель имеют пренебр. R).
    T_cur = t_face_C
    for i, layer in enumerate(cfg.layers):
        if i in cfg.back_path_indices:
            T_cur -= q_loss * layer.thickness / layer.lam
        T.append(T_cur)
    return T


# ── Структуры результатов ──────────────────────────────────────────────────


@dataclass
class EdgeLoss:
    layer_name: str
    T_avg_C: float
    A_edge_m2: float
    alpha: float
    q_conv_Wm2: float
    q_rad_Wm2: float
    Q_edge_W: float


@dataclass
class ProfileLoss:
    """Потери через алюминиевый профиль с термическим разрывом."""
    T_inner_weighted_C: float    # средневзвешенная T внутренней грани прокладки
    T_profile_C: float           # T внешней поверхности профиля (изотермичен)
    A_profile_m2: float          # площадь внешней поверхности профиля
    alpha: float                 # α на внешней поверхности
    q_conv_Wm2: float
    q_rad_Wm2: float
    q_through_Wm2: float         # удельный поток через прокладку+профиль
    Q_profile_W: float           # суммарные потери через профиль


@dataclass
class RegimeResult:
    regime_idx: int
    q_w: float
    t_face_C: float

    # Тыльный путь
    T_steel_outer_C: float
    q_back_Wm2: float
    Q_back_W: float
    alpha_back: float
    q_conv_back: float
    q_rad_back: float

    # Торцы (как голые — для диагностики/сравнения)
    edges: List[EdgeLoss]
    Q_edges_bare_W: float        # Σ голых торцов

    # Профиль (если включён в cfg.profile.enabled)
    profile_loss: Optional['ProfileLoss']
    Q_edges_effective_W: float   # = Q_profile если профиль вкл, иначе Q_edges_bare

    # Полные значения (с учётом активного режима по торцам)
    Q_useful_W: float
    Q_loss_W: float
    Q_electr_W: float
    eta_back_pct: float
    eta_edges_pct: float
    eta_total_pct: float

    # Профиль температур по толщине
    T_interfaces_C: List[float]


def _layer_avg_temps(t_face_C: float, T_steel_outer_C: float,
                     q_w: float, q_loss: float,
                     cfg: SandwichConfig) -> List[float]:
    """Средняя температура каждого слоя (для расчёта потерь с торца)."""
    T_iface = layer_interface_temps(t_face_C, T_steel_outer_C,
                                    q_w, q_loss, cfg)
    if len(T_iface) < 2:
        return [t_face_C] * len(cfg.layers)
    return [(T_iface[i] + T_iface[i + 1]) / 2.0 for i in range(len(cfg.layers))]


def _T_inner_weighted(T_layer_avgs: List[float],
                      cfg: SandwichConfig) -> float:
    """Средневзвешенная T внутренней стороны рамки профиля
    (взвешивание по толщине каждого слоя в пакете)."""
    total = cfg.total_thickness
    if total <= 0 or not cfg.layers:
        return cfg.t_inf_C
    return sum(L.thickness * T for L, T in zip(cfg.layers, T_layer_avgs)) / total


def _solve_T_profile(T_inner_C: float, cfg: SandwichConfig) -> float:
    """Поиск T внешней поверхности алюминиевого профиля методом Брента.

    Уравнение баланса:
        (T_inner − T_profile) / R_cond = q_conv(T_profile) + q_rad(T_profile)
    """
    R = cfg.profile.R_cond
    if R <= 0:
        return T_inner_C

    def residual(T_p: float) -> float:
        q_through = (T_inner_C - T_p) / R
        q_conv, q_rad, _, _ = q_outside(
            T_p, cfg.profile.eps_profile, cfg, cfg.L_height)
        return q_through - (q_conv + q_rad)

    T_lo = cfg.t_inf_C + 0.05
    T_hi = T_inner_C - 0.01
    if T_hi <= T_lo:
        return cfg.t_inf_C
    return brentq(residual, T_lo, T_hi, xtol=1e-7)


def solve_regime(idx: int, q_w: float, t_face_C: float,
                 cfg: SandwichConfig) -> RegimeResult:
    """Полный расчёт одного режима (тыл + торцы [+ профиль])."""
    # 1) Тыльная поверхность
    T_outer = solve_T_steel_outer(t_face_C, cfg)
    R_back = cfg.R_cond_back()
    q_back = (t_face_C - T_outer) / R_back
    q_conv_b, q_rad_b, _, alpha_b = q_outside(
        T_outer, cfg.eps_back, cfg, cfg.L_height)
    Q_back = q_back * cfg.A_face

    # 2) Голые торцы (всегда считаем — для диагностики и сравнения)
    edges: List[EdgeLoss] = []
    Q_edges_bare = 0.0
    T_layer_avg = _layer_avg_temps(t_face_C, T_outer, q_w, q_back, cfg)
    if cfg.include_edges:
        for layer, T_avg in zip(cfg.layers, T_layer_avg):
            A_edge = cfg.perimeter * layer.thickness
            q_conv, q_rad, _, alpha = q_outside(
                T_avg, layer.eps_edge, cfg, cfg.L_height)
            Q_e = (q_conv + q_rad) * A_edge
            edges.append(EdgeLoss(
                layer_name=layer.name, T_avg_C=T_avg,
                A_edge_m2=A_edge, alpha=alpha,
                q_conv_Wm2=q_conv, q_rad_Wm2=q_rad,
                Q_edge_W=Q_e,
            ))
            Q_edges_bare += Q_e

    # 3) Алюминиевый профиль с термическим разрывом (если включён)
    profile_loss: Optional[ProfileLoss] = None
    if cfg.profile.enabled and cfg.layers and cfg.include_edges:
        T_inner = _T_inner_weighted(T_layer_avg, cfg)
        T_prof = _solve_T_profile(T_inner, cfg)
        R_prof = cfg.profile.R_cond
        q_through = (T_inner - T_prof) / R_prof if R_prof > 0 else 0.0
        q_conv_p, q_rad_p, _, alpha_p = q_outside(
            T_prof, cfg.profile.eps_profile, cfg, cfg.L_height)
        A_prof = cfg.perimeter * cfg.total_thickness
        Q_profile = (q_conv_p + q_rad_p) * A_prof
        profile_loss = ProfileLoss(
            T_inner_weighted_C=T_inner, T_profile_C=T_prof,
            A_profile_m2=A_prof, alpha=alpha_p,
            q_conv_Wm2=q_conv_p, q_rad_Wm2=q_rad_p,
            q_through_Wm2=q_through, Q_profile_W=Q_profile,
        )
        Q_edges_eff = Q_profile
    else:
        Q_edges_eff = Q_edges_bare

    # 4) Балансы
    Q_useful = q_w * cfg.A_face
    Q_loss = Q_back + Q_edges_eff
    Q_electr = Q_useful + Q_loss

    eta_back = Q_back / Q_electr * 100.0
    eta_edges = Q_edges_eff / Q_electr * 100.0
    eta_total = Q_loss / Q_electr * 100.0

    T_interfaces = layer_interface_temps(t_face_C, T_outer, q_w, q_back, cfg)

    return RegimeResult(
        regime_idx=idx,
        q_w=q_w, t_face_C=t_face_C,
        T_steel_outer_C=T_outer,
        q_back_Wm2=q_back, Q_back_W=Q_back,
        alpha_back=alpha_b,
        q_conv_back=q_conv_b, q_rad_back=q_rad_b,
        edges=edges, Q_edges_bare_W=Q_edges_bare,
        profile_loss=profile_loss, Q_edges_effective_W=Q_edges_eff,
        Q_useful_W=Q_useful, Q_loss_W=Q_loss, Q_electr_W=Q_electr,
        eta_back_pct=eta_back, eta_edges_pct=eta_edges,
        eta_total_pct=eta_total,
        T_interfaces_C=T_interfaces,
    )


def solve_all_regimes(regimes: List[Tuple[float, float]],
                      cfg: SandwichConfig) -> List[RegimeResult]:
    return [solve_regime(i + 1, q_w, t_face, cfg)
            for i, (q_w, t_face) in enumerate(regimes)]


# ── Обратная совместимость с прежним CLI ───────────────────────────────────
# (старая константа, оставлена для скриптов / документации)
DELTA_PIR = 0.030
LAMBDA_PIR = 0.022
DELTA_STEEL = 0.003
LAMBDA_STEEL = 50.0
EPS_STEEL = 0.3
A_BACK = 0.2


def _print_cli_report(results: List[RegimeResult], cfg: SandwichConfig) -> None:
    print('=' * 92)
    print('Тепловые потери сэндвич-конструкции (тыл + торцы [+ профиль])')
    print('=' * 92)
    print(f'A_face = {cfg.A_face:.4f} м², периметр = {cfg.perimeter:.3f} м, '
          f'толщина пакета = {cfg.total_thickness*1000:.1f} мм')
    print(f'R_cond (тыл) = {cfg.R_cond_back():.4f} (м²·К)/Вт '
          f'(слои: ' + ', '.join(cfg.layers[i].name
                                  for i in cfg.back_path_indices) + ')')
    print(f'Сталь (тыл): ε = {cfg.eps_back};   среда: t∞ = {cfg.t_inf_C} °C, '
          f'P = {cfg.P_Pa} Па')
    print(f'Корреляция: Черчилль–Чу (вертикальная стенка), '
          f'учёт торцов: {cfg.include_edges}')
    if cfg.profile.enabled:
        print(f'Профиль: ВКЛЮЧЁН — прокладка δ={cfg.profile.delta_break*1000:.1f} мм '
              f'(λ={cfg.profile.lambda_break}) + Al-профиль '
              f'δ={cfg.profile.delta_profile*1000:.1f} мм (ε={cfg.profile.eps_profile}), '
              f'R_cond_проф = {cfg.profile.R_cond:.4f} (м²·К)/Вт')
    else:
        print('Профиль: ОТКЛЮЧЁН (голые торцы)')
    print('=' * 92)

    label = 'Q_profile' if cfg.profile.enabled else 'Q_edges'
    print()
    print(f'### Сводная таблица (активный режим торцов: '
          f'{"профиль" if cfg.profile.enabled else "голые торцы"})')
    print()
    print(f'| # | q_w | t_face | T_steel | Q_back | {label} | Q_loss | '
          f'Q_электр | η_back | η_edges | η_total |')
    print('|---|----:|------:|------:|-----:|------:|------:|-------:|'
          '------:|-------:|------:|')
    for r in results:
        print(f'| {r.regime_idx} | {r.q_w:.0f} | {r.t_face_C:.1f} | '
              f'{r.T_steel_outer_C:.2f} | {r.Q_back_W:.2f} | '
              f'{r.Q_edges_effective_W:.2f} | '
              f'{r.Q_loss_W:.2f} | {r.Q_electr_W:.2f} | '
              f'{r.eta_back_pct:.2f} | {r.eta_edges_pct:.2f} | '
              f'{r.eta_total_pct:.2f} |')

    if cfg.profile.enabled and any(r.profile_loss for r in results):
        print()
        print('### Состояние профиля по режимам')
        print()
        print('| # | T_inner_avg | T_profile | q_through | q_conv | q_rad | '
              'A_proΦ | Q_profile |')
        print('|---|-----------:|---------:|---------:|------:|------:|'
              '-------:|---------:|')
        for r in results:
            p = r.profile_loss
            if p is None:
                continue
            print(f'| {r.regime_idx} | {p.T_inner_weighted_C:.2f} | '
                  f'{p.T_profile_C:.2f} | {p.q_through_Wm2:.2f} | '
                  f'{p.q_conv_Wm2:.2f} | {p.q_rad_Wm2:.2f} | '
                  f'{p.A_profile_m2:.4f} | {p.Q_profile_W:.2f} |')

        # Сравнение «голые торцы vs с профилем»
        r_top = results[-1]
        print()
        print('### Сравнение: голые торцы vs профиль (макс. режим)')
        print()
        print(f'  Q_edges (голые)   = {r_top.Q_edges_bare_W:.1f} Вт   '
              f'→ η = {r_top.Q_edges_bare_W/(r_top.Q_useful_W+r_top.Q_back_W+r_top.Q_edges_bare_W)*100:.1f} %')
        print(f'  Q_profile         = {r_top.Q_edges_effective_W:.1f} Вт   '
              f'→ η = {r_top.eta_edges_pct:.1f} %')
        ratio = (r_top.Q_edges_bare_W / r_top.Q_edges_effective_W
                 if r_top.Q_edges_effective_W > 0 else float('inf'))
        print(f'  Эффект профиля: снижение в {ratio:.2f} раза')

    print()
    print('### Декомпозиция потерь по слоям (торцы, режим максимальный)')
    print()
    if results:
        r_top = results[-1]
        print('| Слой | T_avg, °C | A_edge, м² | α, Вт/(м²·К) | q_conv | '
              'q_rad | Q_edge, Вт |')
        print('|------|---------:|---------:|------------:|------:|------:|---------:|')
        for e in r_top.edges:
            print(f'| {e.layer_name} | {e.T_avg_C:.2f} | {e.A_edge_m2:.4f} | '
                  f'{e.alpha:.2f} | {e.q_conv_Wm2:.2f} | {e.q_rad_Wm2:.2f} | '
                  f'{e.Q_edge_W:.3f} |')

    eta_back_min = min(r.eta_back_pct for r in results)
    eta_back_max = max(r.eta_back_pct for r in results)
    eta_total_min = min(r.eta_total_pct for r in results)
    eta_total_max = max(r.eta_total_pct for r in results)

    eta_edges_min = min(r.eta_edges_pct for r in results)
    eta_edges_max = max(r.eta_edges_pct for r in results)

    print()
    print('### Вывод')
    print()
    print(f'Тыльные потери:    η_back  = {eta_back_min:.2f}…{eta_back_max:.2f} %')
    print(f'Торцевые потери:   η_edges = {eta_edges_min:.2f}…{eta_edges_max:.2f} %')
    print(f'Полные потери:     η_total = {eta_total_min:.2f}…{eta_total_max:.2f} %')

    bare_dominance = (eta_edges_max > 2.0 * eta_back_max
                      and cfg.include_edges
                      and not cfg.profile.enabled)
    if bare_dominance:
        edge_area = sum(L.thickness for L in cfg.layers) * cfg.perimeter
        share = edge_area / cfg.A_face * 100.0
        print()
        print(f'⚠ Торцы доминируют ({eta_edges_max/max(eta_back_max,1e-9):.1f}× '
              f'тыла): площадь торцов {edge_area:.4f} м² = {share:.0f} % '
              f'от лицевой при узкой геометрии (b/L = '
              f'{cfg.b_width/cfg.L_height:.2f}).')
        print(f'  Рекомендация: обернуть торцы фольгированной изоляцией '
              f'(ε ≈ 0.05) или установить алюминиевый профиль с термическим '
              f'разрывом (запустить с флагом --profile).')
    elif cfg.profile.enabled and eta_edges_max > eta_back_max:
        # Профиль включён, но торцы всё ещё доминируют
        print()
        if eta_total_max > 15.0:
            print(f'⚠ Полные потери > 15 % даже с профилем. Меры: увеличить '
                  f'δ_прокладки (МБОР) до 8–10 мм или снизить ε профиля '
                  f'(mill-finish AД31 → ε ≈ 0.10–0.15).')
        elif eta_total_max > 10.0:
            print(f'ℹ Полные потери {eta_total_min:.1f}…{eta_total_max:.1f} % '
                  f'— на верхней границе целевого диапазона. '
                  f'Профиль уже снизил Q_edges в '
                  f'{results[-1].Q_edges_bare_W / max(results[-1].Q_edges_effective_W, 1e-9):.1f} раз.')
        else:
            print(f'✓ Полные потери {eta_total_min:.1f}…{eta_total_max:.1f} % — '
                  f'профиль обеспечивает целевой уровень.')
    else:
        if eta_total_max > 15.0:
            print(f'⚠ Полные потери > 15 % — рекомендуется увеличить δ_ПИР до 50 мм.')
        elif eta_total_max > 10.0:
            print(f'ℹ Полные потери на верхней границе целевого диапазона (10 %).')
        elif eta_total_max >= 5.0:
            print(f'✓ Полные потери в целевом диапазоне 5–10 %.')
        else:
            print(f'✓ Полные потери < 5 % — существенный запас.')


def main():
    import sys
    cfg = default_config()
    if '--profile' in sys.argv or '-p' in sys.argv:
        cfg.profile.enabled = True
    results = solve_all_regimes(DEFAULT_REGIMES, cfg)
    _print_cli_report(results, cfg)


if __name__ == '__main__':
    main()
