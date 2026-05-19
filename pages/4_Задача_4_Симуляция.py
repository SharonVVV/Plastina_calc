"""
Задача №4 — Симулятор лабораторного стенда Керимова.

Имитирует реальный эксперимент: на пластине b × L × δ установлено
20 термопар по высоте. Пользователь задаёт мощность нагревателей и
нажимает «Включить» — стенд прогревается с анимацией ~5 сек до
установившегося режима. Внизу — таблица с реальными показаниями
термопар и блоки с инженерными расчётами по методике.

Визуализация:
  • Тепловая карта пластины (Inferno) с маркерами термопар.
  • Псевдо-BOS / шлирен-картина погранслоя сбоку (|∂T/∂x| в воздухе).
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots


# Явная сборка Inferno-палитры в виде [[fraction, hex], ...].
# Plotly 6.x при двух heatmap'ах с одинаковым именованным `colorscale` и
# разными zmin/zmax иногда «портит» первую запись (#000004 → #ffabab);
# пасс explicit list этого избегает.
_INFERNO_RAW = pc.sequential.Inferno
INFERNO_SCALE = [
    [i / (len(_INFERNO_RAW) - 1), c] for i, c in enumerate(_INFERNO_RAW)
]
# Plotly 6.x при двух heatmap'ах на одной фигуре подменяет очень тёмные
# цвета colorscale на светло-розовый #ffabab (визуальный артефакт).
# Заменяем «почти-чёрный» #000004 на чуть более светлый #0a0418 —
# визуально неотличимо от чёрного, но баг обходится.
INFERNO_SCALE[0] = [0.0, '#0a0418']

from config import DEFAULTS_TASK2, DEFAULTS_ENV
from correlations_meta import ORDER, label_for
from task2_solver import solve_task2, MATERIALS
from plotting_task2 import plot_temperature_profile, plot_alpha_ra_profile


# ── Координаты термопар по высоте (от низа пластины), мм ─────────────────
# Сгущение к низу — там самый интересный участок свободно-конвективного
# пограничного слоя (ламинар, минимальная δ_BL → высокий α).
THERMOCOUPLE_Z_MM: tuple = (
    50.0, 65.0, 80.0, 95.0, 110.0, 130.0, 160.0, 195.0, 235.0, 285.0,
    345.0, 415.0, 505.0, 615.0, 745.0, 900.0, 1095.0, 1325.0, 1610.0, 1950.0,
)

ANIMATION_FRAMES = 22
ANIMATION_DURATION_S = 5.0


# ── Streamlit setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Задача №4 — Симуляция стенда',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      .stMarkdown, .stMarkdown p, .stMarkdown li,
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
      .stDataFrame, .stCaption,
      [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
      [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
      [data-testid="stSelectbox"] {
        font-family: "IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif;
        font-feature-settings: "ss01", "tnum";
      }
      [data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *,
      .material-symbols-rounded, .material-symbols-outlined, .material-icons {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        font-variant-ligatures: common-ligatures !important;
        letter-spacing: normal !important;
      }
      .t4-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 11px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.35rem 0;
      }
      .t4-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.95rem; line-height: 1.15;
        color: #1a1a1a; margin: 0 0 0.35rem 0;
      }
      .t4-lede {
        max-width: 60rem; color: #4b4b4b; font-size: 0.96rem;
        line-height: 1.55; margin: 0 0 1.2rem 0;
      }
      .t4-rule {border:0; border-top:1px solid #d8d8d8; margin:1.6rem 0 1.2rem 0;}
      .t4-section-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.25rem 0;
      }
      .t4-section-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.25rem; line-height: 1.2;
        color: #1a1a1a; margin: 0 0 0.9rem 0;
      }
      .t4-status {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 11.5px; letter-spacing: 0.10em;
        text-transform: uppercase;
      }
      .t4-status.on  {background: #fdebe2; color: #9c3a17;}
      .t4-status.off {background: #eef0f3; color: #4b5563;}
      .t4-status .dot {
        width: 8px; height: 8px; border-radius: 50%;
      }
      .t4-status.on  .dot {background: #d24f2c; box-shadow: 0 0 0 3px rgba(210,79,44,0.18);}
      .t4-status.off .dot {background: #6b7280;}
      .t4-airbox {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.45rem 0.9rem;
        background: #f1f5fb; border: 1px solid #d6e2f0;
        border-radius: 4px;
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 0.92rem; color: #1f3a5f;
        margin: 0.7rem 0 0.3rem 0;
      }
      .t4-airbox b {font-family: "IBM Plex Mono", ui-monospace, monospace;}
      .t4-tc-table {
        width: 100%; border-collapse: collapse;
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 0.92rem; margin: 0.5rem 0 0.4rem 0;
      }
      .t4-tc-table th, .t4-tc-table td {
        padding: 5px 10px; border-bottom: 1px solid #ececec; text-align: left;
      }
      .t4-tc-table th {
        background: #f5f5f3; font-weight: 600; color: #1a1a1a;
        border-bottom: 2px solid #c8c8c8;
        font-size: 0.85rem;
      }
      .t4-tc-table td.num, .t4-tc-table th.num {
        text-align: right;
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-feature-settings: "tnum";
      }
      .t4-tc-table tr.hot td {background: #fcefe6;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="t4-eyebrow">Задача №4</p>
    <h1 class="t4-title">Симулятор лабораторного стенда</h1>
    <p class="t4-lede">
      Виртуальная копия экспериментальной установки. На пластине <b>b × L × δ</b>
      с тыльной стороны установлены силиконовые нагреватели; в металл вдоль
      высоты заделаны <b>20 термопар</b> со сгущением к нижней кромке (там
      наиболее тонкий пограничный слой). Задайте мощность, нажмите
      «Включить стенд» — и понаблюдайте за прогревом до установившегося
      режима. Расчёт ведётся теми же модулями, что и в Задаче №2,
      результаты — в реальном масштабе.
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Session-state по умолчанию ────────────────────────────────────────────

ss = st.session_state
ss.setdefault('t4_power_on', False)
ss.setdefault('t4_animate', False)          # одноразовый триггер анимации
ss.setdefault('t4_anim_direction', 'on')    # 'on' (прогрев) или 'off' (остывание)


# ── Параметры стенда (свёрнутый expander) ─────────────────────────────────

with st.expander('Параметры стенда (геометрия, материал, среда, методика)'):
    pp1, pp2, pp3 = st.columns(3)
    with pp1:
        st.markdown('**Геометрия пластины**')
        b_mm = st.number_input(
            'Ширина b, мм', min_value=10.0, max_value=1000.0,
            value=float(DEFAULTS_TASK2['b_mm']), step=5.0,
            key='t4_b_mm',
        )
        L_mm = st.number_input(
            'Высота L, мм', min_value=100.0, max_value=10000.0,
            value=float(DEFAULTS_TASK2['L_mm']), step=50.0,
            key='t4_L_mm',
        )
        delta_mm = st.number_input(
            'Толщина δ, мм', min_value=0.5, max_value=200.0,
            value=float(DEFAULTS_TASK2['delta_mm']), step=0.5,
            key='t4_delta_mm',
        )
    with pp2:
        st.markdown('**Материал и излучение**')
        material_name = st.selectbox(
            'Материал пластины',
            options=list(MATERIALS.keys()),
            index=list(MATERIALS.keys()).index(DEFAULTS_TASK2['material']),
            key='t4_material',
        )
        lambda_metal = st.number_input(
            'λ металла, Вт/(м·К)',
            min_value=1.0, max_value=600.0, step=5.0,
            value=float(MATERIALS[material_name]['lambda']),
            key='t4_lambda',
        )
        eps_surface = st.number_input(
            'ε поверхности',
            min_value=0.03, max_value=0.99, step=0.01,
            value=float(DEFAULTS_TASK2['eps_surface']),
            key='t4_eps',
            help='Для голого Д16-Т ≈ 0,05…0,15; оксидированный → 0,2…0,3.',
        )
    with pp3:
        st.markdown('**Среда и методика**')
        t_fluid_C = st.number_input(
            't воздуха в помещении, °C',
            min_value=-20.0, max_value=50.0, step=0.5,
            value=float(DEFAULTS_ENV['t_fluid_C']),
            key='t4_t_inf',
        )
        P_Pa = st.number_input(
            'Давление P, Па',
            min_value=80000.0, max_value=120000.0, step=100.0,
            value=float(DEFAULTS_ENV['P_Pa']),
            key='t4_P',
        )
        methodology = st.selectbox(
            'Методика свободной конвекции',
            options=ORDER,
            index=ORDER.index(DEFAULTS_TASK2['methodology']),
            format_func=label_for,
            key='t4_method',
            help='UHF-корреляция для вертикальных граней (лицевая + 2 торца). '
                 'Верх — Леонтьев 1979; днище — только излучение.',
        )

    st.markdown('---')
    axial_conduction = st.checkbox(
        'Учитывать продольную теплопроводность в металле',
        value=True,
        key='t4_axial',
        help='ON (по умолчанию): включён член λ_м·A_кр·d²T/dz² '
             '— тепло перетекает по высоте в самом металле, профиль '
             'T(z) сглаживается (для Д16 масштаб сглаживания ~355 мм). '
             'OFF: каждое сечение решается как локальный баланс '
             '(без переноса по металлу) — получается классический '
             '«хампообразный» профиль с пиком в районе перехода '
             'lam→trans, где α минимален.',
    )

    st.markdown(
        f'**Термопары** ({len(THERMOCOUPLE_Z_MM)} шт.) — '
        f'координаты z от днища в мм: <span style="font-family:IBM Plex Mono">'
        + ', '.join(f'{int(z)}' for z in THERMOCOUPLE_Z_MM) + '</span>',
        unsafe_allow_html=True,
    )

# Узлы солвера — для гладких графиков и анимации (от 121 хватает).
N_NODES_SOLVER = 161
X_MIN_MM = 10.0


# ── Управление мощностью ──────────────────────────────────────────────────

st.markdown('<hr class="t4-rule">', unsafe_allow_html=True)

cc1, cc2, cc3, cc4 = st.columns([3, 1.2, 1.2, 2])
with cc1:
    N_setpoint = st.slider(
        'Мощность нагревателей N, Вт',
        min_value=0.0, max_value=1500.0, step=10.0,
        value=float(DEFAULTS_TASK2['N_total_W']),
        key='t4_N',
        help='Полная электрическая мощность матов. q_w = N / (b·L).',
    )
with cc2:
    st.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)
    btn_on = st.button(
        'Включить стенд', type='primary', use_container_width=True,
        disabled=(N_setpoint <= 0),
    )
with cc3:
    st.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)
    btn_off = st.button(
        'Выключить', use_container_width=True,
        disabled=not ss.t4_power_on,
    )
with cc4:
    st.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)
    if ss.t4_power_on:
        st.markdown(
            '<span class="t4-status on"><span class="dot"></span>стенд работает</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="t4-status off"><span class="dot"></span>стенд выключен</span>',
            unsafe_allow_html=True,
        )

# Обработка кнопок (после отрисовки чтобы получить актуальные значения)
if btn_on and N_setpoint > 0:
    ss.t4_power_on = True
    ss.t4_animate = True
    ss.t4_anim_direction = 'on'
    st.rerun()

if btn_off:
    ss.t4_power_on = False
    ss.t4_animate = True
    ss.t4_anim_direction = 'off'
    st.rerun()


# ── Расчёт установившегося режима ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _solve_steady(key, N_total_W, b_mm, L_mm, delta_mm, lambda_metal,
                  material_name, t_fluid_C, P_Pa, eps_surface,
                  axial_conduction):
    return solve_task2(
        key=key, N_total_W=max(N_total_W, 1e-6),
        b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
        lambda_metal=lambda_metal, material_name=material_name,
        t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps_surface,
        N_nodes=N_NODES_SOLVER, x_min_mm=X_MIN_MM,
        axial_conduction=axial_conduction,
    )


if ss.t4_power_on and N_setpoint > 0:
    with st.spinner('Расчёт установившегося режима…'):
        result = _solve_steady(
            methodology, N_setpoint, b_mm, L_mm, delta_mm,
            lambda_metal, material_name, t_fluid_C, P_Pa, eps_surface,
            axial_conduction,
        )
else:
    result = None


# ── Построение визуализации (heatmap пластины + BOS-шлирен) ───────────────

def _compute_visualization_arrays(L_mm: float, b_mm: float,
                                   t_fluid_C: float,
                                   result_or_none) -> dict:
    """Сетка по высоте и заготовки массивов для отрисовки.

    Возвращает словарь с:
      z_viz (N_z,), T_ss_viz (N_z,) — установившийся профиль (или = T_inf)
      delta_BL_mm (N_z,) — толщина погранслоя на лицевой грани
      x_plate (N_xp,), x_bos_mm (N_xb,)
      T_max_overall — для фиксации цветовой шкалы во время анимации
    """
    N_z = 220
    z_viz = np.linspace(0.0, L_mm, N_z)

    if result_or_none is None:
        T_ss_viz = np.full_like(z_viz, t_fluid_C)
        delta_BL_mm = np.full_like(z_viz, 5.0)
        T_max_overall = t_fluid_C + 0.5
        T_min_overall = t_fluid_C
    else:
        z_nodes_mm = np.array([n.z_m * 1000 for n in result_or_none.nodes])
        T_ss_nodes = np.array([n.T_s_C for n in result_or_none.nodes])
        Ra_nodes = np.array([max(n.Ra_x, 1.0) for n in result_or_none.nodes])

        T_ss_viz = np.interp(z_viz, z_nodes_mm, T_ss_nodes)
        # Толщина погранслоя из числа Грасгофа (для воздуха Pr≈0,71):
        # δ_99 ≈ 5·z·Gr_z^(-1/4) в ламинаре; в турбулентке растёт быстрее
        # — берём грубо δ ~ z·(Gr_z)^(-1/10), но для визуальной картинки
        # достаточно унифицированной формулы.
        delta_BL_mm = np.zeros_like(z_viz)
        for i, z_mm in enumerate(z_viz):
            z_m = max(z_mm / 1000.0, 0.005)
            Ra = float(np.interp(z_mm, z_nodes_mm, Ra_nodes))
            if Ra < 1e3:
                delta_BL_mm[i] = 4.0
                continue
            Gr = Ra / 0.71
            d_lam = 5.0 * z_m * Gr ** (-0.25) * 1000.0   # → мм
            d_turb = 0.37 * z_m * Gr ** (-0.1) * 1000.0  # турб. оценка
            # Плавный переход lam→turb при Gr·Pr ~ 10⁹:
            w = 1.0 / (1.0 + (Ra / 1e9) ** 2)
            delta_BL_mm[i] = max(3.0, w * d_lam + (1 - w) * d_turb)
        T_max_overall = float(T_ss_nodes.max())
        T_min_overall = float(T_ss_nodes.min())

    N_xp = 14
    x_plate = np.linspace(0.0, b_mm, N_xp)
    # Окно течения справа от плиты — 130 мм, на нём рисуем линии тока.
    x_flow_max_mm = 130.0

    # Границы режимов lam → trans → turb по высоте (если есть в расчёте).
    z_trans_mm = None
    z_turb_mm = None
    if result_or_none is not None:
        for n in result_or_none.nodes:
            z_mm_n = n.z_m * 1000
            if z_trans_mm is None and n.regime in ('trans', 'turb'):
                z_trans_mm = z_mm_n
            if z_turb_mm is None and n.regime == 'turb':
                z_turb_mm = z_mm_n

    return dict(
        z_viz=z_viz, T_ss_viz=T_ss_viz, delta_BL_mm=delta_BL_mm,
        x_plate=x_plate, x_flow_max_mm=x_flow_max_mm,
        T_max_overall=T_max_overall, T_min_overall=T_min_overall,
        z_trans_mm=z_trans_mm, z_turb_mm=z_turb_mm,
    )


def _build_figure(state: dict, factor: float,
                  t_fluid_C: float, b_mm: float, L_mm: float,
                  narrow_scale: bool = False) -> go.Figure:
    """factor: 0 — плита холодная (T = T_∞), 1 — установившееся состояние.

    narrow_scale: True — цветовая шкала ограничена реальным диапазоном
        температур плиты (T_min…T_max). Внутренний градиент по высоте
        становится виден. Используется в статике установившегося режима.
        False — шкала от T_∞ до T_max (видна динамика прогрева). Применяется
        во время анимации и для холодного состояния.
    """
    z_viz = state['z_viz']
    T_ss_viz = state['T_ss_viz']
    delta_BL_mm = state['delta_BL_mm']
    x_plate = state['x_plate']
    x_flow_max_mm = state['x_flow_max_mm']
    T_max = state['T_max_overall']
    z_trans_mm = state.get('z_trans_mm')
    z_turb_mm = state.get('z_turb_mm')

    # Текущее распределение T(z)
    T_curr = t_fluid_C + (T_ss_viz - t_fluid_C) * factor

    # 2D температура пластины (по толщине — изотермия)
    T_plate_2d = np.tile(T_curr.reshape(-1, 1), (1, len(x_plate)))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.30, 0.70],
        shared_yaxes=True,
        horizontal_spacing=0.085,        # место для колорбара
        subplot_titles=('Пластина (температурное поле)',
                        'T(z) по показаниям 20 термопар'),
    )

    # Цветовая шкала.
    # Широкая ([T_∞, T_max]): видна динамика прогрева в анимации.
    # Узкая ([T_min_plate, T_max_plate]): виден внутренний градиент плиты
    # в установившемся режиме (когда ΔT_внутренний ≪ T - T_∞).
    T_min_plate = state.get('T_min_overall', t_fluid_C)
    if narrow_scale and (T_max - T_min_plate) > 0.5:
        margin = max(0.3, 0.05 * (T_max - T_min_plate))
        zmin = T_min_plate - margin
        zmax = T_max + margin
    else:
        zmin = t_fluid_C
        zmax = max(T_max, t_fluid_C + 1.0)

    fig.add_trace(
        go.Heatmap(
            z=T_plate_2d, x=x_plate, y=z_viz,
            colorscale=INFERNO_SCALE, zmin=zmin, zmax=zmax,
            colorbar=dict(
                title=dict(text='T, °C', side='right'),
                len=1.0, thickness=14,
                x=0.345, y=0.5, yanchor='middle',
                tickfont=dict(size=10, family='IBM Plex Mono'),
            ),
            hovertemplate='z = %{y:.0f} мм<br>T = %{z:.1f} °C<extra></extra>',
            name='Пластина',
        ),
        row=1, col=1,
    )

    # Маркеры термопар — на правой кромке пластины (там, где обычно
    # заведены провода). Залиты в цвет соответствующей температуры
    # через тот же colorscale.
    T_at_tc = np.interp(np.array(THERMOCOUPLE_Z_MM), z_viz, T_curr)
    fig.add_trace(
        go.Scatter(
            x=[b_mm * 0.92] * len(THERMOCOUPLE_Z_MM),
            y=list(THERMOCOUPLE_Z_MM),
            mode='markers',
            marker=dict(
                symbol='diamond', size=10,
                color=T_at_tc, colorscale=INFERNO_SCALE,
                cmin=zmin, cmax=zmax, showscale=False,
                line=dict(color='#1a1a1a', width=1.1),
            ),
            customdata=np.stack([np.arange(1, len(THERMOCOUPLE_Z_MM) + 1),
                                  T_at_tc], axis=-1),
            hovertemplate='ТП №%{customdata[0]:d} · z = %{y:.0f} мм<br>'
                          'T = %{customdata[1]:.2f} °C<extra></extra>',
            showlegend=False,
            name='Термопары',
        ),
        row=1, col=1,
    )

    # ── График T(z) по термопарам (правая панель) ────────────────────────
    powered = factor > 0.05
    tc_z = np.array(THERMOCOUPLE_Z_MM)
    if powered:
        tc_T = np.interp(tc_z, z_viz, T_curr)
    else:
        tc_T = np.full_like(tc_z, t_fluid_C, dtype=float)

    # Опорная линия t_∞.
    fig.add_shape(
        type='line',
        x0=t_fluid_C, x1=t_fluid_C, y0=0, y1=L_mm,
        line=dict(color='rgba(110,140,180,0.55)', width=1, dash='dot'),
        xref='x2', yref='y2',
    )
    fig.add_annotation(
        x=t_fluid_C, y=L_mm * 0.96, xref='x2', yref='y2',
        text=f'T<sub>∞</sub> = {t_fluid_C:.1f} °C',
        showarrow=False, xanchor='left',
        font=dict(family='IBM Plex Mono, monospace',
                  size=10, color='#5a7395'),
        bgcolor='rgba(255,255,255,0.85)',
    )

    # Соединяющая линия (между термопарами).
    fig.add_trace(
        go.Scatter(
            x=tc_T, y=tc_z, mode='lines',
            line=dict(color='rgba(140,140,140,0.6)', width=1.4),
            hoverinfo='skip', showlegend=False,
            name='T(z)_line',
        ),
        row=1, col=2,
    )

    # Маркеры термопар — закрашены в цвет температуры через тот же
    # Inferno colorscale, что и пластина.
    fig.add_trace(
        go.Scatter(
            x=tc_T, y=tc_z, mode='markers',
            marker=dict(
                symbol='circle', size=11,
                color=tc_T, colorscale=INFERNO_SCALE,
                cmin=zmin, cmax=zmax, showscale=False,
                line=dict(color='#1a1a1a', width=1.0),
            ),
            customdata=np.stack([np.arange(1, len(tc_z) + 1), tc_T,
                                  tc_T - t_fluid_C], axis=-1),
            hovertemplate='ТП №%{customdata[0]:d} · z = %{y:.0f} мм<br>'
                          'T = %{customdata[1]:.2f} °C '
                          '(ΔT = %{customdata[2]:+.1f} K)<extra></extra>',
            showlegend=False,
            name='Термопары',
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=30, t=42, b=40),
        plot_bgcolor='#fbfbfa',
        paper_bgcolor='white',
        font=dict(family='IBM Plex Sans, sans-serif', size=12),
    )
    fig.update_xaxes(
        title_text='b, мм', range=[0, b_mm],
        row=1, col=1, showgrid=False, zeroline=False,
    )
    # Диапазон оси T: от T_∞ - небольшой margin до T_max + margin.
    if powered and T_max > t_fluid_C + 1.0:
        t_left = min(t_fluid_C, float(tc_T.min())) - 5.0
        t_right = max(float(tc_T.max()), T_max) + 5.0
    else:
        t_left = t_fluid_C - 5.0
        t_right = t_fluid_C + 5.0
    fig.update_xaxes(
        title_text='T, °C',
        range=[t_left, t_right], row=1, col=2,
        showgrid=True, gridcolor='#eaeaea', zeroline=False,
    )
    fig.update_yaxes(
        title_text='z, мм (от днища)', range=[0, L_mm],
        row=1, col=1, gridcolor='#eaeaea',
    )
    fig.update_yaxes(
        range=[0, L_mm], row=1, col=2, gridcolor='#eaeaea',
    )

    return fig


# ── Отрисовка с (опциональной) анимацией ──────────────────────────────────

viz_state = _compute_visualization_arrays(L_mm, b_mm, t_fluid_C, result)

st.markdown('<hr class="t4-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t4-section-eyebrow">Стенд</p>'
    '<h2 class="t4-section-title">Температурное поле пластины и пограничный слой</h2>',
    unsafe_allow_html=True,
)

chart_holder = st.empty()

if ss.t4_animate:
    ss.t4_animate = False
    direction = ss.t4_anim_direction
    n_fr = ANIMATION_FRAMES
    dt = ANIMATION_DURATION_S / n_fr
    for i in range(n_fr + 1):
        u = i / n_fr
        # Экспоненциальный выход; коэффициент 4 — к концу анимации
        # фактор ≈ 0.98, последний кадр насильно = 1.
        f = 1.0 - np.exp(-4.0 * u)
        if i == n_fr:
            f = 1.0
        if direction == 'off':
            f = 1.0 - f
        # Во время анимации — широкая шкала, чтобы было видно сам прогрев.
        fig = _build_figure(viz_state, f, t_fluid_C, b_mm, L_mm,
                            narrow_scale=False)
        chart_holder.plotly_chart(
            fig, use_container_width=True,
            config={'displayModeBar': False},
        )
        time.sleep(dt)
    # Финальный кадр в установившемся режиме — переключаем на узкую шкалу,
    # чтобы стал виден внутренний градиент плиты по высоте.
    if direction == 'on':
        fig = _build_figure(viz_state, 1.0, t_fluid_C, b_mm, L_mm,
                            narrow_scale=True)
        chart_holder.plotly_chart(
            fig, use_container_width=True,
            config={'displayModeBar': False},
        )
else:
    factor = 1.0 if ss.t4_power_on else 0.0
    # При включённом стенде — узкая шкала (виден градиент по высоте).
    # При выключенном — широкая (всё равно равномерно T_∞).
    fig = _build_figure(viz_state, factor, t_fluid_C, b_mm, L_mm,
                        narrow_scale=ss.t4_power_on)
    chart_holder.plotly_chart(
        fig, use_container_width=True,
        config={'displayModeBar': False},
    )

st.caption(
    '<b>Слева</b> — температурное поле металлической пластины '
    '(заливка по палитре <i>Inferno</i>). Ромбы на правой кромке — '
    f'позиции {len(THERMOCOUPLE_Z_MM)} термопар (сгущение к нижней '
    'кромке, где пограничный слой самый тонкий и градиенты максимальны). '
    f'<b>Справа</b> — измеренные значения T(z) в {len(THERMOCOUPLE_Z_MM)} '
    'точках термопар: маркеры закрашены в цвет температуры по той же '
    'шкале, тонкая серая линия их соединяет. Пунктирная вертикальная '
    'линия — температура окружающего воздуха T<sub>∞</sub>.',
    unsafe_allow_html=True,
)


# ── Информационный блок: температура воздуха ──────────────────────────────

st.markdown(
    f'<div class="t4-airbox">Температура воздуха в помещении: '
    f'<b>{t_fluid_C:.1f} °C</b>'
    f'&nbsp;·&nbsp;давление: <b>{P_Pa:.0f}</b> Па</div>',
    unsafe_allow_html=True,
)


# ── Таблица термопар ──────────────────────────────────────────────────────

st.markdown('<p class="t4-section-eyebrow" style="margin-top:0.6rem">'
            'Показания термопар</p>'
            f'<h2 class="t4-section-title">Установившийся режим '
            f'({"стенд работает" if ss.t4_power_on else "стенд выключен"})</h2>',
            unsafe_allow_html=True)

# Текущие показания: при выключенном стенде — T_∞, при включённом —
# интерполяция расчётного профиля в точки термопар.
if result is None:
    T_tc = np.full(len(THERMOCOUPLE_Z_MM), t_fluid_C)
    alpha_tc = np.zeros(len(THERMOCOUPLE_Z_MM))
    regime_tc = ['—'] * len(THERMOCOUPLE_Z_MM)
else:
    z_nodes_mm = np.array([n.z_m * 1000 for n in result.nodes])
    T_nodes = np.array([n.T_s_C for n in result.nodes])
    alpha_nodes = np.array([n.alpha_front for n in result.nodes])
    T_tc = np.interp(THERMOCOUPLE_Z_MM, z_nodes_mm, T_nodes)
    alpha_tc = np.interp(THERMOCOUPLE_Z_MM, z_nodes_mm, alpha_nodes)
    # Режим — ближайший узел
    regime_tc = []
    for z_mm in THERMOCOUPLE_Z_MM:
        idx = int(np.argmin(np.abs(z_nodes_mm - z_mm)))
        regime_tc.append(result.nodes[idx].regime)

regime_ru = {
    'lam': 'лам.', 'trans': 'перех.', 'turb': 'турб.',
    'out_of_range': '—', '—': '—',
}

# Сортируем по высоте (по умолчанию уже отсортировано).
rows_html = []
T_max_seen = float(T_tc.max()) if result else t_fluid_C
for i, (z_mm, t_val) in enumerate(
        zip(THERMOCOUPLE_Z_MM, T_tc), start=1):
    hot = result is not None and t_val >= t_fluid_C + 0.7 * (T_max_seen - t_fluid_C)
    cls = 'hot' if hot else ''
    rows_html.append(
        f'<tr class="{cls}">'
        f'<td>ТП №{i:02d}</td>'
        f'<td class="num">{z_mm:.0f}</td>'
        f'<td class="num"><b>{t_val:.2f}</b></td>'
        f'</tr>'
    )

table_html = (
    '<table class="t4-tc-table">'
    '<thead><tr>'
    '<th>№</th>'
    '<th class="num">z, мм</th>'
    '<th class="num">T<sub>тп</sub>, °C</th>'
    '</tr></thead>'
    '<tbody>' + ''.join(rows_html) + '</tbody>'
    '</table>'
)
st.markdown(table_html, unsafe_allow_html=True)

if result is not None:
    st.caption(
        f'KPI: T<sub>min</sub> = <b>{result.T_min_C:.2f}</b> °C · '
        f'T<sub>avg</sub> = <b>{result.T_avg_C:.2f}</b> °C · '
        f'T<sub>max</sub> = <b>{result.T_max_C:.2f}</b> °C · '
        f'q<sub>w</sub> = <b>{result.q_w:.0f}</b> Вт/м² · '
        f'невязка баланса = <b>{result.residual_pct:+.3f}</b>%. '
        f'Методика конвекции: <b>{result.meta.name_ru}</b>. '
        f'Итераций Picard: {result.iterations}.',
        unsafe_allow_html=True,
    )
else:
    st.caption(
        'Стенд выключен: все термопары показывают температуру воздуха в '
        'помещении. Задайте мощность и нажмите «Включить стенд».',
    )


# ── Расчётные блоки по методичке (свёрнуты по умолчанию) ─────────────────

st.markdown('<hr class="t4-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t4-section-eyebrow">Инженерные расчёты по методике</p>'
    '<h2 class="t4-section-title">Анализ свободно-конвективного теплообмена: '
    'локальные характеристики и критерии подобия</h2>',
    unsafe_allow_html=True,
)

if result is None:
    st.info(
        'Расчётные блоки покажут содержимое после первого включения стенда. '
        'Они используют те же модули, что и Задача №2 (Picard-итерация '
        '1D fin-уравнения по высоте).'
    )
else:
    with st.expander('Профиль температуры T(z) по высоте'):
        st.plotly_chart(
            plot_temperature_profile(result),
            use_container_width=True,
            config={'displayModeBar': False},
        )
        st.caption(
            'Сплошная кривая — расчётное распределение T<sub>s</sub>(z). '
            'Полупрозрачные горизонтальные полосы — режим свободной '
            'конвекции на лицевой грани в данной координате z '
            f'(по методике <b>{result.meta.name_ru}</b>). '
            'Метки термопар в реальном эксперименте «снимают» эту кривую '
            'в дискретных точках.',
            unsafe_allow_html=True,
        )

    with st.expander('Коэффициент теплоотдачи α(z) и число Рэлея Ra(z)'):
        st.plotly_chart(
            plot_alpha_ra_profile(result),
            use_container_width=True,
            config={'displayModeBar': False},
        )
        st.caption(
            '<b>α(z)</b> — местный коэф. свободной конвекции (Вт/(м²·К)). '
            'У нижней кромки α максимален (тонкий пограничный слой), '
            'затем падает. На границе lam→trans/turb α может заметно '
            'измениться. <b>Ra(z)</b> (логарифмическая шкала) — критерий, '
            'по которому методика разделяет режимы. Для свободной '
            'конвекции «обычное» число Рейнольдса определяется как '
            'Re<sub>z</sub> = √Gr<sub>z</sub> = √(Ra/Pr) — отдельная '
            'кривая на графике не нужна.',
            unsafe_allow_html=True,
        )

    with st.expander('Определение режимов конвекции — по высоте'):
        # Сводка: где у нас лам, где переход, где турб (если есть)
        z_nodes_mm = np.array([n.z_m * 1000 for n in result.nodes])
        regs = [n.regime for n in result.nodes]
        # Найдём границы переходов
        transitions = []
        prev = regs[0]
        prev_z = z_nodes_mm[0]
        for i, r in enumerate(regs[1:], start=1):
            if r != prev:
                transitions.append((prev, prev_z, z_nodes_mm[i]))
                prev = r
                prev_z = z_nodes_mm[i]
        transitions.append((prev, prev_z, z_nodes_mm[-1]))

        rows = ['<table class="t4-tc-table"><thead><tr>'
                '<th>Режим</th><th class="num">z от, мм</th>'
                '<th class="num">z до, мм</th>'
                '<th class="num">высота зоны, мм</th></tr></thead><tbody>']
        for reg, z0, z1 in transitions:
            rows.append(
                f'<tr><td>{regime_ru.get(reg, reg)}</td>'
                f'<td class="num">{z0:.0f}</td>'
                f'<td class="num">{z1:.0f}</td>'
                f'<td class="num">{(z1 - z0):.0f}</td></tr>'
            )
        rows.append('</tbody></table>')
        st.markdown(''.join(rows), unsafe_allow_html=True)
        st.caption(
            f'Методика <b>{result.meta.name_ru}</b> применяет разные '
            'формулы Nu в каждой зоне. На границах α(z) может иметь '
            'разрыв производной — это нормально и отражено в формулах.',
            unsafe_allow_html=True,
        )

    with st.expander('Ключевые числа подобия в точках термопар'):
        # Таблица: z, Ra, Gr*·Pr, α, режим
        z_nodes_mm = np.array([n.z_m * 1000 for n in result.nodes])
        Ra_nodes = np.array([n.Ra_x for n in result.nodes])
        GrPr_star_nodes = np.array([n.GrPr_star for n in result.nodes])
        alpha_nodes_full = np.array([n.alpha_front for n in result.nodes])
        regs_arr = np.array([n.regime for n in result.nodes])

        rows = ['<table class="t4-tc-table"><thead><tr>'
                '<th>№</th><th class="num">z, мм</th>'
                '<th class="num">Ra<sub>z</sub></th>'
                '<th class="num">Gr*·Pr</th>'
                '<th class="num">α, Вт/(м²·К)</th>'
                '<th>Режим</th></tr></thead><tbody>']
        for i, z_mm in enumerate(THERMOCOUPLE_Z_MM, start=1):
            idx = int(np.argmin(np.abs(z_nodes_mm - z_mm)))
            Ra_v = Ra_nodes[idx]
            GrPr = GrPr_star_nodes[idx]
            a_val = alpha_nodes_full[idx]
            reg = regs_arr[idx]
            rows.append(
                f'<tr><td>ТП №{i:02d}</td>'
                f'<td class="num">{z_mm:.0f}</td>'
                f'<td class="num">{Ra_v:.2e}</td>'
                f'<td class="num">{GrPr:.2e}</td>'
                f'<td class="num">{a_val:.2f}</td>'
                f'<td>{regime_ru.get(reg, reg)}</td></tr>'
            )
        rows.append('</tbody></table>')
        st.markdown(''.join(rows), unsafe_allow_html=True)
        st.caption(
            'Ra<sub>z</sub> = g·β·ΔT·z³·Pr/ν² — основной критерий режима '
            'для UHF-методик с опорной t_∞ (как у Керимова 1992). '
            'Gr*·Pr = g·β·q<sub>w</sub>·z⁴·Pr/(λ·ν²) — модифицированный '
            'критерий через тепловой поток, эквивалентен Ra*<sub>z</sub>. '
            'α = Nu<sub>z</sub>·λ<sub>возд</sub>/z — местный коэффициент '
            'теплоотдачи свободной конвекцией на лицевой грани.',
            unsafe_allow_html=True,
        )
