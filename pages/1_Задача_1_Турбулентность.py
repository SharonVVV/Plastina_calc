"""
Задача №1. Высота достижения турбулентного режима естественной конвекции
на вертикальной плите (UHF, q_w = const).

Оба варианта нагрева — Джоулев (ток I) и силиконовые маты (мощность N) —
ставятся как граничное условие 2-го рода, q_w = const. Для каждой из 7
верифицированных локальных UHF-корреляций (см. correlations_uhf.py) считаем
профиль T_s(x), границы режимов и обратную задачу — минимальный ток / мощность
для достижения турбулентности на доле высоты (по умолчанию 80%).

Все формулы сверены по PDF/сканам первоисточников:
/Users/SharonovVV/Downloads/convection_methods.md.
"""

import streamlit as st
import pandas as pd

from config import (
    DEFAULTS_TASK1, DEFAULTS_HEATER, DEFAULTS_QCONST,
    DEFAULTS_ENV, DEFAULTS_CALC,
)
from correlations_meta import META, ORDER, label_for
from task1_solver import (
    run_methodology_joule, run_methodology_qconst,
    find_min_input_for_turbulence,
    q_w_from_N, q_w_from_I_at_T,
)
from plotting_task1 import plot_methodology_strips


st.set_page_config(
    page_title='Задача №1 — Турбулентный режим',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Типографика и сетка (IBM Plex + хайрлайн-разделители) ──────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      /* Применяем IBM Plex только к текстовым элементам. Иконки Streamlit
         (стрелки expander'ов и т.п.) используют шрифт-лигатуру Material
         Symbols — его НЕЛЬЗЯ перебивать, иначе текст «keyboard_arrow_right»
         покажется буквами. */
      .stMarkdown, .stMarkdown p, .stMarkdown li,
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
      .stDataFrame, .stCaption,
      [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input {
        font-family: "IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif;
        font-feature-settings: "ss01", "tnum";
      }
      /* Возвращаем иконкам их icon-font ПРИНУДИТЕЛЬНО.
         Critically: Material Symbols использует лигатуры — нужен
         font-feature-settings: 'liga'. Мой ss01/tnum выше отрубал liga. */
      [data-testid="stIconMaterial"],
      [data-testid="stIconMaterial"] *,
      .material-symbols-rounded, .material-symbols-outlined, .material-icons {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        font-variant-ligatures: common-ligatures !important;
        letter-spacing: normal !important;
      }
      .task1-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 11px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.35rem 0;
      }
      .task1-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.95rem; line-height: 1.15;
        color: #1a1a1a; margin: 0 0 0.35rem 0;
      }
      .task1-lede {
        max-width: 60rem; color: #4b4b4b; font-size: 0.96rem;
        line-height: 1.55; margin: 0 0 1.5rem 0;
      }
      .task1-rule {
        border: 0; border-top: 1px solid #d8d8d8;
        margin: 1.6rem 0 1.2rem 0;
      }
      .task1-section-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.25rem 0;
      }
      .task1-section-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.25rem; line-height: 1.2;
        color: #1a1a1a; margin: 0 0 0.9rem 0;
      }
      .task1-plot-head {
        display: flex; align-items: baseline; gap: 1rem;
        margin: 0 0 0.4rem 0;
      }
      .task1-plot-head .ph-label {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
      }
      .task1-plot-head .ph-value {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-weight: 600; font-size: 1.05rem; color: #1a1a1a;
      }
      .task1-plot-head .ph-meta {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.85rem; color: #6b6b6b;
      }
      .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        font-feature-settings: "tnum";
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
      <p class="task1-eyebrow">Задача №1</p>
      <h1 class="task1-title">Высота достижения турбулентного режима</h1>
      <p class="task1-lede">
        Сравнение семи верифицированных локальных UHF-корреляций
        Nu_x = f(Ra, Pr) для свободной конвекции у вертикальной плиты.
        Толщина пренебрежимо мала (двумерная задача), нагрев — постоянный
        тепловой поток q<sub>w</sub> = const: Джоулев через ток I либо
        силиконовые маты с суммарной мощностью N.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Основные параметры ─────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    b_mm = st.number_input('Ширина b, мм',
                           min_value=10.0, max_value=1000.0,
                           value=float(DEFAULTS_TASK1['b_mm']), step=10.0)
with c2:
    L_mm = st.number_input('Высота L, мм',
                           min_value=100.0, max_value=10000.0,
                           value=float(DEFAULTS_TASK1['L_mm']), step=50.0)
with c3:
    I = st.number_input('Ток Джоуля I, А',
                        min_value=1.0, max_value=3000.0,
                        value=float(DEFAULTS_TASK1['I']), step=10.0)
with c4:
    N_total_W = st.number_input(
        'Суммарная мощность матов N, Вт',
        min_value=10.0, max_value=10000.0,
        value=float(DEFAULTS_TASK1['N_total_W']),
        step=10.0,
        help='Полная электрическая мощность всех матов вместе. Распределяется '
             'равномерно по площади пластины b·L; q_w = N/(b·L).',
    )


# ── Скрытые параметры ─────────────────────────────────────────────────────

with st.expander('Доп. параметры (среда, излучение, методики, точность)'):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown('**Среда**')
        t_fluid_C = st.number_input('t_∞, °C',
                                    value=float(DEFAULTS_ENV['t_fluid_C']),
                                    min_value=-50.0, max_value=200.0, step=1.0)
        P_Pa = st.number_input('Давление P, Па',
                               value=float(DEFAULTS_ENV['P_Pa']),
                               min_value=50000.0, max_value=200000.0,
                               step=100.0)
        g = float(DEFAULTS_ENV['g'])
    with cc2:
        st.markdown('**Излучение и Джоуль**')
        eps_surface = st.number_input(
            'ε поверхности', value=float(DEFAULTS_QCONST['eps_surface']),
            min_value=0.05, max_value=0.99, step=0.05,
            help='Степень черноты — используется и для матов, и для Джоуля. '
                 'q_rad = ε·σ·(T_s⁴ − T_∞⁴).')
        R20 = st.number_input(
            'R₂₀, Ом/м', value=float(DEFAULTS_HEATER['R20']),
            min_value=1e-4, max_value=1e-1, step=1e-4, format='%.4e',
            help='Погонное сопротивление плиты при 20 °C — для Джоулева q_w.')
        alpha_R = st.number_input(
            'α_R, К⁻¹', value=float(DEFAULTS_HEATER['alpha_R']),
            min_value=0.0, max_value=1e-2, step=1e-4, format='%.4e',
            help='ТКС — для Джоулева q_w(T_s).')
    with cc3:
        st.markdown('**Расчёт**')
        N_points = st.number_input(
            'Точек по высоте', value=int(DEFAULTS_TASK1['N_points']),
            min_value=20, max_value=400, step=10)
        target_pct = st.slider(
            'Целевая высота турб. режима, %',
            min_value=10, max_value=100,
            value=int(DEFAULTS_TASK1['target_fraction'] * 100), step=5)
        x_min_mm = st.number_input(
            'x_min, мм', value=float(DEFAULTS_CALC['x_min_mm']),
            min_value=1.0, max_value=50.0, step=1.0,
            help='Начальная координата (избегает сингулярности x→0).')

    st.markdown('**Выбор методик**')
    methods_selected = st.multiselect(
        'Сравниваем эти локальные UHF-корреляции',
        options=ORDER, default=ORDER, format_func=label_for,
    )

target_fraction = target_pct / 100.0
target_x_m = target_fraction * L_mm / 1000.0

if not methods_selected:
    st.warning('Выберите хотя бы одну методику в «Доп. параметрах».')
    st.stop()


# ── Прямые расчёты ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_joule_all(methods, I, R20, alpha_R, b_mm, L_mm,
                   t_fluid_C, g, P_Pa, eps_surface, x_min_mm, N_points):
    return [run_methodology_joule(
        k, I=I, R20=R20, alpha_R=alpha_R,
        b_mm=b_mm, L_mm=L_mm,
        t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
        eps_surface=eps_surface, x_min_mm=x_min_mm, N=N_points,
    ) for k in methods]


@st.cache_data(show_spinner=False)
def _run_qconst_all(methods, N_total_W, b_mm, L_mm,
                    t_fluid_C, g, P_Pa, eps_surface, x_min_mm, N_points):
    return [run_methodology_qconst(
        k, N_total_W=N_total_W,
        b_mm=b_mm, L_mm=L_mm,
        t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
        eps_surface=eps_surface, x_min_mm=x_min_mm, N=N_points,
    ) for k in methods]


with st.spinner('Прямые расчёты по методикам…'):
    joule_results = _run_joule_all(
        tuple(methods_selected), I, R20, alpha_R, b_mm, L_mm,
        t_fluid_C, g, P_Pa, eps_surface, x_min_mm, int(N_points),
    )
    qconst_results = _run_qconst_all(
        tuple(methods_selected), N_total_W, b_mm, L_mm,
        t_fluid_C, g, P_Pa, eps_surface, x_min_mm, int(N_points),
    )


# ── Обратные задачи ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _inv_joule_all(methods, target_x_m, R20, alpha_R, b_mm, L_mm,
                   t_fluid_C, g, P_Pa, eps_surface, x_min_mm, N_points):
    return {k: find_min_input_for_turbulence(
        mode='joule', key=k, target_x_m=target_x_m,
        b_mm=b_mm, L_mm=L_mm,
        t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
        eps_surface=eps_surface, x_min_mm=x_min_mm, N=N_points,
        R20=R20, alpha_R=alpha_R,
        search_min=float(DEFAULTS_TASK1['I_search_min']),
        search_max=float(DEFAULTS_TASK1['I_search_max']),
    ) for k in methods}


@st.cache_data(show_spinner=False)
def _inv_qconst_all(methods, target_x_m, b_mm, L_mm,
                    t_fluid_C, g, P_Pa, eps_surface, x_min_mm, N_points):
    return {k: find_min_input_for_turbulence(
        mode='qconst', key=k, target_x_m=target_x_m,
        b_mm=b_mm, L_mm=L_mm,
        t_fluid_C=t_fluid_C, g=g, P_Pa=P_Pa,
        eps_surface=eps_surface, x_min_mm=x_min_mm, N=N_points,
        search_min=float(DEFAULTS_TASK1['N_search_min']),
        search_max=float(DEFAULTS_TASK1['N_search_max']),
    ) for k in methods}


with st.spinner('Обратная задача: минимальный I и N…'):
    inv_joule = _inv_joule_all(
        tuple(methods_selected), target_x_m, R20, alpha_R, b_mm, L_mm,
        t_fluid_C, g, P_Pa, eps_surface, x_min_mm, int(N_points),
    )
    inv_qconst = _inv_qconst_all(
        tuple(methods_selected), target_x_m, b_mm, L_mm,
        t_fluid_C, g, P_Pa, eps_surface, x_min_mm, int(N_points),
    )


# ── Сводные таблицы ────────────────────────────────────────────────────────

def _fmt_mm(x_m):
    return '—' if x_m is None else f'{x_m * 1000:.0f} мм'


def _fmt_inv(res, units):
    if res is None:
        return '—'
    if not res.achievable:
        return 'не достигается'
    return f'{res.value:.1f} {units}'


def _param_at(point, boundary_param: str) -> float:
    """Выбор фактического значения параметра, по которому методика классифицирует
    режим: 'Ra' — Ra_x (через ΔT); 'Gr*·Pr' и 'Ra*' — Gr_x*·Pr (через q_w).
    Безопасен к point=None — вернёт NaN."""
    if point is None:
        return float('nan')
    if boundary_param == 'Ra':
        return point.Ra_x
    return point.GrPr_star


def _summary_df(results, inverse_map, units_inv):
    rows = []
    for mr in results:
        last = next((p for p in reversed(mr.points) if p.converged), None)
        bp = mr.meta.boundary_param
        v = _param_at(last, bp)
        param_at_L = '—' if v != v else f'{v:.2e}'   # NaN-проверка через v != v
        rows.append({
            'Методика': mr.meta.name_ru,
            'Параметр': bp,
            'lam→trans/turb': _fmt_mm(mr.x_lam_to_trans_m),
            'turb (начало)': _fmt_mm(mr.x_turb_start_m),
            f'{bp} на x=L': param_at_L,
            f'Минимум для турб. на {target_pct}%·L':
                _fmt_inv(inverse_map.get(mr.key), units_inv),
        })
    return pd.DataFrame(rows)


# ── Отрисовка ──────────────────────────────────────────────────────────────

q_w_mats = q_w_from_N(N_total_W, b_mm, L_mm)
q_w_joule_at_20 = q_w_from_I_at_T(I, R20, b_mm, t_s_C=20.0, alpha_R=alpha_R)

def _plot_head(eyebrow: str, value_html: str, meta_html: str) -> str:
    return (
        f'<div class="task1-plot-head">'
        f'<span class="ph-label">{eyebrow}</span>'
        f'<span class="ph-value">{value_html}</span>'
        f'<span class="ph-meta">{meta_html}</span>'
        f'</div>'
    )


def _render_block(eyebrow: str, value_html: str, meta_html: str,
                  results, inverse_map, units_inv):
    """Один блок: диаграмма во всю ширину + таблица под ней."""
    st.markdown(
        _plot_head(eyebrow, value_html, meta_html),
        unsafe_allow_html=True,
    )
    fig = plot_methodology_strips(
        width_mm=b_mm, height_mm=L_mm,
        results=results, target_fraction=target_fraction,
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': False})
    st.caption(
        'Внутри полос: пунктир — граница lam→trans; тире — начало '
        'турбулентного режима. Числовые координаты переходов — под '
        f'каждой полосой. Цель {target_pct}% × L используется в обратной '
        'задаче в таблице ниже.'
    )
    st.dataframe(_summary_df(results, inverse_map, units_inv),
                 hide_index=True, use_container_width=True)


# ── Раздел 1: Джоулев нагрев ───────────────────────────────────────────────

st.markdown('<hr class="task1-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="task1-section-eyebrow">Раздел&nbsp;1 · Джоулев нагрев</p>'
    f'<h2 class="task1-section-title">'
    f'Пластина b × L = {b_mm:.0f} × {L_mm:.0f} мм, '
    f'ток I = {I:.0f} А</h2>',
    unsafe_allow_html=True,
)
_render_block(
    'Джоулев нагрев',
    f'I = {I:.0f} А',
    f'q<sub>w</sub> ≈ {q_w_joule_at_20:.0f} Вт/м² (при 20 °C)',
    joule_results, inv_joule, 'А',
)


# ── Раздел 2: Силиконовые маты ────────────────────────────────────────────

st.markdown('<hr class="task1-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="task1-section-eyebrow">Раздел&nbsp;2 · Силиконовые маты</p>'
    f'<h2 class="task1-section-title">'
    f'Пластина b × L = {b_mm:.0f} × {L_mm:.0f} мм, '
    f'суммарная мощность N = {N_total_W:.0f} Вт</h2>',
    unsafe_allow_html=True,
)
_render_block(
    'Силиконовые маты',
    f'N = {N_total_W:.0f} Вт',
    f'q<sub>w</sub> = {q_w_mats:.0f} Вт/м²',
    qconst_results, inv_qconst, 'Вт',
)


st.caption(
    'Параметр — критерий, по которому методика классифицирует режим '
    '(Ra через ΔT, Gr*·Pr или Ra* через q_w). '
    'lam→trans/turb — координата перехода из ламинарного в переходный '
    '(либо сразу в турбулент для стыковых методик). '
    'turb (начало) — координата начала турбулентного режима. '
    f'Минимум для турб. на {target_pct}%·L — наименьший ток (А) или '
    'мощность матов (Вт), при котором турбулент достигается не выше '
    f'{target_pct}% от высоты пластины (бисекция, ±0,5 в финале).'
)


# ── Характеристические уравнения ──────────────────────────────────────────

st.markdown('<hr class="task1-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="task1-section-eyebrow">Раздел&nbsp;3 · Корреляции</p>'  # noqa

    '<h2 class="task1-section-title">'
    'Характеристические уравнения, использованные в расчёте</h2>',
    unsafe_allow_html=True,
)

with st.expander('Развернуть формулы и диапазоны применимости'):
    st.markdown(
        'Все формулы — **локальные**, для UHF (q_w = const), свойства воздуха '
        'при плёночной T_f = (T_s + T_∞)/2. Сверены по первоисточникам '
        '(см. справочник `convection_methods.md`).'
    )
    for key in methods_selected:
        m = META[key]
        st.markdown(f'### {m.name_ru}')
        st.caption(m.full_ru)
        if m.nu_lam_tex:
            st.markdown('**Ламинарный режим:**')
            st.latex(m.nu_lam_tex)
        if m.nu_turb_tex:
            st.markdown('**Турбулентный режим:**')
            st.latex(m.nu_turb_tex)
        bounds = []
        if m.lam_max is not None:
            bounds.append(f'ламинар: {m.boundary_param} ≤ {m.lam_max:.2e}')
        if m.turb_min is not None:
            bounds.append(f'турбулент: {m.boundary_param} ≥ {m.turb_min:.2e}')
        if bounds:
            st.markdown('**Границы режимов:** ' + '; '.join(bounds) + '.')
        if m.notes_ru:
            st.caption(m.notes_ru)
        st.divider()
