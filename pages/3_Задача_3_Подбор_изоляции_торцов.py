"""
Задача №3. Подбор изоляции для торцов плиты.

Та же постановка, что в Задаче 2 (плита b × L × δ, нагрев q_w = const с
тыльной стороны, теплоотвод с лицевой грани и торцов), но **торцы могут
быть утеплены** слоем теплоизоляции. Изоляция работает как последовательное
сопротивление:

    1/h_eff = δ_изо/λ_изо + 1/α_внеш

где α_внеш — коэф. конвекции на внешней поверхности изоляции (имеет ту же
ориентацию, что и изначальный торец, поэтому приближённо берётся равным
нативному α голого металла). Через изоляцию радиация не проходит — потери
излучением с изолированных торцов считаются нулевыми (по ТЗ).

Страница позволяет:
  • Включить/выключить изоляцию на каждом из трёх типов торцов (боковые,
    верхний, днище);
  • Выбрать материал изоляции (λ из справочника) и толщину;
  • Сравнить баланс «без изоляции / с текущей конфигурацией»;
  • Прогон-чувствительность: как меняется доля потерь по торцам в
    зависимости от толщины изоляции (sweep);
  • Сравнение всех материалов при текущей толщине.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULTS_TASK2, DEFAULTS_ENV
from correlations_meta import META, ORDER, label_for
from task2_solver import solve_task2, MATERIALS, INSULATION_MATERIALS


st.set_page_config(
    page_title='Задача №3 — Подбор изоляции торцов',
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
      .t3-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 11px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.35rem 0;
      }
      .t3-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.95rem; line-height: 1.15;
        color: #1a1a1a; margin: 0 0 0.35rem 0;
      }
      .t3-lede {
        max-width: 60rem; color: #4b4b4b; font-size: 0.96rem;
        line-height: 1.55; margin: 0 0 1.5rem 0;
      }
      .t3-rule {border:0; border-top:1px solid #d8d8d8; margin:1.6rem 0 1.2rem 0;}
      .t3-section-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.25rem 0;
      }
      .t3-section-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.25rem; line-height: 1.2;
        color: #1a1a1a; margin: 0 0 0.9rem 0;
      }
      .t3-kpi {display: flex; gap: 1.4rem; flex-wrap: wrap; margin: 0 0 1.0rem 0;}
      .t3-kpi .item {min-width: 110px;}
      .t3-kpi .label {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.10em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 2px 0;
      }
      .t3-kpi .value-mono {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-weight: 500; font-size: 1.2rem; color: #1a1a1a;
      }
      .t3-cmp {
        width: 100%; border-collapse: collapse;
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 0.92rem; margin: 0.5rem 0;
      }
      .t3-cmp th, .t3-cmp td {
        padding: 7px 11px; border-bottom: 1px solid #e6e6e6; text-align: left;
      }
      .t3-cmp th {
        background: #f5f5f3; font-weight: 600; color: #1a1a1a;
        border-bottom: 2px solid #c8c8c8;
      }
      .t3-cmp td.num, .t3-cmp th.num {
        text-align: right;
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-feature-settings: "tnum";
      }
      .t3-cmp tr.path td { color: #2a2a2a; }
      .t3-cmp tr.totals td {
        font-weight: 700; background: #efeee9; border-top: 2px solid #888;
      }
      .t3-cmp td.delta-pos {color: #1b6a37;}
      .t3-cmp td.delta-neg {color: #a93636;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="t3-eyebrow">Задача №3</p>
    <h1 class="t3-title">Подбор изоляции для торцов</h1>
    <p class="t3-lede">
      Та же физика что в Задаче 2 (плита b × L × δ, нагрев q<sub>w</sub> = const
      с тыла, теплоотвод свободной конвекцией и излучением с боковых граней
      и торцов), но <b>на торцы можно надеть слой теплоизоляции</b>.
      Через изолированный торец радиация не проходит, а конвекция идёт уже
      с внешней поверхности изоляции (через последовательное тепловое
      сопротивление 1/h<sub>eff</sub> = δ<sub>изо</sub>/λ<sub>изо</sub> + 1/α<sub>внеш</sub>).
      Цель страницы — увидеть, как меняются доли потерь при разной толщине
      и разных материалах изоляции.
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Основные параметры плиты ─────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    b_mm = st.number_input('Ширина b, мм',
                           min_value=10.0, max_value=1000.0,
                           value=float(DEFAULTS_TASK2['b_mm']), step=10.0)
with c2:
    L_mm = st.number_input('Высота L, мм',
                           min_value=100.0, max_value=10000.0,
                           value=float(DEFAULTS_TASK2['L_mm']), step=50.0)
with c3:
    delta_mm = st.number_input('Толщина δ, мм',
                               min_value=0.5, max_value=200.0,
                               value=float(DEFAULTS_TASK2['delta_mm']),
                               step=0.5)
with c4:
    N_total_W = st.number_input(
        'Суммарная мощность матов N, Вт',
        min_value=10.0, max_value=20000.0,
        value=float(DEFAULTS_TASK2['N_total_W']), step=50.0,
    )

c5, c6 = st.columns(2)
with c5:
    material_name = st.selectbox(
        'Материал пластины',
        options=list(MATERIALS.keys()),
        index=list(MATERIALS.keys()).index(DEFAULTS_TASK2['material']),
    )
with c6:
    methodology = st.selectbox(
        'Методика свободной конвекции (вертикальные грани)',
        options=ORDER,
        index=ORDER.index(DEFAULTS_TASK2['methodology']),
        format_func=label_for,
    )
mat = MATERIALS[material_name]


# ── Доп. параметры ────────────────────────────────────────────────────────

with st.expander('Доп. параметры (среда, излучение, сетка, λ_металла)'):
    cc1, cc2 = st.columns(2)
    with cc1:
        t_fluid_C = st.number_input(
            't_∞, °C', value=float(DEFAULTS_ENV['t_fluid_C']),
            min_value=-50.0, max_value=200.0, step=1.0,
        )
        P_Pa = st.number_input(
            'Давление P, Па', value=float(DEFAULTS_ENV['P_Pa']),
            min_value=50000.0, max_value=200000.0, step=100.0,
        )
        eps_surface = st.number_input(
            'ε поверхности (голый металл)',
            value=float(DEFAULTS_TASK2['eps_surface']),
            min_value=0.03, max_value=0.99, step=0.01,
        )
    with cc2:
        N_nodes = st.number_input(
            'Узлов по высоте', value=int(DEFAULTS_TASK2['N_nodes']),
            min_value=21, max_value=501, step=10,
        )
        lambda_metal = st.number_input(
            'λ металла, Вт/(м·К)', value=float(mat['lambda']),
            min_value=1.0, max_value=600.0, step=5.0,
        )
        x_min_mm = st.number_input(
            'x_min (UHF), мм', value=float(DEFAULTS_TASK2['x_min_mm']),
            min_value=1.0, max_value=50.0, step=1.0,
        )
g = float(DEFAULTS_ENV['g'])


# ── Конфигурация изоляции торцов ──────────────────────────────────────────

st.markdown('<hr class="t3-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t3-section-eyebrow">Конфигурация изоляции</p>'
    '<h2 class="t3-section-title">Какие торцы утеплять и чем</h2>',
    unsafe_allow_html=True,
)

iso_col1, iso_col2 = st.columns([1, 2])
with iso_col1:
    iso_material = st.selectbox(
        'Материал изоляции',
        options=list(INSULATION_MATERIALS.keys()),
        index=4,  # Минеральная вата по умолчанию
        format_func=lambda k: f"{k} — λ = {INSULATION_MATERIALS[k]['lambda']:.3f} Вт/(м·К)",
    )
    iso_lambda = INSULATION_MATERIALS[iso_material]['lambda']
    iso_thickness_mm = st.number_input(
        'Толщина изоляции δ_изо, мм',
        min_value=0.0, max_value=200.0, value=20.0, step=1.0,
        help='Толщина применяется ко всем выбранным торцам одинаково.',
    )

with iso_col2:
    st.markdown('**Какие торцы покрыть изоляцией:**')
    iso_sides = st.checkbox(
        'Боковые торцы (2 шт., каждый δ × L)', value=True,
        help='Самый значимый канал потерь — площадь 2·δ·L ≈ 24% от лицевой грани.',
    )
    iso_top = st.checkbox(
        'Верхний торец (b × δ)', value=True,
    )
    iso_bottom = st.checkbox(
        'Нижний торец «днище» (b × δ)', value=True,
        help='Изолированное днище фактически перекрывает радиационный сток '
             '(нативной конвекции там и так нет).',
    )
    st.caption(
        f'Текущий выбор: <b>{iso_material}</b>, λ_изо = '
        f'<span style="font-family:IBM Plex Mono">{iso_lambda:.3f}</span> Вт/(м·К). '
        f'Толщина = <span style="font-family:IBM Plex Mono">'
        f'{iso_thickness_mm:.1f}</span> мм. '
        f'<i>{INSULATION_MATERIALS[iso_material]["note"]}</i>',
        unsafe_allow_html=True,
    )

sides_t = iso_thickness_mm if iso_sides else 0.0
top_t = iso_thickness_mm if iso_top else 0.0
bottom_t = iso_thickness_mm if iso_bottom else 0.0


# ── Расчёты ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run(key, N_total_W, b_mm, L_mm, delta_mm, lambda_metal, material_name,
         t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g,
         st_mm, tt_mm, bt_mm, iso_lambda):
    return solve_task2(
        key=key, N_total_W=N_total_W,
        b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
        lambda_metal=lambda_metal, material_name=material_name,
        t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps_surface,
        N_nodes=int(N_nodes), x_min_mm=x_min_mm, g=g,
        sides_iso_thickness_mm=st_mm, sides_iso_lambda=iso_lambda,
        top_iso_thickness_mm=tt_mm, top_iso_lambda=iso_lambda,
        bottom_iso_thickness_mm=bt_mm, bottom_iso_lambda=iso_lambda,
    )


with st.spinner('Расчёт baseline + конфигурация с изоляцией…'):
    r_base = _run(methodology, N_total_W, b_mm, L_mm, delta_mm,
                  lambda_metal, material_name,
                  t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g,
                  0.0, 0.0, 0.0, iso_lambda)
    r_iso = _run(methodology, N_total_W, b_mm, L_mm, delta_mm,
                 lambda_metal, material_name,
                 t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g,
                 sides_t, top_t, bottom_t, iso_lambda)


# ── Сравнение балансов ──────────────────────────────────────────────────

st.markdown('<hr class="t3-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t3-section-eyebrow">Раздел 1 · Сравнение балансов</p>'
    '<h2 class="t3-section-title">Без изоляции vs с текущей конфигурацией</h2>',
    unsafe_allow_html=True,
)


def _kpi_block(eyebrow: str, r):
    return f"""
    <div class="t3-kpi">
      <div class="item">
        <p class="label">{eyebrow}</p>
        <p class="value-mono">&nbsp;</p>
      </div>
      <div class="item">
        <p class="label">T<sub>max</sub></p>
        <p class="value-mono">{r.T_max_C:.1f} <span style="font-size:0.7em;color:#6b6b6b">°C</span></p>
      </div>
      <div class="item">
        <p class="label">T<sub>avg</sub></p>
        <p class="value-mono">{r.T_avg_C:.1f} <span style="font-size:0.7em;color:#6b6b6b">°C</span></p>
      </div>
      <div class="item">
        <p class="label">Q<sub>лиц</sub></p>
        <p class="value-mono">{r.Q_front_W:.1f} <span style="font-size:0.7em;color:#6b6b6b">Вт ({100*r.Q_front_W/r.Q_input_W:.1f}%)</span></p>
      </div>
      <div class="item">
        <p class="label">Q<sub>торцов</sub></p>
        <p class="value-mono">{r.Q_sides_W + r.Q_top_W + r.Q_bottom_W:.1f} <span style="font-size:0.7em;color:#6b6b6b">Вт ({100*(r.Q_sides_W+r.Q_top_W+r.Q_bottom_W)/r.Q_input_W:.1f}%)</span></p>
      </div>
    </div>
    """


st.markdown(_kpi_block('Без изоляции (baseline)', r_base),
            unsafe_allow_html=True)
st.markdown(_kpi_block(
    f'С изоляцией ({iso_material}, {iso_thickness_mm:.0f} мм)', r_iso),
    unsafe_allow_html=True)

Q_in = r_base.Q_input_W


def _pct(Q):
    return 100.0 * Q / Q_in


rows_data = [
    ('Лицевая грань', 'нет (вся в воздух)',
     r_base.Q_front_W, r_iso.Q_front_W),
    ('Боковые торцы (2 шт.)',
     ('изолированы — ' + iso_material if iso_sides else 'голый металл'),
     r_base.Q_sides_W, r_iso.Q_sides_W),
    ('Верхний торец',
     ('изолирован — ' + iso_material if iso_top else 'голый металл'),
     r_base.Q_top_W, r_iso.Q_top_W),
    ('Днище',
     ('изолировано — ' + iso_material if iso_bottom else 'голый металл'),
     r_base.Q_bottom_W, r_iso.Q_bottom_W),
]

table_html = [
    '<table class="t3-cmp">'
    '<thead><tr>'
    '<th>Грань / торец</th>'
    '<th>Состояние</th>'
    '<th class="num">Без изо, Вт</th>'
    '<th class="num">Без изо, %</th>'
    '<th class="num">С изо, Вт</th>'
    '<th class="num">С изо, %</th>'
    '<th class="num">Δ, Вт</th>'
    '</tr></thead><tbody>'
]
for name, status, Q_b, Q_i in rows_data:
    delta = Q_i - Q_b
    delta_cls = 'delta-neg' if delta > 0 else 'delta-pos'
    delta_str = ('+' if delta > 0 else '') + f'{delta:.1f}'
    table_html.append(
        f'<tr class="path">'
        f'<td><b>{name}</b></td>'
        f'<td>{status}</td>'
        f'<td class="num">{Q_b:.1f}</td>'
        f'<td class="num">{_pct(Q_b):.1f}</td>'
        f'<td class="num">{Q_i:.1f}</td>'
        f'<td class="num">{_pct(Q_i):.1f}</td>'
        f'<td class="num {delta_cls}">{delta_str}</td>'
        f'</tr>'
    )
table_html.append(
    f'<tr class="totals">'
    f'<td colspan="2">Σ потерь (= Q<sub>вх</sub> = q<sub>w</sub>·b·L)</td>'
    f'<td class="num">{r_base.Q_output_W:.1f}</td>'
    f'<td class="num">{_pct(r_base.Q_output_W):.1f}</td>'
    f'<td class="num">{r_iso.Q_output_W:.1f}</td>'
    f'<td class="num">{_pct(r_iso.Q_output_W):.1f}</td>'
    f'<td class="num">—</td>'
    f'</tr></tbody></table>'
)
st.markdown(''.join(table_html), unsafe_allow_html=True)

t_max_delta = r_iso.T_max_C - r_base.T_max_C
st.caption(
    f'Зелёный — потери уменьшились, красный — увеличились. '
    'Сумма потерь равна Q<sub>вх</sub> в обоих прогонах (закон сохранения). '
    'Уменьшение потерь через торцы перенаправляет ту же мощность по '
    f'оставшимся каналам — главным образом по лицевой грани. Это поднимает '
    f'температуру плиты: <b>T<sub>max</sub> {r_base.T_max_C:.1f} → '
    f'{r_iso.T_max_C:.1f} °C (Δ = {t_max_delta:+.1f} К)</b>.',
    unsafe_allow_html=True,
)


# ── Sweep по толщине ─────────────────────────────────────────────────────

st.markdown('<hr class="t3-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t3-section-eyebrow">Экспорт данных</p>'
    '<h2 class="t3-section-title">Поле температуры пластины по высоте (CSV)</h2>',
    unsafe_allow_html=True,
)


def _temperature_field_df(r_no_iso, r_with_iso):
    """Узловой профиль T(z) обоих прогонов (с изоляцией и baseline) в одном
    DataFrame. Сетка узлов у прогонов одинаковая (один L и N_nodes),
    поэтому координату z берём из прогона с изоляцией."""
    t_inf = r_with_iso.t_fluid_C
    return pd.DataFrame({
        'z, мм':                   [n.z_m * 1000.0 for n in r_with_iso.nodes],
        'z, м':                    [n.z_m for n in r_with_iso.nodes],
        'T_пов. с изоляцией, °C':  [n.T_s_C for n in r_with_iso.nodes],
        'T_пов. без изоляции, °C': [n.T_s_C for n in r_no_iso.nodes],
        'ΔT = T−t_∞, К':           [n.T_s_C - t_inf for n in r_with_iso.nodes],
        'Режим конвекции':         [n.regime for n in r_with_iso.nodes],
        'α конв, Вт/(м²·К)':       [n.alpha_front for n in r_with_iso.nodes],
        'q конв, Вт/м²':           [n.q_conv_per_m2 for n in r_with_iso.nodes],
        'q рад, Вт/м²':            [n.q_rad_per_m2 for n in r_with_iso.nodes],
        'Ra_x':                    [n.Ra_x for n in r_with_iso.nodes],
    })


_field_df = _temperature_field_df(r_base, r_iso)
# Разделитель «;» + десятичная запятая + BOM — чтобы Excel (RU) открыл сразу.
_field_csv = _field_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
_iso_tag = (
    f'iso{int(iso_thickness_mm)}mm' if (iso_sides or iso_top or iso_bottom)
    else 'noiso'
)
_field_name = f'task3_T_field_{r_iso.key}_L{int(L_mm)}mm_{_iso_tag}.csv'

ec1, ec2 = st.columns([1, 2])
with ec1:
    st.download_button(
        '📥 Скачать T(z) в CSV',
        data=_field_csv,
        file_name=_field_name,
        mime='text/csv',
        use_container_width=True,
        help='Профиль температуры поверхности по высоте (с изоляцией и без) '
             'плюс α(z), плотности потоков, Ra_x и режим конвекции '
             'в каждом узле сетки.',
    )
with ec2:
    st.caption(
        f'Выгрузка содержит <b>{len(_field_df)}</b> узлов по высоте '
        f'(0…{int(L_mm)} мм). Колонки: T поверхности для текущей конфигурации '
        'с изоляцией и для baseline без неё, перегрев ΔT, режим конвекции, '
        'α(z), плотности конвективного и радиационного потоков и число '
        'Ra_x. Формат: разделитель «;», десятичная запятая, кодировка '
        'UTF-8 (Excel открывает напрямую).',
        unsafe_allow_html=True,
    )


# ── Sweep по толщине ─────────────────────────────────────────────────────

st.markdown('<hr class="t3-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t3-section-eyebrow">Раздел 2 · Чувствительность по толщине</p>'
    f'<h2 class="t3-section-title">'
    f'Как меняются доли потерь с ростом толщины ({iso_material})</h2>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _sweep(key, N_total_W, b_mm, L_mm, delta_mm, lambda_metal, material_name,
           t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g, iso_lambda,
           iso_sides, iso_top, iso_bottom, thicknesses):
    runs = []
    for tk in thicknesses:
        st_mm = tk if iso_sides else 0.0
        tt_mm = tk if iso_top else 0.0
        bt_mm = tk if iso_bottom else 0.0
        r = solve_task2(
            key=key, N_total_W=N_total_W,
            b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
            lambda_metal=lambda_metal, material_name=material_name,
            t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps_surface,
            N_nodes=int(N_nodes), x_min_mm=x_min_mm, g=g,
            sides_iso_thickness_mm=st_mm, sides_iso_lambda=iso_lambda,
            top_iso_thickness_mm=tt_mm, top_iso_lambda=iso_lambda,
            bottom_iso_thickness_mm=bt_mm, bottom_iso_lambda=iso_lambda,
        )
        runs.append(r)
    return runs


thicknesses = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0)
with st.spinner('Прогон солвера для серии толщин…'):
    sweep_runs = _sweep(
        methodology, N_total_W, b_mm, L_mm, delta_mm,
        lambda_metal, material_name,
        t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g, iso_lambda,
        iso_sides, iso_top, iso_bottom, thicknesses,
    )

front_pct = [100*r.Q_front_W/r.Q_input_W for r in sweep_runs]
sides_pct = [100*r.Q_sides_W/r.Q_input_W for r in sweep_runs]
top_pct = [100*r.Q_top_W/r.Q_input_W for r in sweep_runs]
bot_pct = [100*r.Q_bottom_W/r.Q_input_W for r in sweep_runs]
t_max_arr = [r.T_max_C for r in sweep_runs]

FONT_BODY = '"IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, monospace'

fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.10,
    subplot_titles=(
        '<b>Доли потерь по граням, %</b>',
        '<b>Максимальная температура плиты, °C</b>',
    ),
)

for label_, ydata, color in [
    ('Лицевая грань', front_pct, '#3b6da3'),
    ('Боковые торцы (2 шт.)', sides_pct, '#6fa1d1'),
    ('Верхний торец', top_pct, '#2c8d52'),
    ('Днище', bot_pct, '#7d6a8c'),
]:
    fig.add_trace(go.Scatter(
        x=thicknesses, y=ydata, mode='lines+markers',
        name=label_,
        line=dict(color=color, width=2.4),
        marker=dict(size=6),
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=thicknesses, y=t_max_arr, mode='lines+markers',
    line=dict(color='#B23A2C', width=2.4),
    marker=dict(size=7, color='#B23A2C',
                line=dict(width=1, color='white')),
    showlegend=False,
    hovertemplate='δ_изо = %{x:.0f} мм<br>T_max = %{y:.1f} °C<extra></extra>',
), row=1, col=2)

# Текущая толщина — вертикальная отметка.
for col in (1, 2):
    fig.add_shape(
        type='line', xref='x' if col == 1 else 'x2', yref='paper',
        x0=iso_thickness_mm, x1=iso_thickness_mm, y0=0, y1=1,
        line=dict(color='rgba(40,40,40,0.4)', width=1, dash='dot'),
    )

fig.update_xaxes(
    title=dict(text='Толщина изоляции δ_изо, мм',
               font=dict(size=11, family=FONT_BODY)),
    tickfont=dict(family=FONT_MONO, size=10),
    gridcolor='rgba(220,220,220,0.5)',
)
fig.update_yaxes(
    title=dict(text='Доля от Q_вх, %',
               font=dict(size=11, family=FONT_BODY)),
    tickfont=dict(family=FONT_MONO, size=10),
    gridcolor='rgba(220,220,220,0.5)',
    row=1, col=1,
)
fig.update_yaxes(
    title=dict(text='T_max, °C',
               font=dict(size=11, family=FONT_BODY)),
    tickfont=dict(family=FONT_MONO, size=10),
    gridcolor='rgba(220,220,220,0.5)',
    row=1, col=2,
)
fig.update_layout(
    height=460,
    margin=dict(t=80, b=60, l=70, r=30),
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family=FONT_BODY),
    showlegend=True,
    legend=dict(
        orientation='h', yanchor='bottom', y=1.10,
        xanchor='left', x=0.0,
        font=dict(size=10.5, family=FONT_BODY),
        bgcolor='rgba(255,255,255,0)', borderwidth=0,
    ),
)
for ann in fig.layout.annotations:
    ann.font.family = FONT_BODY
    ann.font.size = 11.5

st.plotly_chart(fig, use_container_width=True,
                config={'displayModeBar': False})
st.caption(
    f'Пунктирная вертикаль — текущее значение толщины '
    f'<span style="font-family:IBM Plex Mono">{iso_thickness_mm:.0f}</span> мм. '
    'Чем толще изоляция — тем меньше доля торцов и тем больше доля лицевой '
    'грани. Закон убывающей отдачи: эффект от 0→20 мм значительно больше, '
    'чем от 50→100 мм.',
    unsafe_allow_html=True,
)


# ── Сравнение всех материалов ───────────────────────────────────────────

st.markdown('<hr class="t3-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t3-section-eyebrow">Раздел 3 · Сравнение материалов</p>'
    f'<h2 class="t3-section-title">'
    f'Все материалы при δ_изо = {iso_thickness_mm:.0f} мм</h2>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _materials_sweep(key, N_total_W, b_mm, L_mm, delta_mm, lambda_metal,
                     material_name, t_fluid_C, P_Pa, eps_surface, N_nodes,
                     x_min_mm, g, thickness_mm, iso_sides, iso_top, iso_bottom):
    out = []
    for name, props in INSULATION_MATERIALS.items():
        lam = props['lambda']
        st_mm = thickness_mm if iso_sides else 0.0
        tt_mm = thickness_mm if iso_top else 0.0
        bt_mm = thickness_mm if iso_bottom else 0.0
        r = solve_task2(
            key=key, N_total_W=N_total_W,
            b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
            lambda_metal=lambda_metal, material_name=material_name,
            t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps_surface,
            N_nodes=int(N_nodes), x_min_mm=x_min_mm, g=g,
            sides_iso_thickness_mm=st_mm, sides_iso_lambda=lam,
            top_iso_thickness_mm=tt_mm, top_iso_lambda=lam,
            bottom_iso_thickness_mm=bt_mm, bottom_iso_lambda=lam,
        )
        out.append((name, lam, r))
    return out


with st.spinner('Прогон солвера для всех материалов…'):
    mat_runs = _materials_sweep(
        methodology, N_total_W, b_mm, L_mm, delta_mm, lambda_metal,
        material_name, t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g,
        iso_thickness_mm, iso_sides, iso_top, iso_bottom,
    )

mat_rows = []
for name, lam, r in mat_runs:
    Q_torch = r.Q_sides_W + r.Q_top_W + r.Q_bottom_W
    pct_torch = 100 * Q_torch / r.Q_input_W
    is_current = (name == iso_material)
    marker = ' ← текущий' if is_current else ''
    mat_rows.append({
        'Материал': name + marker,
        'λ, Вт/(м·К)': f'{lam:.3f}',
        'T_max, °C': f'{r.T_max_C:.1f}',
        'Q_лиц, Вт': f'{r.Q_front_W:.1f}',
        'Q_лиц, %': f'{100*r.Q_front_W/r.Q_input_W:.1f}',
        'Q_торцов, Вт': f'{Q_torch:.2f}',
        'Q_торцов, %': f'{pct_torch:.2f}',
    })
st.dataframe(pd.DataFrame(mat_rows), hide_index=True,
             use_container_width=True)
st.caption(
    f'Прогон всех материалов из справочника при той же толщине '
    f'<span style="font-family:IBM Plex Mono">{iso_thickness_mm:.0f}</span> мм '
    'и тех же выбранных торцах. Чем меньше λ_изо — тем меньше потери торцов '
    'и тем выше T<sub>max</sub>. Аэрогель — лидер, но дорог; '
    'минеральная вата даёт практичный компромисс по цене/эффективности.',
    unsafe_allow_html=True,
)
