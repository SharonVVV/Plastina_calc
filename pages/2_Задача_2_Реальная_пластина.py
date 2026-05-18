"""
Задача №2. Реальная вертикальная пластина с толщиной δ. Нагрев q_w = const
с тыльной стороны (силиконовые маты); теплоотвод со всех остальных граней.

Шаг 1 — распределение температуры T(z) по высоте.
Шаг 2 — баланс тепла: сколько уходит через лицевую грань, боковые торцы,
        верхний торец, днище (в Вт и %).
"""

import streamlit as st
import pandas as pd

from config import DEFAULTS_TASK2, DEFAULTS_ENV
from correlations_meta import META, ORDER, label_for
from task2_solver import solve_task2, MATERIALS
from plotting_task2 import (
    plot_temperature_profile, plot_heat_balance, plot_alpha_ra_profile,
)


st.set_page_config(
    page_title='Задача №2 — Реальная пластина',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# Тот же css/шрифты, что и на странице Задачи 1 (IBM Plex + защита Material-иконок).
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
      .t2-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 11px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.35rem 0;
      }
      .t2-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.95rem; line-height: 1.15;
        color: #1a1a1a; margin: 0 0 0.35rem 0;
      }
      .t2-lede {
        max-width: 60rem; color: #4b4b4b; font-size: 0.96rem;
        line-height: 1.55; margin: 0 0 1.5rem 0;
      }
      .t2-rule {border:0; border-top:1px solid #d8d8d8; margin:1.6rem 0 1.2rem 0;}
      .t2-section-eyebrow {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.14em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 0.25rem 0;
      }
      .t2-section-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-weight: 600; font-size: 1.25rem; line-height: 1.2;
        color: #1a1a1a; margin: 0 0 0.9rem 0;
      }
      .t2-kpi {
        display: flex; gap: 1.4rem; flex-wrap: wrap; margin: 0 0 1.0rem 0;
      }
      .t2-kpi .item {min-width: 110px;}
      .t2-kpi .label {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 10.5px; letter-spacing: 0.10em;
        text-transform: uppercase; color: #6b6b6b;
        margin: 0 0 2px 0;
      }
      .t2-kpi .value {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-weight: 600; font-size: 1.2rem; color: #1a1a1a;
      }
      .t2-kpi .value-mono {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-weight: 500; font-size: 1.2rem; color: #1a1a1a;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="t2-eyebrow">Задача №2</p>
    <h1 class="t2-title">Реальная пластина с толщиной — баланс тепла</h1>
    <p class="t2-lede">
      Вертикальная пластина <b>b × L × δ</b>: нагрев q<sub>w</sub> = const
      с тыльной грани (силиконовые маты), теплоотвод свободной конвекцией
      и излучением со всех остальных граней. Шаг 1 — распределение
      температуры T(z) по высоте. Шаг 2 — баланс: какая доля подведённого
      тепла уходит через лицевую грань, боковые торцы, верхний и нижний
      торцы (в Вт и %).
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Основные параметры ────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    b_mm = st.number_input(
        'Ширина b, мм', min_value=10.0, max_value=1000.0,
        value=float(DEFAULTS_TASK2['b_mm']), step=10.0,
    )
with c2:
    L_mm = st.number_input(
        'Высота L, мм', min_value=100.0, max_value=10000.0,
        value=float(DEFAULTS_TASK2['L_mm']), step=50.0,
    )
with c3:
    delta_mm = st.number_input(
        'Толщина δ, мм', min_value=0.5, max_value=200.0,
        value=float(DEFAULTS_TASK2['delta_mm']), step=0.5,
    )
with c4:
    N_total_W = st.number_input(
        'Суммарная мощность матов N, Вт',
        min_value=10.0, max_value=20000.0,
        value=float(DEFAULTS_TASK2['N_total_W']), step=50.0,
        help='Полная электрическая мощность всех матов; равномерно подводится '
             'к тыльной грани b×L. q_w = N/(b·L).',
    )

c5, c6 = st.columns(2)
with c5:
    material_name = st.selectbox(
        'Материал пластины',
        options=list(MATERIALS.keys()),
        index=list(MATERIALS.keys()).index(DEFAULTS_TASK2['material']),
        help='λ_металла — берётся из встроенной таблицы (по умолчанию Д16-Т).',
    )
with c6:
    methodology = st.selectbox(
        'Методика свободной конвекции (вертикальные грани)',
        options=ORDER,
        index=ORDER.index(DEFAULTS_TASK2['methodology']),
        format_func=label_for,
        help='Локальная UHF-корреляция Nu_x для лицевой грани и боковых торцов. '
             'Верхний торец — формула Леонтьева 1979 (7.30) для нагретой '
             'горизонтальной поверхности лицом вверх. Днище — только радиация.',
    )

mat = MATERIALS[material_name]


# ── Доп. параметры ────────────────────────────────────────────────────────

with st.expander('Доп. параметры (среда, излучение, сетка, λ_металла)'):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown('**Среда**')
        t_fluid_C = st.number_input(
            't_∞, °C', value=float(DEFAULTS_ENV['t_fluid_C']),
            min_value=-50.0, max_value=200.0, step=1.0,
        )
        P_Pa = st.number_input(
            'Давление P, Па', value=float(DEFAULTS_ENV['P_Pa']),
            min_value=50000.0, max_value=200000.0, step=100.0,
        )
        g = float(DEFAULTS_ENV['g'])
    with cc2:
        st.markdown('**Излучение**')
        eps_surface = st.number_input(
            'ε поверхности', value=float(DEFAULTS_TASK2['eps_surface']),
            min_value=0.03, max_value=0.99, step=0.01,
            help='Степень черноты — применяется ко всем излучающим граням. '
                 'Для голого Д16-Т (полированный) ε ≈ 0,04…0,09; '
                 'для оксидированного — 0,2…0,3; '
                 'для матовой эмали / краски — 0,9…0,95.',
        )
    with cc3:
        st.markdown('**Расчётная сетка и материал**')
        N_nodes = st.number_input(
            'Узлов по высоте', value=int(DEFAULTS_TASK2['N_nodes']),
            min_value=21, max_value=501, step=10,
        )
        lambda_metal = st.number_input(
            'λ металла, Вт/(м·К)', value=float(mat['lambda']),
            min_value=1.0, max_value=600.0, step=5.0,
            help=f'По умолчанию для «{material_name}»: {mat["note"]}.',
        )
        x_min_mm = st.number_input(
            'x_min (UHF), мм', value=float(DEFAULTS_TASK2['x_min_mm']),
            min_value=1.0, max_value=50.0, step=1.0,
            help='Минимальный «эффективный» x для UHF-корреляций — '
                 'избегает сингулярности α = Nu·λ/x при x → 0 на нижней кромке.',
        )
        axial_conduction = st.checkbox(
            'Учитывать продольную теплопроводность в металле',
            value=True,
            help='ON (по умолчанию): включён член λ_м·A_кр·d²T/dz² '
                 '— тепло перетекает по высоте в самом металле, профиль '
                 'T(z) сглаживается. Для алюминия Д16 сглаживание сильное. '
                 'OFF: каждое сечение решается как локальный баланс '
                 '(без переноса по металлу) — получается классический '
                 '«хамповидный» профиль с пиком в районе перехода '
                 'lam→trans, как в Задаче 1 или legacy-UI.',
        )


# ── Расчёт ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run(key, N_total_W, b_mm, L_mm, delta_mm, lambda_metal, material_name,
         t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g, axial_conduction):
    return solve_task2(
        key=key,
        N_total_W=N_total_W,
        b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
        lambda_metal=lambda_metal, material_name=material_name,
        t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps_surface,
        N_nodes=int(N_nodes), x_min_mm=x_min_mm, g=g,
        axial_conduction=axial_conduction,
    )


with st.spinner('Picard-итерация фин-уравнения…'):
    result = _run(
        methodology, N_total_W, b_mm, L_mm, delta_mm,
        lambda_metal, material_name,
        t_fluid_C, P_Pa, eps_surface, N_nodes, x_min_mm, g,
        axial_conduction,
    )


# ── KPI ───────────────────────────────────────────────────────────────────

q_w = result.q_w
st.markdown(
    f"""
    <div class="t2-kpi">
      <div class="item">
        <p class="label">q<sub>w</sub></p>
        <p class="value-mono">{q_w:.0f} <span style="font-size:0.7em;color:#6b6b6b">Вт/м²</span></p>
      </div>
      <div class="item">
        <p class="label">T<sub>min</sub></p>
        <p class="value-mono">{result.T_min_C:.1f} <span style="font-size:0.7em;color:#6b6b6b">°C</span></p>
      </div>
      <div class="item">
        <p class="label">T<sub>avg</sub></p>
        <p class="value-mono">{result.T_avg_C:.1f} <span style="font-size:0.7em;color:#6b6b6b">°C</span></p>
      </div>
      <div class="item">
        <p class="label">T<sub>max</sub></p>
        <p class="value-mono">{result.T_max_C:.1f} <span style="font-size:0.7em;color:#6b6b6b">°C</span></p>
      </div>
      <div class="item">
        <p class="label">Q<sub>вход</sub></p>
        <p class="value-mono">{result.Q_input_W:.1f} <span style="font-size:0.7em;color:#6b6b6b">Вт</span></p>
      </div>
      <div class="item">
        <p class="label">Невязка</p>
        <p class="value-mono">{result.residual_pct:+.2f}<span style="font-size:0.7em;color:#6b6b6b"> %</span></p>
      </div>
      <div class="item">
        <p class="label">Итер.</p>
        <p class="value-mono">{result.iterations}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Шаг 1: T(z) ───────────────────────────────────────────────────────────

st.markdown('<hr class="t2-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t2-section-eyebrow">Шаг 1 · Распределение температуры</p>'
    f'<h2 class="t2-section-title">T(z) — профиль температуры по высоте, '
    f'{result.meta.name_ru}, {material_name}</h2>',
    unsafe_allow_html=True,
)
st.plotly_chart(plot_temperature_profile(result),
                use_container_width=True,
                config={'displayModeBar': False})
if axial_conduction:
    cond_note = (
        'продольная теплопроводность по металлу <b>включена</b> '
        f'(λ_м = {result.lambda_metal:.0f} Вт/(м·К)): тепло перетекает '
        'из горячих участков в холодные, профиль сглажен.'
    )
else:
    cond_note = (
        'продольная теплопроводность по металлу <b>отключена</b>: '
        'каждое сечение решается как локальный баланс. Видна классическая '
        'форма «хампа» — пик в районе перехода lam→trans, где коэффициент '
        'теплоотдачи α(z) минимален.'
    )
st.caption(
    'Полупрозрачные горизонтальные полосы фона — режим свободной конвекции '
    'на лицевой грани в данной координате z (по выбранной UHF-методике). '
    'Кривая — рассчитанная температура T<sub>s</sub>(z). '
    + cond_note, unsafe_allow_html=True,
)


# ── α(z) и Ra(z) — диагностика режима ────────────────────────────────────

st.markdown('<p class="t2-section-eyebrow" style="margin-top:1.2rem">'
            'Диагностика режима</p>',
            unsafe_allow_html=True)
st.plotly_chart(plot_alpha_ra_profile(result),
                use_container_width=True,
                config={'displayModeBar': False})
st.caption(
    '<b>α(z)</b> — местный коэффициент теплоотдачи свободной конвекцией '
    'на лицевой грани (Вт/(м²·К)). У самой нижней кромки α максимален '
    '(пограничный слой только формируется, тонкий), затем падает по мере '
    'роста ПС; при переходе lam→trans/turb α иногда заметно меняется. '
    '<b>Ra(z)</b> (логарифмическая шкала) — число Рэлея через ΔT = T<sub>s</sub>(z) − T<sub>∞</sub>, '
    'основной критерий, по которому методика определяет режим конвекции '
    f'({result.meta.name_ru}). '
    'Если интересно «классическое» число Рейнольдса для свободной '
    'конвекции — оно определяется как Re<sub>z</sub> = √Gr<sub>z</sub>, '
    'т.е. эквивалентно Gr (или √(Ra/Pr) с точностью до Pr ≈ const для воздуха).',
    unsafe_allow_html=True,
)


# ── Шаг 2: Баланс тепла ───────────────────────────────────────────────────

st.markdown('<hr class="t2-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t2-section-eyebrow">Шаг 2 · Баланс тепла</p>'
    '<h2 class="t2-section-title">'
    'Куда уходит подведённое q<sub>w</sub>·b·L</h2>',
    unsafe_allow_html=True,
)
st.plotly_chart(plot_heat_balance(result),
                use_container_width=True,
                config={'displayModeBar': False})

# ── Двухуровневая таблица: первичная разбивка по граням, вторичная по механизму
Q_in = result.Q_input_W
surfaces = [
    {
        'name': 'Лицевая грань',
        'area_m2': result.b_mm * result.L_mm * 1e-6,
        'conv_label': 'свободная конвекция (UHF, ' + result.meta.name_short + ')',
        'conv_Q': result.Q_front_conv_W,
        'rad_Q': result.Q_front_rad_W,
    },
    {
        'name': 'Боковые торцы (2 шт.)',
        'area_m2': 2 * result.delta_mm * result.L_mm * 1e-6,
        'conv_label': 'свободная конвекция (UHF, ' + result.meta.name_short + ')',
        'conv_Q': result.Q_sides_conv_W,
        'rad_Q': result.Q_sides_rad_W,
    },
    {
        'name': 'Верхний торец',
        'area_m2': result.b_mm * result.delta_mm * 1e-6,
        'conv_label': 'свободная конвекция (Леонтьев 1979, ↑)',
        'conv_Q': result.Q_top_conv_W,
        'rad_Q': result.Q_top_rad_W,
    },
    {
        'name': 'Днище',
        'area_m2': result.b_mm * result.delta_mm * 1e-6,
        'conv_label': '— (только излучение, по ТЗ)',
        'conv_Q': 0.0,
        'rad_Q': result.Q_bottom_rad_W,
    },
]

# HTML-таблица с двойной шапкой и под-строками. st.dataframe с MultiIndex
# рендерится не очень аккуратно; делаем явно через HTML.
def _fmt_W(x: float) -> str:
    return f'{x:.2f}'

def _fmt_pct(x: float) -> str:
    return f'{100 * x / Q_in:.2f}'

table_html = ['''
<style>
.t2-balance {
  width: 100%; border-collapse: collapse;
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  font-size: 0.92rem; margin: 0.5rem 0 0.2rem 0;
}
.t2-balance th, .t2-balance td {
  padding: 6px 10px; border-bottom: 1px solid #e6e6e6; text-align: left;
}
.t2-balance th {
  background: #f5f5f3; font-weight: 600; color: #1a1a1a;
  border-bottom: 2px solid #c8c8c8;
}
.t2-balance td.num,
.t2-balance th.num { text-align: right;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-feature-settings: "tnum"; }
.t2-balance tr.surface {
  background: #fafaf7;
}
.t2-balance tr.surface td {
  font-weight: 600; color: #1a1a1a; border-top: 2px solid #c8c8c8;
}
.t2-balance tr.mechanism td { color: #4b4b4b; padding-left: 24px; }
.t2-balance tr.mechanism td.first { padding-left: 28px; }
.t2-balance tfoot tr td {
  font-weight: 700; background: #efeee9; border-top: 2px solid #888;
  border-bottom: none;
}
</style>
<table class="t2-balance">
  <thead>
    <tr>
      <th>Грань / механизм</th>
      <th class="num">Q, Вт</th>
      <th class="num">% от Q<sub>вх</sub></th>
      <th class="num">Площадь, м²</th>
    </tr>
  </thead>
  <tbody>
''']
for s in surfaces:
    Q_total = s['conv_Q'] + s['rad_Q']
    table_html.append(
        f'<tr class="surface">'
        f'<td>{s["name"]}</td>'
        f'<td class="num">{_fmt_W(Q_total)}</td>'
        f'<td class="num"><b>{_fmt_pct(Q_total)}</b></td>'
        f'<td class="num">{s["area_m2"]:.4f}</td>'
        f'</tr>'
    )
    # Sub-row: convection (if applicable)
    if s['conv_Q'] > 0 or 'только излучение' not in s['conv_label']:
        table_html.append(
            f'<tr class="mechanism">'
            f'<td class="first">— {s["conv_label"]}</td>'
            f'<td class="num">{_fmt_W(s["conv_Q"])}</td>'
            f'<td class="num">{_fmt_pct(s["conv_Q"])}</td>'
            f'<td class="num">—</td>'
            f'</tr>'
        )
    # Sub-row: radiation
    table_html.append(
        f'<tr class="mechanism">'
        f'<td class="first">— излучение, ε·σ·(T<sub>s</sub>⁴ − T<sub>∞</sub>⁴)</td>'
        f'<td class="num">{_fmt_W(s["rad_Q"])}</td>'
        f'<td class="num">{_fmt_pct(s["rad_Q"])}</td>'
        f'<td class="num">—</td>'
        f'</tr>'
    )

# Footer with totals
Q_out = result.Q_output_W
table_html.append(
    f'</tbody><tfoot>'
    f'<tr><td>Σ потерь (всего)</td>'
    f'<td class="num">{_fmt_W(Q_out)}</td>'
    f'<td class="num">{_fmt_pct(Q_out)}</td>'
    f'<td class="num">—</td>'
    f'</tr>'
    f'<tr><td>Q<sub>вх</sub> = q<sub>w</sub>·b·L</td>'
    f'<td class="num">{_fmt_W(Q_in)}</td>'
    f'<td class="num">100,00</td>'
    f'<td class="num">{result.b_mm * result.L_mm * 1e-6:.4f}</td>'
    f'</tr>'
    f'</tfoot></table>'
)
st.markdown(''.join(table_html), unsafe_allow_html=True)

st.caption(
    f'Невязка баланса = {result.residual_W:+.3f} Вт '
    f'({result.residual_pct:+.4f} %) — контроль сходимости итерации.',
    unsafe_allow_html=True,
)


# ── Чувствительность к степени черноты ε ─────────────────────────────────

st.markdown('<hr class="t2-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t2-section-eyebrow">Альтернатива · Чувствительность к ε</p>'
    '<h2 class="t2-section-title">Доля радиации в балансе сильно зависит от ε</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    'Текущее значение ε = '
    f'<b><span style="font-family:IBM Plex Mono">{eps_surface:.2f}</span></b> '
    f'(C<sub>пр</sub> = ε·σ·100⁴ = '
    f'<b><span style="font-family:IBM Plex Mono">{eps_surface * 5.670374419e-8 * 1e8:.2f}</span></b> '
    'Вт/(м²·К⁴)) — выбрано в expander «Доп. параметры». '
    'Эта величина критически влияет на долю радиации в балансе. '
    'Ниже — повторный прогон солвера при нескольких эталонных значениях ε.',
    unsafe_allow_html=True,
)

with st.expander('Развернуть сравнение и справку по типичным ε'):
    st.markdown('''
**Эквивалентность форм записи** (то же самое, разные обозначения):

- «Стандартная»: $q_\\text{рад} = \\varepsilon\\sigma\\,(T_s^4 - T_\\infty^4)$,
  где $\\sigma = 5{,}670374\\cdot 10^{-8}$ Вт/(м²·К⁴).
- «Через C_пр» (как у Керимова 1992): $q_\\text{рад} = C_\\text{пр}\\,
  [(T_s/100)^4 - (T_\\infty/100)^4]$, где $C_\\text{пр} = \\varepsilon\\sigma\\cdot 10^8$.
  Численно: $C_\\text{пр} [Вт/(м^2\\cdot K^4)] \\approx 5{,}67\\cdot\\varepsilon$.

В работе Керимова 1992 эмпирически принято **C_пр = 2 Вт/(м²·К⁴)** для
нержавеющей стали — это соответствует ε ≈ **0,353** (взаимный обмен
плита-комната).

**Типичные значения ε** (по Holman 2010, Table A-10, стр. 663):

| Поверхность | ε | C_пр, Вт/(м²·К⁴) |
|---|---|---|
| Алюминий полированный, 98,3% чист. | 0,04–0,06 | 0,23–0,34 |
| Алюминий коммерческий лист (Д16) | **0,09** | **0,51** |
| Алюминий сильно оксидированный | 0,20–0,31 | 1,13–1,76 |
| Нержавеющая сталь, окисленная (Керимов 1992) | **0,353** | **2,0** |
| Анодированный алюминий | 0,7–0,9 | 4,0–5,1 |
| Матовое чёрное покрытие / эмаль | 0,90–0,95 | 5,1–5,4 |
    ''')

    # ── Прогон солвера при нескольких ε ───────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _sweep_eps(eps_values, key, N_total_W, b_mm, L_mm, delta_mm,
                   lambda_metal, material_name, t_fluid_C, P_Pa,
                   N_nodes, x_min_mm, g, axial_conduction):
        out = []
        for eps in eps_values:
            r = solve_task2(
                key=key, N_total_W=N_total_W,
                b_mm=b_mm, L_mm=L_mm, delta_mm=delta_mm,
                lambda_metal=lambda_metal, material_name=material_name,
                t_fluid_C=t_fluid_C, P_Pa=P_Pa, eps_surface=eps,
                N_nodes=int(N_nodes), x_min_mm=x_min_mm, g=g,
                axial_conduction=axial_conduction,
            )
            out.append((eps, r))
        return out

    eps_grid = (0.07, 0.10, 0.20, 0.35, 0.50, 0.95)
    with st.spinner('Прогон солвера при нескольких ε…'):
        sweep = _sweep_eps(
            eps_grid, methodology, N_total_W, b_mm, L_mm, delta_mm,
            lambda_metal, material_name, t_fluid_C, P_Pa,
            N_nodes, x_min_mm, g, axial_conduction,
        )

    # Таблица сравнения.
    SIGMA = 5.670374419e-8
    rows_html = []
    for eps, r in sweep:
        C_pr = eps * SIGMA * 1e8
        Q_conv_total = r.Q_front_conv_W + r.Q_sides_conv_W + r.Q_top_conv_W
        Q_rad_total = r.Q_front_rad_W + r.Q_sides_rad_W + r.Q_top_rad_W + r.Q_bottom_rad_W
        pct_rad = 100 * Q_rad_total / r.Q_input_W
        is_current = abs(eps - eps_surface) < 0.025
        marker = ' ← текущее' if is_current else ''
        rows_html.append(
            f'<tr class="{"surface" if is_current else "mechanism"}">'
            f'<td class="first"><b>{eps:.2f}</b>{marker}</td>'
            f'<td class="num">{C_pr:.2f}</td>'
            f'<td class="num">{r.T_max_C:.1f}</td>'
            f'<td class="num">{r.T_avg_C:.1f}</td>'
            f'<td class="num">{Q_conv_total:.1f}</td>'
            f'<td class="num">{Q_rad_total:.1f}</td>'
            f'<td class="num"><b>{pct_rad:.1f}</b></td>'
            f'</tr>'
        )
    sweep_html = '''
<table class="t2-balance" style="margin-top:0.5rem">
  <thead>
    <tr>
      <th>ε</th>
      <th class="num">C<sub>пр</sub>, Вт/(м²·К⁴)</th>
      <th class="num">T<sub>max</sub>, °C</th>
      <th class="num">T<sub>avg</sub>, °C</th>
      <th class="num">Σ Q<sub>конв</sub>, Вт</th>
      <th class="num">Σ Q<sub>рад</sub>, Вт</th>
      <th class="num">Доля рад., %</th>
    </tr>
  </thead>
  <tbody>
''' + ''.join(rows_html) + '''
  </tbody>
</table>
'''
    st.markdown(sweep_html, unsafe_allow_html=True)

    st.markdown('''
**Физический смысл:**

- При **низком ε** радиация слабая → T_s растёт, чтобы конвекция могла
  отвести то же q_w. Доля радиации в балансе мала, конвекции — велика.
- При **высоком ε** радиация интенсивная → T_s падает (меньше конвекции
  нужно), Σ Q_рад растёт. На пределе ε → 0,95 в нашей задаче радиация
  выносит больше половины подведённого тепла.

**Что рекомендовать для дюралюминия Д16-Т:**

В наших условиях (плита без специального покрытия, со временем покрытая
естественной оксидной плёнкой) реальное ε ≈ 0,10…0,20. При таких
значениях доля радиации в балансе составляет ~10…20%, а конвекция —
доминирующий механизм отвода тепла. Это согласуется с инженерной
интуицией для голого металла.

Дефолт страницы ε = 0,95 был унаследован из Задачи №1 (где он задавался
для матов с эмалью CERTA). **Для Д16-Т рекомендуется поставить ε = 0,15
в expander «Доп. параметры».**
    ''')


# ── Раздел 3: как считается T(z) — простыми словами ──────────────────────

# Доп. CSS для «карточек шагов».
st.markdown(
    """
    <style>
      .t2-step {
        display: grid;
        grid-template-columns: 64px 1fr;
        gap: 1.4rem;
        padding: 1.1rem 0;
        border-top: 1px solid #e6e6e6;
      }
      .t2-step:last-child { border-bottom: 1px solid #e6e6e6; }
      .t2-step .t2-num {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 2rem; font-weight: 300; color: #b6b6b6;
        line-height: 1; padding-top: 4px;
      }
      .t2-step h3 {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-size: 1.08rem; font-weight: 600;
        color: #1a1a1a; margin: 0 0 0.55rem 0; line-height: 1.25;
      }
      .t2-step p { margin: 0 0 0.55rem 0; color: #2a2a2a; }
      .t2-step .t2-why {
        font-size: 0.86rem; color: #6b6b6b; font-style: italic;
        margin-top: 0.45rem; padding-left: 0.9rem;
        border-left: 2px solid #cfcfcf;
      }
      .t2-step .t2-num-line {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.9rem; color: #2a2a2a;
        background: #f5f5f3; padding: 0.45rem 0.7rem;
        border-radius: 3px; margin: 0.4rem 0;
        display: inline-block;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="t2-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t2-section-eyebrow">Раздел 3 · Как идёт расчёт</p>'
    '<h2 class="t2-section-title">'
    'От подведённого тепла к профилю температуры — простыми словами</h2>',
    unsafe_allow_html=True,
)

# Конкретные числа из текущего прогона.
q_w_val = result.q_w
T_max_val = result.T_max_C
T_min_val = result.T_min_C
n_iter_val = result.iterations
resid_val = result.residual_pct

steps = [
    # (номер, заголовок, тело-html, why-html — опционально)
    (
        '1',
        'Что подводится к плите',
        f'Силиконовые маты на тыльной стороне дают суммарную мощность '
        f'<b>N = {result.N_total_W:.0f} Вт</b>, которую можно представить '
        f'как удельный тепловой поток через тыльную грань площадью b·L:'
        f'<div class="t2-num-line">q<sub>w</sub> = N / (b · L) = '
        f'{result.N_total_W:.0f} / ({result.b_mm:.0f} мм × '
        f'{result.L_mm:.0f} мм) ≈ <b>{q_w_val:.0f} Вт/м²</b></div>'
        'Это число одно и то же на каждом миллиметре высоты — нагрев равномерный.',
        None,
    ),
    (
        '2',
        'Что мы хотим найти',
        'Температуру поверхности плиты вдоль её высоты — <b>T(z)</b>. '
        'Плита тонкая (δ = ' + f'{result.delta_mm:.0f}' +
        ' мм), а тепло через такую толщину проходит мгновенно — поэтому '
        'считаем, что в каждом сечении (на каждой высоте z) температура '
        'одна и та же по всей толщине. Остаётся одна неизвестная функция: '
        'T от z.',
        None,
    ),
    (
        '3',
        'Каналы теплоотвода со всех граней',
        'Подведённая мощность распределяется по четырём каналам:'
        '<ul style="margin:0.4rem 0 0.4rem 1.1rem">'
        '<li><b>Лицевая грань</b> — конвекция + излучение</li>'
        '<li><b>Два боковых торца</b> — то же самое (это вертикальные '
        'поверхности, ведут себя как лицевая)</li>'
        '<li><b>Верхний торец</b> — конвекция (горячая горизонтальная '
        'грань лицом вверх) + излучение</li>'
        '<li><b>Днище</b> — только излучение (тёплый воздух не стекает '
        'вниз, конвективный теплоотвод пренебрежимо мал)</li>'
        '</ul>'
        'Кроме того, внутри металла тепло перераспределяется по высоте '
        'теплопроводностью — от участков с более высокой температурой к '
        'более холодным.',
        'В стационаре сумма всех каналов теплоотвода равна подведённой '
        'мощности. Это закон сохранения, на котором строится уравнение.',
    ),
    (
        '4',
        'Стационарное уравнение в каждой точке высоты',
        'Для элементарного слоя высоты dz составляется баланс: '
        'сумма приходящих тепловых потоков (от матов и от теплопроводности '
        'в металле) равна сумме уходящих в воздух (конвекция и излучение '
        'с вертикальных граней). Подробный вывод дифференциального '
        'уравнения — в нижнем раскрывающемся блоке.',
        None,
    ),
    (
        '5',
        'Особые уравнения для крайних точек',
        'Элементарный слой при z = 0 дополнительно теряет тепло '
        'излучением через днище (других путей отвода через нижний торец '
        'нет — по условию задачи). Аналогично при z = L работает '
        '<b>конвекция по Леонтьеву (формула 7.30, стр. 309)</b> + '
        'излучение через верхний торец. Для этих двух точек уравнение '
        'дополняется соответствующими граничными условиями.',
        None,
    ),
    (
        '6',
        'Уравнение нелинейное: коэффициенты зависят от искомой температуры',
        '<ul style="margin:0.4rem 0 0.4rem 1.1rem">'
        '<li>Коэффициент конвекции <b>α(z, T)</b> по UHF-методике '
        f'(<b>{result.meta.name_ru}</b>) считается через число Нуссельта, '
        'а тому нужны свойства воздуха при опорной температуре — то есть '
        'α сам зависит от T в этой же точке.</li>'
        '<li>Излучение <b>q<sub>рад</sub> = ε·σ·(T⁴ − T<sub>∞</sub>⁴)</b> '
        'содержит T в четвёртой степени.</li>'
        '</ul>'
        'Значит, T входит в уравнение и в левую, и в правую часть, '
        'причём в правой — нелинейно. Прямого аналитического решения нет.',
        'Применяется метод последовательных приближений: задаём пробное '
        'распределение T(z), считаем α и q<sub>рад</sub> при этом T, '
        'получаем новое T(z), и повторяем до сходимости.',
    ),
    (
        '7',
        'Метод последовательных приближений',
        'В качестве первого приближения принимается '
        '<b>T₀(z) ≈ T<sub>∞</sub> + ΔT₀</b>, где ΔT₀ оценивается из '
        'упрощённого баланса с типовым коэффициентом теплоотдачи '
        'h ≈ 8 Вт/(м²·К). Далее на каждой итерации:'
        '<ol style="margin:0.4rem 0 0.4rem 1.1rem">'
        '<li>При текущем T(z) вычисляются α(z) и коэффициент излучения '
        'в каждой точке.</li>'
        '<li>С этими коэффициентами составляется и решается система '
        'уравнений — получается новое T(z).</li>'
        '<li>Сравнивается с предыдущим распределением. Если максимальное '
        'отклонение превышает 0,01 К — итерация повторяется. Иначе '
        'считается сошедшейся.</li>'
        '</ol>'
        f'Текущий прогон: сходимость достигнута за <b>{n_iter_val} итераций</b>; '
        f'на последней итерации максимальное изменение T составило '
        f'<span style="font-family:IBM Plex Mono">{result.max_delta_K:.4f}</span> К.',
        None,
    ),
    (
        '8',
        'Расчёт интегральных тепловых потоков по граням',
        'По сошедшемуся T(z) вычисляются α(z) и q<sub>рад</sub>(z) в каждой '
        'точке. Затем для каждой грани суммируется тепловой поток вдоль '
        'высоты (численное интегрирование трапециевидным методом):'
        '<ul style="margin:0.4rem 0 0.4rem 1.1rem">'
        '<li>Лицевая грань — интегрирование с шириной b даёт Q<sub>лиц</sub>.</li>'
        '<li>Боковые торцы — интегрирование с суммарной шириной 2δ даёт Q<sub>бок</sub>.</li>'
        '<li>Верхний и нижний торец — точечный расчёт q·A при T(L) и T(0) '
        'соответственно.</li>'
        '</ul>'
        'Конвекция и излучение учитываются раздельно — отсюда разбивка '
        'из таблицы баланса выше.',
        None,
    ),
    (
        '9',
        'Контроль качества — невязка теплового баланса',
        'Сумма всех каналов теплоотвода сравнивается с подведённой '
        'мощностью '
        f'<b>Q<sub>вх</sub> = q<sub>w</sub>·b·L = '
        f'{result.Q_input_W:.1f} Вт</b>. По закону сохранения они должны '
        'совпадать. Невязка текущего прогона: '
        f'<b><span style="font-family:IBM Plex Mono">'
        f'{resid_val:+.4f}%</span></b> (на порядки меньше типичной '
        'инженерной точности 1%).',
        'Невязка не используется самим солвером — она вычисляется уже '
        'после сходимости и служит независимой проверкой корректности '
        'модели и численной реализации.',
    ),
]

for num, title, body, why in steps:
    why_block = (
        f'<div class="t2-why">{why}</div>' if why else ''
    )
    st.markdown(
        f'<div class="t2-step">'
        f'<div class="t2-num">{num}</div>'
        f'<div class="t2-body">'
        f'<h3>{title}</h3>'
        f'<div>{body}</div>'
        f'{why_block}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# Уравнение баланса (Шаг 4) — вставляем отдельно из st.latex, чтобы LaTeX
# отрендерился. Делаем под "Шагом 4" контекстно.
with st.expander('Уравнение баланса в развёрнутом виде (формула)'):
    # ── Обозначения ────────────────────────────────────────────────────
    st.markdown('**Обозначения, используемые в формулах ниже:**')
    st.markdown('''
| Символ | Что означает | Единица |
|---|---|---|
| $z$ | координата по высоте, отсчёт от днища | м |
| $T(z)$ | температура поверхности плиты в точке z | °C (К в радиации) |
| $T_\\infty$ | температура окружающего воздуха | °C (К в радиации) |
| $b$ | ширина плиты | м |
| $L$ | высота плиты | м |
| $\\delta$ | толщина плиты | м |
| $A_\\text{кр} = b\\,\\delta$ | площадь поперечного сечения металла | м² |
| $\\lambda_\\text{м}$ | теплопроводность металла плиты | Вт/(м·К) |
| $q_w = N/(b\\,L)$ | удельный тепловой поток от матов | Вт/м² |
| $\\alpha(z, T)$ | коэф. теплоотдачи свободной конвекцией на вертикальной грани (по UHF-методике) | Вт/(м²·К) |
| $\\varepsilon$ | степень черноты поверхности (одинакова для всех граней) | — |
| $\\sigma = 5{,}670\\cdot 10^{-8}$ | постоянная Стефана-Больцмана | Вт/(м²·К⁴) |
| $\\Phi(z)$ | тепловой поток внутри металла через сечение z | Вт |
| $\\dot Q_\\text{отвод}(z)$ | линейная плотность теплоотвода через вертикальные грани | Вт/м |
    ''')
    st.divider()

    # ── 1. Закон Фурье ─────────────────────────────────────────────────
    st.markdown(
        '**1. Тепловой поток в металле вдоль высоты — закон Фурье.** '
        'В любом сечении $z$ через площадь поперечного сечения металла '
        '$A_\\text{кр} = b\\,\\delta$ протекает тепловой поток'
    )
    st.latex(
        r'\Phi(z) \;=\; -\,\lambda_\text{м}\,A_\text{кр}\,'
        r'\dfrac{dT}{dz}\quad [\text{Вт}]'
    )
    st.markdown(
        '— это закон Фурье, переведённый из удельного потока '
        '(Вт/м²) в полный тепловой поток (Вт) умножением на площадь '
        'сечения $A_\\text{кр}$.'
    )
    st.caption(
        'Знак «минус» означает, что тепло течёт в сторону, где '
        'температура ниже. Если плита горячее в середине, чем у краёв, '
        'Φ(z) на нижней половине направлен вниз, а на верхней — вверх.'
    )

    # ── 2. Баланс на элементарном слое ─────────────────────────────────
    st.markdown(
        '**2. Баланс на элементарном слое высоты $dz$** '
        '(между сечениями $z$ и $z+dz$):'
    )
    st.latex(
        r'\Phi(z) \;+\; q_w\,b\,dz \;=\; '
        r'\Phi(z+dz) \;+\; \dot Q_\text{отвод}(z)\,dz'
    )
    st.markdown(
        'Слева — что приходит в слой: тепловой поток снизу '
        '$\\Phi(z)$ и подвод от матов на этом слое $q_w\\,b\\,dz$. '
        'Справа — что уходит: тепловой поток вверх через верхнее '
        'сечение $\\Phi(z+dz)$ и боковые потери $\\dot Q_\\text{отвод}(z)\\,dz$ '
        'с трёх вертикальных граней (лицевая + 2 боковых торца), где'
    )
    st.latex(
        r'\dot Q_\text{отвод}(z) \;=\; \big[\alpha(z,T)\,(T - T_\infty) '
        r'+ \varepsilon\sigma(T^4 - T_\infty^4)\big]\cdot (b + 2\delta)'
        r'\quad [\text{Вт/м}]'
    )
    st.markdown('''
В этой формуле:

- $\\alpha(z, T)\\,(T - T_\\infty)$ — конвективный удельный поток с
  вертикальной грани, Вт/м². $\\alpha$ берётся из выбранной UHF-методики
  (для каждой точки $z$ при текущем $T$).
- $\\varepsilon\\,\\sigma\\,(T^4 - T_\\infty^4)$ — удельный радиационный
  поток, Вт/м². Температуры в Кельвинах: $T_K = t_{°C} + 273{,}15$.
- $(b + 2\\delta)$ — суммарная «ширина» вертикальных граней, м:
  лицевая грань ($b$) плюс два боковых торца ($\\delta$ каждый).
  Умножение на эту ширину переводит удельный поток (Вт/м²) в линейную
  плотность (Вт/м), т.е. поток на единицу высоты.
    ''')

    # ── 3. Производная второго порядка ─────────────────────────────────
    st.markdown(
        '**3. Разность $\\Phi(z+dz) - \\Phi(z)$** — это нетто-разность '
        'тепловых потоков внутри металла на границах слоя. Применяя '
        'закон Фурье к обеим точкам и раскладывая в ряд:'
    )
    st.latex(
        r'\Phi(z+dz) - \Phi(z) \;=\; -\,\lambda_\text{м}\,A_\text{кр}\,'
        r'\dfrac{d^2 T}{dz^2}\,dz'
    )
    st.markdown(
        'Здесь $d^2T/dz^2$ — вторая производная температуры по высоте, '
        'характеризующая кривизну температурного профиля. Если $T(z)$ '
        'имеет максимум, то $d^2T/dz^2 < 0$ — тепло из этой точки '
        'оттекает к соседям с двух сторон.'
    )

    # ── 4. Финальная форма ─────────────────────────────────────────────
    st.markdown(
        '**4. Подставляем (3) в баланс (2) и сокращаем на $dz$** — '
        'получается дифференциальное уравнение, решаемое солвером:'
    )
    st.latex(
        r'\boxed{\;\lambda_\text{м}\,A_\text{кр}\,\dfrac{d^2 T}{dz^2} '
        r'\;+\; q_w\,b '
        r'\;=\; \big[\alpha(z,T)\,(T - T_\infty) '
        r'+ \varepsilon\sigma(T^4 - T_\infty^4)\big]\cdot (b + 2\delta)\;}'
    )
    st.markdown(
        'Каждое слагаемое имеет размерность Вт/м '
        '(тепловая мощность на единицу высоты):'
    )
    st.markdown('''
- $\\lambda_\\text{м}\\,A_\\text{кр}\\,d^2T/dz^2$ — нетто-приток тепла
  за счёт теплопроводности в металле (от/к соседям по высоте).
- $q_w\\,b$ — приток от матов через тыльную грань.
- $\\alpha(z,T)\\,(T - T_\\infty)\\,(b+2\\delta)$ — конвективные потери
  с трёх вертикальных граней.
- $\\varepsilon\\,\\sigma\\,(T^4 - T_\\infty^4)\\,(b+2\\delta)$ —
  радиационные потери с трёх вертикальных граней.
    ''')

    st.markdown(
        '**Граничные условия** к этому уравнению — отдельные балансы '
        'для крайних точек, описанные в Шаге 5 (днище — только излучение '
        'через нижний торец; верхний торец — конвекция по Леонтьеву + '
        'излучение). Их полные формулы — в Разделе 4 ниже.'
    )

with st.expander('Подробности численной реализации (необязательное чтение)'):
    st.markdown(f'''
- Дискретизация по z — {len(result.nodes)} равных шагов
  (dz ≈ {result.L_mm/(len(result.nodes)-1):.2f} мм).
- На каждой итерации составляется и решается линейная система из {len(result.nodes)}
  уравнений (тридиагональная, потому что каждая точка связана только с двумя
  соседями) — методом прогонки Томаса. Это просто очень быстрый прямой
  способ решения таких систем; математически эквивалентен обычной формуле
  Гаусса.
- T⁴ в радиационном члене на одной итерации заменяется на эквивалентное
  «коэффициент × ΔT» (с зафиксированным T из предыдущей итерации) — чтобы
  система оставалась линейной. Это типовой приём, на сходимость не влияет.
- Между итерациями применяется лёгкое «сглаживание»:
  T_new := 0,3·T_old + 0,7·T_solved. Это страхует от резких скачков при
  смене режима конвекции (lam ↔ turb), когда коэффициент α может прыгнуть.
- Контроль сходимости — по максимальному изменению T между итерациями.
- В постобработке (Шаг 8) тепловые потоки считаются по точному T⁴, а не
  по упрощённому виду из шага итерации.
    ''')


# ── Раздел 4: использованные формулы ──────────────────────────────────────

st.markdown('<hr class="t2-rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="t2-section-eyebrow">Раздел 4 · Используемые формулы</p>'
    '<h2 class="t2-section-title">Тепловая модель</h2>',
    unsafe_allow_html=True,
)
with st.expander('Развернуть полную математическую модель'):
    # ── 1. Основное уравнение ─────────────────────────────────────────
    st.markdown('**Одномерное уравнение теплового баланса по высоте z:**')
    st.latex(
        r'\lambda_\text{м}\,A_\text{кр}\,\dfrac{d^2T}{dz^2} '
        r'\;+\; q_w\,b \;-\; '
        r'\big[\alpha_v(T,z)\,(T - T_\infty) + \varepsilon\sigma(T^4-T_\infty^4)\big]'
        r'\,(b+2\delta) \;=\; 0'
    )
    st.markdown('''
**Обозначения** (все температуры — в К для членов с T⁴; в °C везде иначе):

- $z$ — координата по высоте от днища, м
- $T(z)$ — температура поверхности плиты на высоте $z$
- $T_\\infty$ — температура окружающего воздуха
- $b$, $L$, $\\delta$ — ширина, высота, толщина плиты, м
- $A_\\text{кр} = b\\,\\delta$ — площадь поперечного сечения металла, м²
- $\\lambda_\\text{м}$ — теплопроводность металла плиты, Вт/(м·К)
- $q_w = N/(b\\,L)$ — удельный тепловой поток от матов, Вт/м²
- $\\alpha_v(T, z)$ — местный коэф. свободной конвекции на вертикальной
  грани (лицевая + боковые торцы), Вт/(м²·К). Берётся по выбранной
  UHF-методике.
- $\\varepsilon$ — степень черноты поверхности, безразмерна
- $\\sigma = 5{,}670374\\cdot 10^{-8}$ — постоянная Стефана-Больцмана,
  Вт/(м²·К⁴)
- $(b + 2\\delta)$ — суммарная ширина вертикальных граней
  (лицевая $b$ + 2 боковых торца по $\\delta$), м
    ''')

    # ── 2. Граничные условия ───────────────────────────────────────────
    st.markdown('**Граничные условия на крайних точках высоты:**')
    st.markdown('Днище (z = 0) — теплоотвод только излучением через нижний торец:')
    st.latex(
        r'-\lambda_\text{м}\,A_\text{кр}\,\dfrac{dT}{dz}\bigg|_{z=0} '
        r'= \varepsilon\,\sigma\,\big(T(0)^4 - T_\infty^4\big)\,A_\text{дн}'
    )
    st.markdown(
        'Слева — тепловой поток, приходящий по теплопроводности к точке '
        '$z=0$. Справа — поток, уходящий через площадь нижнего торца '
        '$A_\\text{дн} = b\\,\\delta$, м². Конвекция тут отсутствует '
        'по условию задачи.'
    )

    st.markdown('Верхний торец (z = L) — конвекция (по Леонтьеву 1979) + излучение:')
    st.latex(
        r'+\lambda_\text{м}\,A_\text{кр}\,\dfrac{dT}{dz}\bigg|_{z=L} '
        r'= \big[\alpha_\text{верх}\,(T(L) - T_\infty) '
        r'+ \varepsilon\,\sigma\,(T(L)^4 - T_\infty^4)\big]\,A_\text{верх}'
    )
    st.markdown(
        '$A_\\text{верх} = b\\,\\delta$, м² — площадь верхнего торца. '
        '$\\alpha_\\text{верх}$ — коэф. теплоотдачи на горячей '
        'горизонтальной грани, обращённой вверх (формула ниже).'
    )

    # ── 3. UHF для вертикальных граней ─────────────────────────────────
    st.markdown(f'**Конвекция лицевой грани и боковых торцов** — методика '
                f'**{result.meta.name_ru}**:')
    if result.meta.nu_lam_tex:
        st.markdown('Ламинарный режим:')
        st.latex(result.meta.nu_lam_tex)
    if result.meta.nu_turb_tex:
        st.markdown('Турбулентный режим:')
        st.latex(result.meta.nu_turb_tex)
    st.markdown('''
- $\\mathrm{Nu}_x = \\alpha_v\\,x/\\lambda_\\text{возд}$ — местное число
  Нуссельта в точке $x = z$
- $\\mathrm{Ra}_x = \\mathrm{Gr}_x\\,\\mathrm{Pr}$ — число Рэлея через ΔT
  ($\\mathrm{Gr}_x = g\\beta(T - T_\\infty)\\,x^3/\\nu^2$)
- $\\mathrm{Gr}_x^* = g\\beta\\,q_w\\,x^4/(\\lambda_\\text{возд}\\,\\nu^2)$
  — модифицированное число Грасгофа через $q_w$
- $\\mathrm{Pr}$, $\\nu$, $\\lambda_\\text{возд}$ — свойства воздуха при
  опорной температуре методики (плёночной или $t_\\infty$);
  $\\beta = 1/T_\\text{опр}$ для идеального газа
- Затем $\\alpha_v = \\mathrm{Nu}_x \\cdot \\lambda_\\text{возд} / x$,
  Вт/(м²·К)
    ''')

    # ── 4. Леонтьев для верхнего торца ─────────────────────────────────
    L_c_mm = (result.b_mm * result.delta_mm) / (2 * (result.b_mm + result.delta_mm))
    st.markdown('**Конвекция верхнего торца — Леонтьев 1979** '
                '(«Теория тепломассообмена», стр. 309, формула (7.30); '
                'стр. 312 — универсальная корреляция для турб. области):')
    st.latex(r'\mathrm{Nu}_{L_c} = 0{,}54\,\mathrm{Ra}_{L_c}^{1/4},\quad '
             r'10^5 \le \mathrm{Ra}_{L_c} \le 2\cdot 10^7')
    st.latex(r'\mathrm{Nu}_{L_c} = 0{,}135\,\mathrm{Ra}_{L_c}^{1/3},\quad '
             r'2\cdot 10^7 \le \mathrm{Ra}_{L_c} \le 10^{13}')
    st.markdown(f'''
- $L_c = A_\\text{{верх}}/P_\\text{{верх}}$ — характеристический размер
  (обобщение «стороны квадрата» на произвольную плоскую форму).
  $P_\\text{{верх}} = 2(b+\\delta)$ — периметр верхнего торца, м.
  Для прямоугольника b × δ = {result.b_mm:.0f} × {result.delta_mm:.0f} мм
  получается $L_c$ = **{L_c_mm:.2f} мм**.
- $\\mathrm{{Ra}}_{{L_c}} = g\\beta(T(L) - T_\\infty)\\,L_c^3\\,\\mathrm{{Pr}}/\\nu^2$
- $\\alpha_\\text{{верх}} = \\mathrm{{Nu}}_{{L_c}}\\,\\lambda_\\text{{возд}}/L_c$
- Физические свойства воздуха — при плёночной температуре
  $T_f = (T(L) + T_\\infty)/2$ (явно указано в источнике).
    ''')

    # ── 5. Радиация ────────────────────────────────────────────────────
    st.markdown('**Радиация** (на каждой излучающей грани):')
    st.latex(r'q_\text{рад} = \varepsilon\,\sigma\,\big(T_s^4 - T_\infty^4\big)\quad[\text{Вт/м}^2]')
    st.markdown('''
- $T_s$, $T_\\infty$ — температуры поверхности и среды в **Кельвинах**
- $\\varepsilon$ — степень черноты поверхности (одинакова для всех
  излучающих граней)
- $\\sigma = 5{,}670374\\cdot 10^{-8}$ Вт/(м²·К⁴)

Эквивалентная запись, принятая в отечественных учебниках (Керимов 1992):
$q_\\text{рад} = C_\\text{пр}\\,[(T_s/100)^4 - (T_\\infty/100)^4]$, где
$C_\\text{пр} = \\varepsilon\\,\\sigma\\cdot 10^8 \\approx 5{,}67\\,\\varepsilon$,
Вт/(м²·К⁴).
    ''')
