"""
Главная страница приложения «Расчёт теплообмена вертикальной плиты».

Landing с тремя карточками, по одной на задачу. Каждая карточка ведёт
на соответствующую страницу в `pages/`.

Старый UI с 8 вкладками (стенд Керимова, qconst-страница, верификация,
потери и т.д.) сохранён в `app_legacy_backup.py` — для запуска вручную
при необходимости. Из основного приложения он скрыт.
"""

import streamlit as st


st.set_page_config(
    page_title='Расчёт теплообмена пластины',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ── Стили (IBM Plex + защита Material-иконок) ────────────────────────────

_LANDING_CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {font-family: "IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif; font-feature-settings: "ss01", "tnum";}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *, .material-symbols-rounded, .material-symbols-outlined, .material-icons {font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important; font-feature-settings: "liga" !important; -webkit-font-feature-settings: "liga" !important; font-variant-ligatures: common-ligatures !important; letter-spacing: normal !important;}
.land-eyebrow {font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: #6b6b6b; margin: 0 0 0.6rem 0;}
.land-title {font-family: "IBM Plex Serif", Georgia, serif; font-weight: 600; font-size: 2.6rem; line-height: 1.12; color: #1a1a1a; margin: 0 0 0.6rem 0;}
.land-lede {max-width: 56rem; color: #4b4b4b; font-size: 1.02rem; line-height: 1.6; margin: 0 0 2.4rem 0;}
.land-rule {border: 0; border-top: 1px solid #d8d8d8; margin: 2.5rem 0 1.6rem 0;}
.task-card {position: relative; border: 1px solid #d8d8d8; border-radius: 4px; padding: 1.5rem 1.5rem 1.2rem 1.5rem; background: #ffffff; height: 100%; display: flex; flex-direction: column; gap: 0.55rem;}
.task-card::before {content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 4px 0 0 4px;}
.task-card.t1::before {background: #1f3a5f;}
.task-card.t2::before {background: #c44e2a;}
.task-card.t3::before {background: #2c8d52;}
.task-card.t4::before {background: #6a4b9c;}
.task-card .task-num {font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: #6b6b6b; margin: 0;}
.task-card h2 {font-family: "IBM Plex Serif", Georgia, serif; font-weight: 600; font-size: 1.32rem; line-height: 1.22; color: #1a1a1a; margin: 0 0 0.2rem 0;}
.task-card .goal {font-size: 0.95rem; color: #2a2a2a; margin: 0 0 0.55rem 0; line-height: 1.5;}
.task-card ul.deliv {list-style: none; padding: 0; margin: 0.4rem 0 0.8rem 0; font-size: 0.88rem; color: #4b4b4b; line-height: 1.55;}
.task-card ul.deliv li {position: relative; padding-left: 1.0rem; margin-bottom: 0.25rem;}
.task-card ul.deliv li::before {content: '·'; position: absolute; left: 0.2rem; color: #6b6b6b; font-weight: 700;}
.task-card .meta-row {margin-top: auto; padding-top: 0.7rem; border-top: 1px solid #ececec; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: #6b6b6b; letter-spacing: 0.08em; text-transform: uppercase; display: flex; gap: 1rem; flex-wrap: wrap;}
.land-footer {margin: 3rem 0 0 0; color: #6b6b6b; font-size: 0.86rem; line-height: 1.55; max-width: 56rem;}
.land-footer code {font-family: "IBM Plex Mono", ui-monospace, monospace; background: #f3f3ef; padding: 1px 6px; border-radius: 3px; font-size: 0.82rem;}
</style>
"""

st.markdown(_LANDING_CSS, unsafe_allow_html=True)


# ── Шапка ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <p class="land-eyebrow">Лаборатория теплообмена · НИУ МЭИ</p>
    <h1 class="land-title">Расчёт свободной конвекции у&nbsp;вертикальной плиты</h1>
    <p class="land-lede">
      Стенд с электрическим нагревом плиты в условиях постоянного теплового
      потока (UHF, q<sub>w</sub> = const). Три задачи последовательно
      раскрывают физическую модель и инженерные решения: от анализа
      турбулентного перехода к расчёту реальной плиты и подбору
      изоляции для торцов.
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Три карточки задач ───────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4, gap='medium')


def _card(col, *, css_class, num, title, goal, deliverables, meta):
    deliv_html = ''.join(f'<li>{d}</li>' for d in deliverables)
    meta_html = ''.join(f'<span>{m}</span>' for m in meta)
    col.markdown(
        f"""
        <div class="task-card {css_class}">
          <p class="task-num">{num}</p>
          <h2>{title}</h2>
          <p class="goal">{goal}</p>
          <ul class="deliv">
            {deliv_html}
          </ul>
          <div class="meta-row">{meta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_card(
    col1,
    css_class='t1',
    num='Задача 1',
    title='Высота достижения турбулентного режима',
    goal='Сравнить восемь верифицированных локальных UHF-корреляций '
         'свободной конвекции и найти высоту, на которой каждая из них '
         'предсказывает переход к турбулентному режиму.',
    deliverables=[
        'Полосовая диаграмма пластины с зонами lam/trans/turb по 8 методикам',
        'Обратная задача: минимальный ток / мощность для достижения '
        'турбулентного режима на 80 % высоты',
        'Таблица границ режимов и характерных чисел Ra / Gr<sup>*</sup>·Pr',
        'Развёрнутые формулы по каждой методике (LaTeX из первоисточников)',
    ],
    meta=[
        'плита: тонкая (без δ)',
        '8 методик',
        'UHF — q<sub>w</sub> = const',
    ],
)
with col1:
    st.page_link(
        'pages/1_Задача_1_Турбулентность.py',
        label='Открыть Задачу 1 →',
        use_container_width=True,
    )

_card(
    col2,
    css_class='t2',
    num='Задача 2',
    title='Реальная пластина — баланс тепла',
    goal='Учесть реальную толщину δ плиты и продольную теплопроводность '
         'в металле. Найти распределение T(z) по высоте и распределение '
         'теплоотвода между лицевой гранью, боковыми торцами, верхним '
         'торцом и днищем.',
    deliverables=[
        'Профиль T(z) с подсветкой режима свободной конвекции',
        'Профили α(z) и Ra(z) для диагностики',
        'Stacked-bar баланса + двухуровневая таблица «грань → механизм»',
        'Чувствительность к степени черноты ε (от полированного Д16 до эмали)',
    ],
    meta=[
        'плита: b × L × δ',
        '1D fin-уравнение',
        'материалы: Д16-Т / нерж. / медь',
    ],
)
with col2:
    st.page_link(
        'pages/2_Задача_2_Реальная_пластина.py',
        label='Открыть Задачу 2 →',
        use_container_width=True,
    )

_card(
    col3,
    css_class='t3',
    num='Задача 3',
    title='Подбор изоляции для торцов',
    goal='Утеплить торцы плиты слоем теплоизоляции и оценить, как меняется '
         'распределение потерь. Найти оптимальную толщину и материал '
         'для заданной геометрии.',
    deliverables=[
        'Сравнение балансов «без / с изоляцией» в одной таблице',
        'Sweep по толщине δ<sub>изо</sub> = 0…100 мм: доли потерь и T<sub>max</sub>',
        'Сравнение 9 материалов изоляции при заданной толщине',
        'Независимые тумблеры для боковых торцов / верха / днища',
    ],
    meta=[
        '1/h<sub>eff</sub> = δ<sub>изо</sub>/λ<sub>изо</sub> + 1/α',
        '9 материалов',
        'через изоляцию нет излучения',
    ],
)
with col3:
    st.page_link(
        'pages/3_Задача_3_Подбор_изоляции_торцов.py',
        label='Открыть Задачу 3 →',
        use_container_width=True,
    )

_card(
    col4,
    css_class='t4',
    num='Задача 4',
    title='Симулятор лабораторного стенда',
    goal='Виртуальная копия установки Керимова. Задайте мощность, '
         'нажмите «Включить» — стенд прогревается с анимацией ~5 сек до '
         'установившегося режима. 20 термопар по высоте «снимают» '
         'распределение температур, рядом — псевдо-BOS погранслоя.',
    deliverables=[
        'Тепловая карта пластины (Inferno) с маркерами 20 термопар',
        'Псевдо-BOS / шлирен-визуализация погранслоя сбоку',
        'Анимация прогрева T(t) = T<sub>∞</sub> + ΔT·(1 − e<sup>−t/τ</sup>)',
        'Таблица показаний термопар + расчётные блоки в expander',
    ],
    meta=[
        '20 термопар',
        'animation ~ 5 c',
        'BOS-шлирен',
    ],
)
with col4:
    st.page_link(
        'pages/4_Задача_4_Симуляция.py',
        label='Открыть Задачу 4 →',
        use_container_width=True,
    )


# ── Низ: техническая справка ─────────────────────────────────────────────

st.markdown('<hr class="land-rule">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="land-footer">
      <p><b>Технические основы.</b>
      Все корреляции сверены по PDF/сканам первоисточников — справочник
      <code>convection_methods.md</code>. Свойства воздуха — через CoolProp.
      Численный метод — конечно-разностная дискретизация по высоте
      с итерационным уточнением коэффициентов.</p>
      <p><b>Источники конвективных корреляций.</b>
      Керимов Р.В. (МЭИ, 1992) · Цветков, Керимов, Величко (2008) ·
      Vliet G.C. (1969) · Леонтьев А.И. ред. (2018) · Holman J.P. (2010) ·
      Bejan A. (2013) · Jiji L.M. (2009) ·
      МакАдамс/Леонтьев для горизонтального торца.</p>
      <p><b>Запуск приложения.</b>
      <code>streamlit run app.py</code> — открывает эту страницу.
      Навигация по задачам — через карточки выше или сайдбар слева.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
