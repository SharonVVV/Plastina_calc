# CLAUDE.md — гид по проекту

Краткая навигация для будущих AI-сессий. Подробности — в `ДОКУМЕНТАЦИЯ_TASK1.md`
и `МЕТОДИКА_РАСЧЁТА.md`.

## Что это

Streamlit-приложение для расчётов свободной конвекции у вертикальной плиты.

**`app.py` теперь — landing-страница** с тремя карточками задач и навигацией.
Старый UI с 8 вкладками (стенд Керимова, qconst-страница, верификация
свойств, потери и т.д.) **скрыт**, исходник сохранён в
`app_legacy_backup.py` (запуск вручную: `streamlit run app_legacy_backup.py`).

Активные треки приложения:

| Трек | Файл | Постановка |
|---|---|---|
| **Лендинг** | `app.py` | Карточки + навигация |
| **Task 1** | `pages/1_Задача_1_Турбулентность.py` | UHF, 8 локальных корреляций, поиск высоты турб. режима |
| **Task 2** | `pages/2_Задача_2_Реальная_пластина.py` | Реальная плита с δ, 1D fin-уравнение, баланс по граням + α(z) + Ra(z) |
| **Task 3** | `pages/3_Задача_3_Подбор_изоляции_торцов.py` | Утепление торцов: материалы, толщина, sweep по % потерь |

Запуск:
```
streamlit run app.py
```
Откроет лендинг с тремя карточками-задачами. Навигация — через карточки
либо через сайдбар слева (multipage Streamlit). Старый UI (`app_legacy_backup.py`)
запускается отдельно, его в основном приложении нет.

## Структура файлов

**Общие (используют все треки):**
```
config.py                   — DEFAULTS_TASK1, DEFAULTS_TASK2, DEFAULTS_ENV,
                              DEFAULTS_HEATER, DEFAULTS_QCONST + LEGACY-блоки
properties.py               — get_air_properties(T, P) через CoolProp
formatting.py               — _fc, _fe, _fcl, _fl (десятичная запятая)
```

**Задача 1 (сравнение методик, UHF):**
```
correlations_uhf.py         — 8 локальных UHF-корреляций (нотация источников)
correlations_meta.py        — метаданные: LaTeX, диапазоны, цвета, ORDER
task1_solver.py             — итерационный UHF-солвер (Brent по T_s),
                              обратная задача (бисекция по I/N)
plotting_task1.py           — диаграмма «плита с цветными полосами»
pages/1_Задача_1_Турбулентность.py — Streamlit-страница
```

**Задачи 2 и 3 (реальная пластина с толщиной, баланс по граням):**
```
correlations_horizontal.py  — Леонтьев 1979 (7.30) для верхнего торца
task2_solver.py             — 1D fin-уравнение по высоте + Picard + Thomas,
                              MATERIALS = {Д16, АМг3, нерж, медь};
                              + compute_back_front_temps() для Task 3
plotting_task2.py           — plot_temperature_profile (Task 2),
                              plot_temperature_profile_back_front (Task 3),
                              plot_heat_balance (общий)
pages/2_Задача_2_Реальная_пластина.py — Task 2: T_mean(z) + баланс
pages/3_Задача_3_Температура_лицевой_грани.py — Task 3: T_front(z) + T_back(z)
```

**Task 3 — отличия от Task 2:** базовый солвер тот же. Дополнительно
вызывается `compute_back_front_temps(result)`, которая по линейному
профилю поперёк δ восстанавливает T_back и T_front из решения T_mean
(допущение Bi ≪ 1). KPI/график/Step-by-step текст переориентированы
на T_front как первичный результат, T_back — вторичный (пунктир).

Legacy-файлы (не трогаем без явной просьбы):
`app_legacy_backup.py` (=исходный `app.py`), `solver.py`, `correlations.py`,
`turbulence_*.py`, `qconst_page.py`, `back_losses*.py`, `comparison_page.py`,
`verification_*.py`, `calculation_log.py`, `plotting.py`.

## Эталон-источник

`/Users/SharonovVV/Downloads/convection_methods.md` — справочник из 1190 строк,
формулы verbatim из верифицированных по PDF первоисточников (Vliet 1969,
Bejan 2013, Holman 2010, Jiji 2009, Incropera 2011, Леонтьев 2018,
ЦКВ 2008). **Любое изменение формул в `correlations_uhf.py` должно
сверяться с этим файлом.** Дополнительно — PDF-скан методички Керимова 1992
(в чате истории).

## Ключевые конвенции (Задача 1)

- **Все 8 методик — UHF (q_w = const)**, локальные Nu_x. UWT-формулы для
  Задачи 1 не пригодны (нагрев физически — q=const и для Джоуля, и для матов).
- **Опорная температура** — per-methodology через `UHFCorrelation.t_ref`:
  - `'film'` (плёночная) — 7 методик
  - `'fluid'` (t_∞) — только Керимов 1992
- **ε_t = (Pr_∞/Pr_c)^0.25** — поправка Керимова 1992; вычисляется всегда,
  но передаётся в Nu только тем методикам, у которых `'eps_t' in needs`.
- **Параметр режима** у каждой методики свой:
  - Ra (через ΔT): ЦКВ, Леонтьев, Керимов 1992
  - Gr·Pr (через q_w): Vliet, Holman, Fujii
  - Ra* (тождественно Gr*·Pr): Bejan/Vliet-Liu, Bejan/air
- **Тождество** `Ra_* = Gr*·Pr` — используется в `compute_nu` без дублирования.
- **Out-of-range** — каждая Nu-функция возвращает `REG_OUT_OF_RANGE`, если
  параметр режима вне декларированного диапазона; визуализатор рисует
  серую штриховку.

## Ключевые конвенции (Задача 2)

- **1D fin-уравнение** по высоте z (Bi через толщину ≪ 1 для всех
  металлов → плита изотермична поперёк).
- **Boundary conditions** через ghost-узлы (half-element):
  - z=0 (днище): только радиация
  - z=L (верх): Леонтьев 1979 (формула 7.30, стр. 309 + универсальная
    корреляция стр. 312) + радиация. Коэффициенты 0,54 (lam, 10⁵…2·10⁷)
    и 0,135 (turb, 2·10⁷…10¹³). L_c = A/P (обобщение «стороны квадрата»).
- **Конвекция на вертикальных гранях** (лицевая + 2 боковых торца) — та же
  UHF-методика, что выбрана в Задаче 1 (один dropdown в UI).
- **Линеаризация радиации** для тридиагональной системы:
  $q_\text{рад} = h_\text{рад}(T)·\Delta T$, где
  $h_\text{рад} = \varepsilon\sigma(T_s+T_\infty)(T_s^2+T_\infty^2)$.
- **Picard-итерация с релаксацией** (`relax=0.7`), точность `tol_K=0.01`,
  обычно сходится за 7-16 итераций.
- **Контроль баланса** — `residual_pct = (Q_in − Q_out)/Q_in` должно быть
  ≪ 0,01% при сходимости. Это smoke-test любого изменения солвера.
- **Материалы** — встроенные ключи в `task2_solver.MATERIALS`
  (Д16-Т, АМг3, AISI 304, медь), + поле в expander для произвольного λ.

## Типографика и UI (Задача 1)

- Шрифты: **IBM Plex Sans / Serif / Mono** (подгружаются с Google Fonts
  через `<link>` в шапке страницы).
- **Не перебивать `font-feature-settings: liga`** для иконок Streamlit —
  иначе Material Symbols ломаются и текст «keyboard_arrow_right» вылезает
  буквами. CSS-блок в шапке `pages/1_*.py` уже это обходит явным
  принудительным `font-family + liga + variant-ligatures` для
  `[data-testid="stIconMaterial"]`.
- Диаграммы — **по очереди вертикально**, не side-by-side, чтобы каждой
  доставалось ≥1000px ширины и 8 полос не наезжали друг на друга.
- Подписи методик и координаты переходов — **ПОД** плитой (не над).

## Команды

| Цель | Команда |
|---|---|
| Запустить UI | `streamlit run app.py` |
| Smoke-test 8 методик (Task 1) | `python3 -c "from task1_solver import *; from correlations_meta import ORDER; …"` |
| Smoke-test Task 2 | `python3 -c "from task2_solver import solve_task2; r = solve_task2(key='kerimov_1992', N_total_W=1000, b_mm=100, L_mm=2000, delta_mm=12, lambda_metal=130, …); print(r.residual_pct)"` |
| Compile-check Task 1 | `python3 -m py_compile correlations_uhf.py task1_solver.py correlations_meta.py plotting_task1.py "pages/1_Задача_1_Турбулентность.py"` |
| Compile-check Task 2 | `python3 -m py_compile correlations_horizontal.py task2_solver.py plotting_task2.py "pages/2_Задача_2_Реальная_пластина.py"` |
| Тесты (legacy) | `pytest natural_convection/tests/` |

## Что НЕ менять без серьёзных оснований

- Формулы в `correlations_uhf.py` — сверены с первоисточниками. Любые
  правки коэффициентов / показателей степени / границ диапазонов должны
  иметь явную ссылку на страницу источника.
- Поле `t_ref` методики — это часть её физической постановки, не косметика.
- Структура `UHFCorrelation.needs` — определяет, что передавать в Nu.
  Добавление аргумента требует обновления `compute_nu`.

## Где история решений

- План Задачи 1: `/Users/SharonovVV/.claude/plans/pure-mixing-pebble.md`
- Аудит-сессии Задачи 1 (4 итерации до 10/10): краткое резюме в
  `ДОКУМЕНТАЦИЯ_TASK1.md` → раздел «История правок».
- Подробное описание Задачи 2 (модель, алгоритм, smoke-результаты):
  `ДОКУМЕНТАЦИЯ_TASK2.md`.
- Legacy документация старого UI: `МЕТОДИКА_РАСЧЁТА.md` (помечена legacy).
