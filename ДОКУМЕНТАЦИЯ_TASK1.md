# Документация расчётных модулей — Задача №1

> **Задача №1.** Определить высоту достижения турбулентного режима
> свободной конвекции на вертикальной плите при постоянном тепловом
> потоке (UHF, q_w = const). Сравнить восемь локальных корреляций
> Nu_x = f(Ra, Pr) из верифицированных первоисточников.

Документ описывает четыре новых модуля (`correlations_uhf.py`,
`correlations_meta.py`, `task1_solver.py`, `plotting_task1.py`) и страницу
`pages/1_Задача_1_Турбулентность.py`. Для **legacy**-кода (стенд Керимова,
старый UI с 8 вкладками) — см. `МЕТОДИКА_РАСЧЁТА.md`.

---

## 1. Постановка задачи

### 1.1. Геометрия и нагрев

Тонкая вертикальная плита, толщина пренебрежимо мала (двумерная задача).

- Ширина `b` (по умолчанию 100 мм), высота `L` (по умолчанию 2000 мм).
- Два источника нагрева, оба сводятся к **UHF (q_w = const)**:
  - **Джоулев**: ток `I` через плиту, q_w(T_s) = (I²·R₂₀/b)·[1 + (T_s−20)·α_R].
    Слабая зависимость от T_s через ТКС — практически q=const.
  - **Силиконовые маты**: суммарная мощность `N`,
    q_w = N/(b·L) — строго постоянна.

### 1.2. Тепловой баланс в каждом сечении x

В точке x ∈ [x_min, L] решается уравнение:

$$
F(T_s) = q_w(T_s) - \alpha(T_s, x)\cdot(T_s - T_\infty)
        - \varepsilon\cdot\sigma\cdot(T_s^4 - T_\infty^4) = 0
$$

где
- α(T_s, x) = Nu_x·λ/x — коэф. теплоотдачи,
- λ, ν, Pr — свойства воздуха при опорной температуре методики,
- σ = 5,670374419·10⁻⁸ Вт/(м²·К⁴),
- ε — степень черноты (одинаковая для обеих модальностей нагрева).

Решается Brent-методом по T_s в интервале [T_∞ + 0.01, T_∞ + 1500 °C]
(fallback до +1700 °C при необходимости).

### 1.3. Критерии режимов

Каждая методика классифицирует течение по СВОЕМУ параметру:

| Параметр | Определение | Кто использует |
|---|---|---|
| Ra (через ΔT) | Gr·Pr; Gr = g·β·ΔT·x³/ν² | ЦКВ, Леонтьев, Керимов 1992 |
| Gr·Pr (через q_w) | Gr_x*·Pr; Gr_x* = g·β·q_w·x⁴/(λ·ν²) | Vliet, Holman, Fujii |
| Ra* | Тождественно Gr_x*·Pr | Bejan/Vliet-Liu, Bejan/air |

В коде эти три формы доступны одновременно в каждой точке:
```python
GrPr_star = Gr_star * Pr   # = Ra_star (Bejan-нотация)
```

---

## 2. Восемь верифицированных методик

Все формулы записаны в нотации первоисточника. Проверены по PDF/сканам.
Полная сверка — в `/Users/SharonovVV/Downloads/convection_methods.md`
и PDF-скане методички Керимова 1992 (приведён пользователем в чате).

| # | Ключ | Источник | Lam-формула | Turb-формула | t_ref |
|---|---|---|---|---|---|
| 1 | `kerimov_1992` | Керимов 1992, МЭИ, форм. (2) | 0,60·(Gr·Pr)^¼·ε_t, 10³…10⁹ | 0,15·(Gr·Pr)^⅓·ε_t, >6·10¹⁰ | **t_∞** |
| 2 | `ckv` | ЦКВ 2008, (3.4)+(3.8) | 0,563·[Ra·Φ(Pr)]^¼, 10⁴…10⁹ | 0,15·[Ra·Φ(Pr)]^⅓, >10¹² | t_film |
| 3 | `vliet` | Vliet 1969, (1)+(2) | 0,60·(Gr*·Pr)^⅕, 10⁸…1,3·10¹³ | 0,30·(Gr*·Pr)^0,24, 10¹⁴…10¹⁶ | t_film |
| 4 | `leontiev` | Леонтьев 2018, Брдлик + Эккерт-Джексон | 0,616·[Pr/(Pr+0,8)]^⅕·(Gr*·Pr)^⅕ | 0,0295·Ra^⅖·Pr^(1/15)·(1+0,494·Pr^⅔)^(−⅖) | t_film |
| 5 | `holman` | Holman 2010, (7-31)+(7-32) | 0,60·(Gr*·Pr)^⅕, 10⁵…10¹¹ | 0,17·(Gr*·Pr)^¼, 2·10¹³…10¹⁶ | t_film |
| 6 | `bejan_vliet_liu` | Bejan 2013, (4.108)+(4.109) | 0,6·Ra*^⅕, 10⁵…10¹³ | 0,568·Ra*^0,22, 10¹³…10¹⁶ | t_film |
| 7 | `bejan_air` | Bejan 2013, (4.110)+(4.111) — для воздуха | 0,55·Ra*^⅕ | 0,17·Ra*^¼ | t_film |
| 8 | `fujii_fujii` | Fujii-Fujii 1976 через Jiji (7.32)+(7.33) | [Pr/(4+9√Pr+10Pr)]^⅕·(Pr·Gr_x*)^⅕ | — (только ламинар) | t_film |

Где:
- $\Phi(Pr) = [1 + (0{,}437/Pr)^{9/16}]^{-16/9}$ — поправка ЦКВ для UHF.
- $\varepsilon_t = (Pr_\infty/Pr_c)^{0{,}25}$ — поправка Керимова на переменность свойств.
- Для воздуха при 20…200 °C: Φ ≈ 0,363; ε_t ≈ 1,003…1,015.

### 2.1. Особый случай — Керимов 1992

**Единственная методика с `t_ref='fluid'`** (свойства при t_∞, не плёночной).
Источник: Керимов Р.В. Лабораторная работа № 9 и 9а по курсу «Тепломас-
сообмен». Местная теплоотдача при свободном движении воздуха около верти-
кальной пластины. — М.: Изд-во МЭИ, 1992. — 10 с.

Геометрия исходного стенда: h=1540 мм, b=205 мм, δ=1,0 мм (нержавеющая
сталь, прямой Джоулев нагрев).

### 2.2. Транзитные зоны

В источниках транзитная зона между ламинаром и турбулентом обычно не
описывается отдельной корреляцией. В коде используется **log-интерпо-
ляция** между значениями Nu на границах:

```python
t = (log10(param) - log10(lam_max)) / (log10(turb_min) - log10(lam_max))
Nu = Nu_lam(lam_max) + t * (Nu_turb(turb_min) - Nu_lam(lam_max))
```

Это явно отмечено в notes каждой методики.

---

## 3. Архитектура модулей

### 3.1. `correlations_uhf.py`

Чистые функции, не зависят от Streamlit и от какого-либо солвера.

```python
@dataclass
class NuResult:
    Nu: float
    regime: str       # 'lam' | 'trans' | 'turb' | 'out'
    param_name: str   # 'Ra' | 'Gr·Pr' | 'Gr*Pr' | 'Ra*'
    param_value: float

@dataclass
class UHFCorrelation:
    key: str
    needs: tuple[str, ...]   # из {'Ra','GrPr_star','Ra_star','Gr_star','Pr','eps_t'}
    fn: Callable
    t_ref: str = 'film'      # 'film' | 'fluid'

REGISTRY: dict[str, UHFCorrelation] = { ... }  # 8 методик

def compute_nu(key, *, Ra, Gr_star, Pr, eps_t=1.0) -> NuResult
def t_ref_for(key) -> str
```

**Диспетчер `compute_nu`** знает, что подставить каждой методике из
полного словаря величин в точке:
```python
args = {'Ra': Ra, 'GrPr_star': Gr_star*Pr, 'Ra_star': Gr_star*Pr,
        'Gr_star': Gr_star, 'Pr': Pr, 'eps_t': eps_t}
needed = {n: args[n] for n in corr.needs}
return corr.fn(**needed)
```

### 3.2. `correlations_meta.py`

Метаданные для UI: LaTeX-формулы, краткие имена, границы режимов,
цвета, источники. Один источник истины:

```python
@dataclass
class CorrelationMeta:
    key, name_ru, name_short, full_ru, source
    boundary_param: 'Ra' | 'Gr·Pr' | 'Gr*·Pr' | 'Ra*'
    lam_min, lam_max, turb_min, turb_max  # Optional[float]
    nu_lam_tex, nu_turb_tex                # LaTeX (Optional[str])
    notes_ru, color

META: dict[str, CorrelationMeta]
ORDER: list[str]                # порядок отображения слева направо
```

Палитра — 8 насыщенных, визуально различимых цветов.

### 3.3. `task1_solver.py`

```python
SIGMA = 5.670374419e-8

def q_w_from_N(N_total_W, b_mm, L_mm) -> float
def q_w_from_I_at_T(I, R20, b_mm, t_s_C, alpha_R) -> float

@dataclass
class PointUHF:
    x_m, t_s_C, t_film_C, q_w
    Gr_x, Ra_x, Gr_x_star, GrPr_star
    Nu_x, alpha, regime, converged

@dataclass
class MethodologyResult:
    key, meta, points
    x_lam_to_trans_m, x_trans_to_turb_m, x_turb_start_m

# Прямая задача
def run_methodology_joule(key, *, I, R20, alpha_R, b_mm, L_mm,
                          t_fluid_C, g, P_Pa, eps_surface,
                          x_min_mm, N) -> MethodologyResult
def run_methodology_qconst(key, *, N_total_W, b_mm, L_mm, ...) -> MethodologyResult

# Обратная задача — минимум I (А) либо N (Вт) для достижения turb на target·L
@dataclass
class InverseSearchResult:
    achievable: bool
    value: Optional[float]
    x_turb_m: Optional[float]
    n_iter: int
    message: str

def find_min_input_for_turbulence(*, mode, key, target_x_m, …,
                                  search_min, search_max,
                                  n_scan=12, bisect_iter=12, bisect_tol=0.5)
```

Алгоритм обратной задачи:
1. Грубое линейное сканирование по `[search_min..search_max]` из `n_scan`
   точек — ищем первую точку, где turb достигается ≤ target_x_m.
2. Бисекция между «не достиг» и «достиг» — `bisect_iter` итераций
   либо до точности `bisect_tol`.

Для методик без турбулентной формулы (Fujii-Fujii) возвращается
`achievable=False, message='Методика не описывает турбулентный режим.'`

### 3.4. `plotting_task1.py`

Одна функция:

```python
def plot_methodology_strips(width_mm, height_mm, results,
                             target_fraction=0.8) -> plotly.graph_objects.Figure
```

Композиция:
- **Сверху**: горизонтальная Plotly-легенда — 4 квадратных чипа
  (Ламинарный / Переходный / Турбулентный / Вне диапазона применимости).
- **Центр**: N узких вертикальных полос (по одной на каждую методику в
  `results`), окрашенных по зонам её точек; контур — в цвете методики.
  Внутри полосы — пунктир (lam→trans) и тире (начало turb).
- **Под плитой**: имя методики (цветное, IBM Plex Sans bold) +
  координаты переходов (моноширинно через `<span style="font-family:…">`).
- **Внизу**: подпись геометрии b × L (моноширинные числа).

Целевая линия 0,8·L **намеренно не рисуется** на диаграмме — значение
используется только в обратной задаче в таблице ниже.

### 3.5. `pages/1_Задача_1_Турбулентность.py`

Streamlit-страница. Структура (по разделам):

1. **Шапка** — eyebrow «ЗАДАЧА №1», серифный заголовок, лиде.
2. **Ввод**: 4 поля (b, L, I, N) + expander «Доп. параметры» (среда,
   ε, R₂₀, α_R, число точек, целевая высота, выбор методик).
3. **Раздел 1 · Джоулев нагрев** — диаграмма (полная ширина) + таблица.
4. **Раздел 2 · Силиконовые маты** — то же.
5. **Раздел 3 · Корреляции** — expander с LaTeX-формулами и описаниями
   каждой методики.

CSS: IBM Plex (Sans + Serif + Mono) через Google Fonts. Селекторы
скоплены к конкретным `data-testid` Streamlit-элементам, чтобы не
ломать Material-иконки (см. CLAUDE.md → «Типографика»).

---

## 4. Поток данных (одна страница, один прогон)

```
[User input]
   b, L, I, N, t_∞, P, ε, R₂₀, α_R, N точек, target%, methods[]
        ↓
[task1_solver.run_methodology_joule(key, …)]  ← для каждой методики
   q_w(T_s) = I²·R/b·(1 + ΔT·α_R)
   for x in linspace(x_min, L, N):
      brentq(F(T_s), …) → T_s
      F(T_s) = q_w − α(T_s)·ΔT − ε·σ·(T_s⁴−T_∞⁴)
      α = Nu_x·λ/x;  Nu_x from compute_nu(key, …)
   ↓ MethodologyResult { points, x_lam_to_trans, x_trans_to_turb }
        ↓
[plotting_task1.plot_methodology_strips(results)]
   for each MethodologyResult:
      paint zones, contour, transition lines
      labels below: name + coord
   ↓ go.Figure
        ↓
[task1_solver.find_min_input_for_turbulence(key, target_x_m=0.8·L)]
   scan + bisect by I (или N)
   ↓ InverseSearchResult
        ↓
[_summary_df → st.dataframe]
   методика | param | lam→trans/turb | turb (начало) | param@L | минимум I/N
```

Кэширование — `@st.cache_data` поверх обоих `_run_*_all` и обоих
`_inv_*_all` с ключом `tuple(methods_selected)`. Прогон 8 методик
× 80 точек × ~10 итераций Brent → ~7000 вычислений свойств воздуха
за прямой расчёт; обратная задача добавляет ещё ~12 итераций × N точек.

---

## 5. Расширение: добавление новой методики

1. **`correlations_uhf.py`** — написать функцию `nu_<name>(...)`,
   возвращающую `NuResult`. Указать диапазоны режимов через именованные
   константы (REG_LAM / REG_TRANS / REG_TURB / REG_OUT_OF_RANGE).
2. **Зарегистрировать** в `REGISTRY`:
   ```python
   'my_corr': UHFCorrelation('my_corr', ('Ra', 'Pr'), nu_my_corr, t_ref='film')
   ```
3. **`correlations_meta.py`** — добавить запись в `META` с LaTeX-форм.,
   границами и цветом. Добавить ключ в `ORDER`.
4. Готово — страница автоматически подхватит методику из `ORDER`.

Если методика требует **нового аргумента** (которого нет в стандартном
наборе `Ra / GrPr_star / Ra_star / Gr_star / Pr / eps_t`):
- Добавить вычисление этой величины в `_balance_residual` и `_solve_point`.
- Расширить `compute_nu(...)` соответствующим kwargs с default-значением.
- Добавить в словарь `args` диспетчера `compute_nu`.

---

## 6. Валидация и аудит

Реализация прошла **четыре итерации аудита** независимым агентом
(физико-вычислительная проверка):

| Итерация | Оценка | Что закрыли |
|---|---|---|
| 1 | 8,5/10 | Исходный набор формул, упущенные out-of-range, хрупкие селекторы |
| 2 | 9,7/10 | Stitched-границы lam→turb, q_w-helpers, legacy-маркеры |
| 3 | 10/10 | Docstrings, legend, layout |
| 4 (после Керимов 1992) | 9,5 → 10/10 | t_ref-инфраструктура, ε_t, обновлённые docstrings |

Smoke-проверка финального состояния (b=100, L=2000, I=300, N=200, t_∞=20):

| Методика | x_lam→trans | x_turb | Ra или Gr*·Pr @ L |
|---|---|---|---|
| Керимов 1992 | 388 мм | 1547 мм | Gr·Pr = 1,35·10¹¹ |
| ЦКВ 2008 | 589 мм | не достиг. | Ra = 4,4·10¹⁰ |
| Vliet 1969 | 1421 мм | не достиг. | Gr*·Pr = 5,86·10¹³ |
| Леонтьев 2018 | 514 мм (стык) | 514 мм | Ra = 4,4·10¹⁰ |
| Holman 2010 | 438 мм | 1572 мм | Gr*·Pr = 5,49·10¹³ |
| Bejan/Vliet-Liu | 1295 мм (стык) | 1295 мм | Ra* = 5,98·10¹³ |
| Bejan/air | 1345 мм (стык) | 1345 мм | Ra* = 5,49·10¹³ |
| Fujii-Fujii | — | (только лам.) | Gr*·Pr = 4,88·10¹³ |

«Не достиг.» = параметр методики не пересёк её собственную турбулентную
границу на L=2 м при данном q_w. Физически осмысленно: ЦКВ требует
Ra > 10¹², а реальная Ra на 2-метровой плите при q_w ≈ 3 кВт/м²
поднимается лишь до ~10¹⁰⋅²; Vliet требует Gr*·Pr > 10¹⁴ — тоже за
пределом этой геометрии.

---

## 7. История правок (high-level)

- **2026-05-18, утро** — план в `~/.claude/plans/pure-mixing-pebble.md`,
  создание 7-методичного UHF-набора (без Керимова 1992).
- **2026-05-18, день** — три цикла аудита, доводка до 10/10.
- **2026-05-18, вечер** — переработка layout под IBM Plex / швейцарскую
  сетку; разнесение диаграмм вертикально для устранения наезда подписей.
- **2026-05-18, поздний вечер** — добавление 8-й методики (Керимов 1992),
  инфраструктура `t_ref` + `eps_t`, повторный аудит, удаление целевой
  линии 0,8·L с диаграммы.
