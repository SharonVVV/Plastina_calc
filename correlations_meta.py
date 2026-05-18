"""
Метаинформация о локальных UHF-корреляциях Nu_x для вертикальной плиты.

Все формулы — verbatim из первоисточников; сверены по PDF/сканам страниц.
См. /Users/SharonovVV/Downloads/convection_methods.md (rev.5, 2026-05-18).

Используется новой страницей «Задача №1: высота турбулентности» для:
  • подписей у цветных полос на пластине,
  • вывода формул в expander «Какие уравнения использованы»,
  • заполнения таблицы границ режимов.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrelationMeta:
    key: str               # совпадает с ключом в correlations_uhf.REGISTRY
    name_ru: str           # имя для таблиц и expander
    name_short: str        # очень короткое имя для подписи над узкой полосой
    full_ru: str           # развёрнутое имя
    source: str            # первоисточник (с страницей)

    # Параметр, по которому методика классифицирует режим: 'Ra', 'Gr*Pr', 'Ra*'.
    boundary_param: str

    # Численные границы режимов в терминах boundary_param.
    lam_min: Optional[float]
    lam_max: Optional[float]
    turb_min: Optional[float]
    turb_max: Optional[float]

    # LaTeX-формулы локального Nu_x (verbatim, в нотации источника).
    nu_lam_tex: Optional[str] = None
    nu_turb_tex: Optional[str] = None

    notes_ru: str = ''
    color: str = '#4f8edc'


_PALETTE = [
    '#0b3954',  # глубокий тёмно-синий — Керимов 1992 (родная методичка стенда)
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2',
]


META: dict[str, CorrelationMeta] = {
    'kerimov_1992': CorrelationMeta(
        key='kerimov_1992',
        name_ru='Керимов 1992',
        name_short='Керимов',
        full_ru='Керимов Р.В. — Лабораторная работа № 9 и 9а, МЭИ (1992), формула (2)',
        source='Керимов Р.В., Лабораторная работа № 9 и 9а по курсу '
               '«Тепломассообмен». Местная теплоотдача при свободном '
               'движении воздуха около вертикальной пластины. — М.: '
               'Изд-во МЭИ, 1992. — 10 с.; UHF, q_c = const',
        boundary_param='Gr·Pr',
        lam_min=1.0e3, lam_max=1.0e9,
        turb_min=6.0e10, turb_max=None,
        nu_lam_tex=(r'\mathrm{Nu}_{\text{ж},x} = 0{,}60\,'
                    r'(\mathrm{Gr}\,\mathrm{Pr})^{0{,}25}\,\varepsilon_t'),
        nu_turb_tex=(r'\mathrm{Nu}_{\text{ж},x} = 0{,}15\,'
                     r'(\mathrm{Gr}\,\mathrm{Pr})^{1/3}\,\varepsilon_t'),
        notes_ru='Оригинальная методичка стенда НИУ МЭИ. '
                 'Свойства воздуха берутся при t_ж (НЕ при плёночной — '
                 'единственная методика в наборе с таким выбором). '
                 'ε_t = (Pr_ж/Pr_c)^0,25 — опытная поправка на переменность '
                 'свойств жидкости. Транзитная зона 10⁹…6·10¹⁰ в источнике '
                 'не описана; здесь — log-интерполяция между значениями на '
                 'границах. Геометрия стенда: h = 1540 мм, b = 205 мм, '
                 'δ = 1,0 мм (нержавеющая сталь, прямой Джоулев нагрев).',
        color=_PALETTE[0],
    ),
    'ckv': CorrelationMeta(
        key='ckv',
        name_ru='ЦКВ 2008',
        name_short='ЦКВ',
        full_ru='Цветков Ф.Ф., Керимов Р.В., Величко В.И. — Задачник МЭИ (2008), стр. 34-35',
        source='ЦКВ 2008, формулы (3.4), (3.8); UHF, q_c = const',
        boundary_param='Ra',
        lam_min=1.0e4, lam_max=1.0e9,
        turb_min=1.0e12, turb_max=None,
        nu_lam_tex=r'\mathrm{Nu}_x = 0{,}563\,[\mathrm{Ra}_x\,\Phi(\mathrm{Pr})]^{1/4}',
        nu_turb_tex=r'\mathrm{Nu}_x = 0{,}15\,[\mathrm{Ra}_x\,\Phi(\mathrm{Pr})]^{1/3}',
        notes_ru='Φ(Pr) = [1 + (0,437/Pr)^(9/16)]^(−16/9); для воздуха Φ ≈ 0,363. '
                 'Транзитная зона 10⁹…10¹² в источнике не описана — '
                 'log-интерполяция между значениями на границах.',
        color=_PALETTE[1],
    ),
    'vliet': CorrelationMeta(
        key='vliet',
        name_ru='Vliet 1969',
        name_short='Vliet',
        full_ru='Vliet G.C., ASME J. Heat Transfer (1969), формулы (1), (2), стр. 511, 515',
        source='Vliet 1969',
        boundary_param='Gr*·Pr',
        lam_min=1.0e8, lam_max=1.3e13,
        turb_min=1.0e14, turb_max=1.0e16,
        nu_lam_tex=r'\mathrm{Nu}_x = 0{,}60\,(\mathrm{Gr}_x^{*}\,\mathrm{Pr})^{1/5}',
        nu_turb_tex=r'\mathrm{Nu}_x = 0{,}30\,(\mathrm{Gr}_x^{*}\,\mathrm{Pr})^{0{,}24}',
        notes_ru='Gr_x* = g·β·q_w·x⁴/(λ·ν²) — модифицированное число Грасгофа. '
                 'Диапазон ламинара — Fig. 3; диапазон турбулентного — Fig. 4. '
                 'Между 1,3·10¹³ и 10¹⁴ — переход (не описан явно).',
        color=_PALETTE[2],
    ),
    'leontiev': CorrelationMeta(
        key='leontiev',
        name_ru='Леонтьев 2018',
        name_short='Леонтьев',
        full_ru='Леонтьев А.И. (ред.), «Теория тепломассообмена», 3-е изд., 2018, стр. 307, 310',
        source='Брдлик (стр. 307) + Эккерт-Джексон (стр. 310); UHF',
        boundary_param='Ra',
        lam_min=None, lam_max=0.7e9,
        turb_min=0.7e9, turb_max=None,
        nu_lam_tex=(r'\mathrm{Nu}_x = 0{,}616\,\left[\dfrac{\mathrm{Pr}}'
                    r'{\mathrm{Pr}+0{,}8}\right]^{1/5}\,'
                    r'(\mathrm{Gr}_x^{*}\,\mathrm{Pr})^{1/5}'),
        nu_turb_tex=(r'\mathrm{Nu}_x = 0{,}0295\,\mathrm{Ra}_x^{2/5}\,'
                     r'\mathrm{Pr}^{1/15}\,(1+0{,}494\,\mathrm{Pr}^{2/3})^{-2/5}'),
        notes_ru='Ламинар — приближённое решение П.М. Брдлика методом '
                 'интегральных соотношений (UHF). Турбулент — полуэмпирический '
                 'метод Эккерта и Джексона. Граница перехода в источнике '
                 '(стр. 310): «при Ra > 0,7·10⁹ переход в турбулентное». '
                 'Гибридная классификация: формула ламинара выражена через '
                 'Gr_x*·Pr, критерий перехода — в обычном Ra_x (через ΔT). '
                 'Верхняя граница локальной формулы Эккерта-Джексона '
                 'в источнике явно не указана; парная средняя формула на '
                 'стр. 310 имеет диапазон 10⁹ < Ra_l < 10¹², но локально '
                 'это не фиксируется.',
        color=_PALETTE[3],
    ),
    'holman': CorrelationMeta(
        key='holman',
        name_ru='Holman 2010',
        name_short='Holman',
        full_ru='Holman J.P., Heat Transfer, 10th ed., 2010, формулы (7-31), (7-32), стр. 336',
        source='Holman 2010; UHF',
        boundary_param='Gr*·Pr',
        lam_min=1.0e5, lam_max=1.0e11,
        turb_min=2.0e13, turb_max=1.0e16,
        nu_lam_tex=r'\mathrm{Nu}_x = 0{,}60\,(\mathrm{Gr}_x^{*}\,\mathrm{Pr})^{1/5}',
        nu_turb_tex=r'\mathrm{Nu}_x = 0{,}17\,(\mathrm{Gr}_x^{*}\,\mathrm{Pr})^{1/4}',
        notes_ru='Коэф. 0,60 в ламинарной формуле совпадает с Vliet 1969 (та же '
                 'эмпирическая зависимость, ссылка Holman → Vliet, ref. 25). '
                 'Транзитная область: начало 3·10¹²…4·10¹³, конец 2·10¹³…10¹⁴ (стр. 336).',
        color=_PALETTE[4],
    ),
    'bejan_vliet_liu': CorrelationMeta(
        key='bejan_vliet_liu',
        name_ru='Bejan / Vliet-Liu',
        name_short='Bejan/VL',
        full_ru='Bejan A., Convection Heat Transfer, 4th ed., 2013, формулы (4.108), (4.109), стр. 204',
        source='Bejan 2013 через Vliet & Liu 1969; UHF',
        boundary_param='Ra*',
        lam_min=1.0e5, lam_max=1.0e13,
        turb_min=1.0e13, turb_max=1.0e16,
        nu_lam_tex=r'\mathrm{Nu}_y = 0{,}6\,\mathrm{Ra}_{*y}^{1/5}',
        nu_turb_tex=r'\mathrm{Nu}_y = 0{,}568\,\mathrm{Ra}_{*y}^{0{,}22}',
        notes_ru='Ra_*y = g·β·y⁴·q″ / (a·ν·k) — модифицированное число Рэлея у Bejan '
                 '(формула 4.70). Эквивалентно Gr_y*·Pr. Граница ламинар/турбулент — '
                 'строго на Ra_* = 10¹³ (стыковая, без транзитной зоны).',
        color=_PALETTE[5],
    ),
    'bejan_air': CorrelationMeta(
        key='bejan_air',
        name_ru='Bejan / для воздуха',
        name_short='Bejan/air',
        full_ru='Bejan A., 2013, формулы (4.110), (4.111), стр. 204 — специально для воздуха',
        source='Bejan 2013, Ref. 35 — рекомендация для воздуха',
        boundary_param='Ra*',
        lam_min=1.0e5, lam_max=1.0e13,
        turb_min=1.0e13, turb_max=1.0e16,
        nu_lam_tex=r'\mathrm{Nu}_y = 0{,}55\,\mathrm{Ra}_{*y}^{1/5}',
        nu_turb_tex=r'\mathrm{Nu}_y = 0{,}17\,\mathrm{Ra}_{*y}^{1/4}',
        notes_ru='Применима только для воздуха. Коэф. 0,17 в турбулентной формуле '
                 'идентичен Holman (7-32). Атрибуция в Bejan — Ref. 35 (требует '
                 'дополнительной сверки).',
        color=_PALETTE[6],
    ),
    'fujii_fujii': CorrelationMeta(
        key='fujii_fujii',
        name_ru='Fujii–Fujii',
        name_short='Fujii',
        full_ru='Fujii T. & Fujii M., Int. J. Heat Mass Transfer 19 (1976) — через Jiji 2009, стр. 275-277, формула (7.33)',
        source='Fujii-Fujii 1976; UHF, только ламинар',
        boundary_param='Gr*·Pr',
        lam_min=None, lam_max=1.0e13,
        turb_min=None, turb_max=None,
        nu_lam_tex=(r'\mathrm{Nu}_x = \left[\dfrac{\mathrm{Pr}}'
                    r'{4 + 9\,\mathrm{Pr}^{1/2} + 10\,\mathrm{Pr}}\right]^{1/5}'
                    r'\,(\mathrm{Pr}\,\mathrm{Gr}_x^{*})^{1/5}'),
        nu_turb_tex=None,
        notes_ru='Только ламинарный режим, 0,001 < Pr < 1000. Развёрнутая форма '
                 'результата (Jiji, Example 7.2 на стр. 277). '
                 'Верхняя граница применимости в источнике дана для UWT как '
                 'Ra_x < 10⁹; в UHF-форме (Gr_x*·Pr) это даёт оценку '
                 'Gr_x*·Pr ≲ Ra·Nu ≲ 10⁹·10⁴ = 10¹³ (Nu ~ 10⁴ в ламинаре '
                 'при Ra=10⁹). Выше — серая полоса «вне диапазона».',
        color=_PALETTE[7],
    ),
}


# Порядок отображения на диаграмме (слева направо). Керимов 1992 первым —
# это родная методика стенда МЭИ.
ORDER = ['kerimov_1992', 'ckv', 'vliet', 'leontiev', 'holman',
         'bejan_vliet_liu', 'bejan_air', 'fujii_fujii']


def all_keys() -> list[str]:
    return list(ORDER)


def get_meta(key: str) -> CorrelationMeta:
    return META[key]


def label_for(key: str) -> str:
    return META[key].name_ru
