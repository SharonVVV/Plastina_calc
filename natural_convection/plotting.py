"""
Визуализация результатов сравнения методик (matplotlib).
"""

import matplotlib.pyplot as plt
import pandas as pd

# Названия методик на русском
METHOD_LABELS = {
    'leontiev': 'Леонтьев (Брдлик / Эккерт—Дж.)',
    'churchill_ozoe': 'Churchill & Ozoe (1973)',
    'vliet': 'Vliet (1969)',
    'fujii': 'Fujii & Fujii (1976)',
    'isachenko': 'Исаченко и др. (1981)',
}

METHOD_STYLES = {
    'leontiev': {'color': '#1f77b4', 'linestyle': '-', 'marker': None},
    'churchill_ozoe': {'color': '#ff7f0e', 'linestyle': '--', 'marker': None},
    'vliet': {'color': '#2ca02c', 'linestyle': '-', 'marker': None},
    'fujii': {'color': '#d62728', 'linestyle': '-.', 'marker': None},
    'isachenko': {'color': '#9467bd', 'linestyle': ':', 'marker': None},
}


def _plot_column(ax, df, y_col, y_label):
    """Вспомогательная: отрисовка одного столбца для всех методик."""
    for method_name in df['method'].unique():
        sub = df[df['method'] == method_name]
        style = METHOD_STYLES.get(method_name, {})
        label = METHOD_LABELS.get(method_name, method_name)
        ax.plot(sub['x'] * 1000, sub[y_col],
                label=label,
                color=style.get('color'),
                linestyle=style.get('linestyle', '-'),
                linewidth=1.5)
    ax.set_xlabel('x, мм')
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_Nu_vs_x(df, ax=None):
    """График Nu_x(x) для всех методик."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    _plot_column(ax, df, 'Nu_x', r'$Nu_x$')
    ax.set_title('Локальное число Нуссельта')
    return ax.figure


def plot_alpha_vs_x(df, ax=None):
    """График alpha_x(x) для всех методик."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    _plot_column(ax, df, 'alpha_x', r'$\alpha_x$, Вт/(м$^2$·К)')
    ax.set_title('Локальный коэффициент теплоотдачи')
    return ax.figure


def plot_T_wall_vs_x(df, ax=None):
    """График T_wall(x) для всех методик."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    _plot_column(ax, df, 'T_wall_C', r'$T_{ст}$, °C')
    ax.set_title('Температура стенки')
    return ax.figure


def plot_comparison_dashboard(df):
    """Панель 2x2: Nu(x), alpha(x), T_wall(x), Nu(Ra*)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение методик расчёта свободной конвекции (UHF)', fontsize=14)

    plot_Nu_vs_x(df, ax=axes[0, 0])
    plot_alpha_vs_x(df, ax=axes[0, 1])
    plot_T_wall_vs_x(df, ax=axes[1, 0])

    # Nu vs Ra* (log-log)
    ax = axes[1, 1]
    for method_name in df['method'].unique():
        sub = df[df['method'] == method_name]
        ra_col = 'Ra_x_star' if sub['Ra_x_star'].notna().any() else 'Ra_x'
        ra_vals = sub[ra_col].dropna()
        if len(ra_vals) == 0:
            continue
        style = METHOD_STYLES.get(method_name, {})
        label = METHOD_LABELS.get(method_name, method_name)
        ax.loglog(sub.loc[ra_vals.index, ra_col], sub.loc[ra_vals.index, 'Nu_x'],
                  label=label,
                  color=style.get('color'),
                  linestyle=style.get('linestyle', '-'),
                  linewidth=1.5)
    ax.set_xlabel(r'$Ra^*_x$ (или $Ra_x$)')
    ax.set_ylabel(r'$Nu_x$')
    ax.set_title(r'$Nu_x$ vs $Ra^*_x$ (log-log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig
