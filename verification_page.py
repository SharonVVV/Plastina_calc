"""
UI вкладки «Верификация» — сравнение корреляций с эталонной библиотекой ht.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from verification_ht import (
    verify_churchill_chu_vs_ht,
    verify_all_methods_vs_ht,
)
from verification_data import compare_coolprop_vs_gsssd
from formatting import _fc


METHOD_COLORS = {
    'Черчилль-Чу (наш)': '#2ca02c',
    'Керимов': '#1f77b4',
    'Кузнецов': '#ff7f0e',
    'Леонтьев': '#d62728',
    'Churchill-Ozoe': '#9467bd',
    'Vliet': '#8c564b',
    'Fujii': '#e377c2',
    'Исаченко': '#7f7f7f',
}


def _build_deviation_chart(df_all: pd.DataFrame) -> go.Figure:
    """График отклонений delta(%) vs Ra для всех методик."""
    fig = go.Figure()

    for method_name in df_all['Методика'].unique():
        sub = df_all[df_all['Методика'] == method_name].sort_values('Ra')
        color = METHOD_COLORS.get(method_name, 'gray')
        fig.add_trace(go.Scatter(
            x=sub['Ra'], y=sub['delta, %'],
            name=method_name,
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color=color, width=1.5),
            hovertemplate=(
                f'{method_name}<br>'
                'Ra = %{x:.2e}<br>'
                'delta = %{y:.1f}%<extra></extra>'
            ),
        ))

    # Нулевая линия
    fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
    # Полоса ±15%
    fig.add_hrect(y0=-15, y1=15, fillcolor='green', opacity=0.08,
                  line_width=0, annotation_text='±15%',
                  annotation_position='top left')

    fig.update_layout(
        title='Отклонение от эталона (ht, Churchill-Chu)',
        xaxis_title='Ra',
        yaxis_title='delta, %',
        xaxis_type='log',
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.5, xanchor='center'),
        separators=', ',
    )
    return fig


def _build_nu_comparison_chart(df_all: pd.DataFrame) -> go.Figure:
    """График Nu vs Ra для всех методик + эталон."""
    fig = go.Figure()

    # Эталон — жирная чёрная линия
    sub_ht = df_all[df_all['Методика'] == 'Черчилль-Чу (наш)'].sort_values('Ra')
    fig.add_trace(go.Scatter(
        x=sub_ht['Ra'], y=sub_ht['Nu (ht)'],
        name='Эталон (ht)',
        mode='lines', line=dict(color='black', width=3),
    ))

    for method_name in df_all['Методика'].unique():
        sub = df_all[df_all['Методика'] == method_name].sort_values('Ra')
        color = METHOD_COLORS.get(method_name, 'gray')
        fig.add_trace(go.Scatter(
            x=sub['Ra'], y=sub['Nu'],
            name=method_name,
            mode='lines',
            line=dict(color=color, width=1.5,
                      dash='dash' if method_name != 'Черчилль-Чу (наш)' else None),
        ))

    fig.update_layout(
        title='Nu vs Ra — все методики vs эталон (ht)',
        xaxis_title='Ra', yaxis_title='Nu',
        xaxis_type='log', yaxis_type='log',
        template='plotly_white', height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.5, xanchor='center'),
        separators=', ',
    )
    return fig


def render_verification_tab(params: dict) -> None:
    """Отрисовка вкладки верификации."""

    st.markdown("""
    Верификация реализованных корреляций через эталонную библиотеку
    **ht** (Heat Transfer, Python) и справочные данные **ГСССД 8-79**.
    """)

    # ── Параметры ──
    col1, col2, col3 = st.columns(3)
    with col1:
        T_inf = st.number_input('T_ж, °C', value=int(params['t_fluid_C']),
                                min_value=0, max_value=100, key='verif_tinf')
    with col2:
        dT = st.number_input('dT, °C', value=100,
                             min_value=10, max_value=500, step=10, key='verif_dt')
    with col3:
        P_Pa = st.number_input('P, Па', value=int(params['P_Pa']),
                               min_value=90000, max_value=110000, key='verif_p')

    do_verify = st.button('Запустить верификацию', type='primary',
                          use_container_width=True, key='btn_verify')

    if do_verify:
        with st.spinner('Верификация...'):
            df_cc = verify_churchill_chu_vs_ht(T_inf_C=float(T_inf), P_Pa=float(P_Pa))
            df_all = verify_all_methods_vs_ht(
                T_inf_C=float(T_inf), dT=float(dT), P_Pa=float(P_Pa),
            )
            df_gsssd = compare_coolprop_vs_gsssd(float(P_Pa))

        st.session_state['verif_cc'] = df_cc
        st.session_state['verif_all'] = df_all
        st.session_state['verif_gsssd'] = df_gsssd

    df_cc = st.session_state.get('verif_cc')
    df_all = st.session_state.get('verif_all')
    df_gsssd = st.session_state.get('verif_gsssd')

    if df_cc is None:
        return

    # ── 1. Churchill-Chu: точное совпадение ──
    st.markdown('---')
    st.markdown('#### 1. Черчилль-Чу: наша реализация vs ht')
    max_delta = df_cc['delta, %'].max()
    if max_delta < 0.01:
        st.success(f'Точное совпадение (макс. расхождение: {max_delta:.6f}%)')
    else:
        st.warning(f'Расхождение до {max_delta:.2f}%')

    with st.expander('Подробная таблица', expanded=False):
        st.dataframe(df_cc, use_container_width=True, hide_index=True)

    # ── 2. Все методики vs эталон ──
    st.markdown('#### 2. Все методики vs эталон (ht, Churchill-Chu)')
    st.caption(f'Условия: T_ж = {T_inf}°C, dT = {dT}°C, P = {P_Pa} Па')

    # Сводная таблица
    summary = df_all.groupby('Методика')['delta, %'].agg(['mean', 'min', 'max']).reset_index()
    summary.columns = ['Методика', 'Среднее δ, %', 'Мин δ, %', 'Макс δ, %']
    summary = summary.sort_values('Среднее δ, %', key=abs)

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("""
    **Пояснение отклонений:**
    - **Черчилль-Чу** = 0% — та же формула, что и эталон
    - **Керимов / Исаченко** > 0 — свойства при t_ж (завышают Nu за счёт меньшей вязкости)
    - **Churchill-Ozoe / Fujii** < 0 — **местные** ламинарные формулы (ожидаемо ниже **среднего** Nu)
    - **Леонтьев (турб.)** < 0 — формула Эккерта-Джексона, местный Nu
    - **Vliet** ≈ 0 — хорошее согласование (калиброван на воздухе)
    """)

    # Графики
    col1, col2 = st.columns(2)
    with col1:
        fig_dev = _build_deviation_chart(df_all)
        st.plotly_chart(fig_dev, use_container_width=True)
    with col2:
        fig_nu = _build_nu_comparison_chart(df_all)
        st.plotly_chart(fig_nu, use_container_width=True)

    # ── 3. Свойства воздуха ──
    st.markdown('#### 3. Свойства воздуха: CoolProp vs ГСССД 8-79')
    if df_gsssd is not None:
        # Макс. отклонение
        delta_col = df_gsssd['δ, %'].str.replace(',', '.').astype(float)
        max_prop_delta = delta_col.max()
        if max_prop_delta < 3.0:
            st.success(f'Макс. расхождение: {max_prop_delta:.2f}% (норма < 3%)')
        else:
            st.warning(f'Макс. расхождение: {max_prop_delta:.2f}%')

        with st.expander('Подробная таблица', expanded=False):
            st.dataframe(df_gsssd, use_container_width=True, hide_index=True)
