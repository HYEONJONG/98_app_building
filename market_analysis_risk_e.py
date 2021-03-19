import pandas as pd
import streamlit as st
import datetime
from datetime import timedelta
import re
import base64
import os
import altair as alt
import numpy as np

current_path = os.getcwd()
file_rai = '\\datasets\\raw_rai_summary.csv'
file_sent = '\\datasets\\raw_sentiment.csv'
file_heat = '\\datasets\\raw_heatmap.csv'
file_index = '\\datasets\\raw_riskboard_index.csv'
file_table = '\\datasets\\raw_riskboard_table.csv'

def download_csv(name, df):
    csv = short.to_csv(index=False)
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    return file

def df_filter(message, df):
    slider_1, slider_2 = st.slider('%s' % (message), 0, len(df) - 1, [len(df) - 200, len(df) - 1], 1)
    start_date = datetime.datetime.strptime(str(df.iloc[slider_1][0]).replace('.0', ''), '%Y-%m-%d')
    start_date = start_date.strftime('%d %b %Y')
    end_date = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0', ''), '%Y-%m-%d')
    end_date = end_date.strftime('%d %b %Y')

    st.info('Start: **%s** End: **%s**' % (start_date, end_date))
    filtered_df = df.iloc[slider_1:slider_2 + 1][:].reset_index(drop=True)
    return filtered_df


if __name__ == '__main__':
    df = pd.read_csv(current_path + file_index)
    df.columns = ['Date', '단기 매크로리스크', '장기 매크로리스크', 'EM 매크로리스크',
    'G7 OECD 경기선행지수',
    'KR 선행지수 순환변동치', 'KR 광공업 생산(YOY %)', 'KR 수출 증감률(YOY %)', 'KR 소매 판매(YOY %)',
    'KR 소비자 심리지수', '미 장단기 금리 스프레드', '달러인덱스',
    'CVIX', '신흥국 국채 스프레드', '원달러 환율',
    'VIX', 'VKOSPI', 'CDS 프리미엄', '외국인 순매수',
    'DM turbulence 지수', 'EM Turbulence 지수', 'KR Turbulence 지수', 'KR Systemic risk',
    'GRCI', 'KRCI', 'GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면','MSCI ACWI','KOSPI']

    df = df.iloc[157:,:]  # since 2005
    df = df.replace(',', '', regex=True)  # remove comma

    df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']] = df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']].replace("  ", 0)
    df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']] = df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']].replace("1",1)

    st.sidebar.markdown('**Global Risk Board**')
    st.sidebar.markdown(''' 
    This app is to give insights about Sentiment in Korea Stock Market.

    The data considerd for this analysis for ... starting from 2005
    Select the different options to vary the Visualization

    All the Charts are interactive. 

    Scroll the mouse over the Charts to feel the interactive features like Tool tip, Zoom, Pan

    Analysis by: **Hyeonjong, Jung**  ''')

    st.header('Risk Board')
    st.markdown('리스크 스코어보드는 매크로, 주식/금리/FX와 같은 리스크 요인을 중심으로 자산시장의 전반적인 위기수준을 한눈에 파악하기 위한 자료입니다. 글로벌과 국내 리스크를 구분해 세부 리스크 요인의 수준을 주간 단위로 제시하고 있습니다. 리스크종합지표(RCI)는 직전 3년 기간 대비 현재 리스크 수준을 측정합니다. 현재 시점이 0~1 사이에서 어느 수준인지를 측정하며 높아질수록 위기에 가까워짐을 의미합니다. ')

    opt = st.radio("Select the option", ('Global', 'Korea'))

    filtered_df = df_filter('기간을 선택하세요', df)
    column_1, column_2 = st.beta_columns(2)

    with column_1:
        st.subheader(f'{opt} RCI Data')
        short = pd.DataFrame(filtered_df[["Date","GRCI","KRCI"]])
        st.write(short)
    with column_2:
        st.subheader(f'{opt} RCI Chart')

        glo = alt.Chart(short).encode(
            x="Date",
            y="GRCI",
            tooltip=["Date", "GRCI"]
        ).interactive()

        kor = alt.Chart(short).encode(
            x="Date",
            y="KRCI",
            tooltip=["Date", "KRCI"]
        ).interactive()

        if opt == 'Global':
            st.altair_chart(glo.mark_line(), use_container_width=True)
        elif opt == 'Korea':
            st.altair_chart(kor.mark_line(),use_container_width=True)

    st.markdown(download_csv('Filtered Data Frame', filtered_df), unsafe_allow_html=True)

    # Risk Percentile Change
    st.subheader('Risk Percentile Change')

    def df_filter2(message, df):
        slider_2 = st.slider('%s' % (message), 0, len(df) - 1, len(df) - 1)
        NOW = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0', ''), '%Y-%m-%d')
        NOW = NOW.strftime('%d %b %Y')
        st.info('Now: **%s**' % (NOW))
        filtered_df2 = df.iloc[(slider_2 + 1 -3 * 52):slider_2 + 1][:].reset_index(drop=True)
        return filtered_df2

    filtered_df2 = df_filter2('시점을 선택하세요', df)
    perc = filtered_df2.iloc[:,1:25].rank(pct = True).tail(1)
    perc = perc.T
    perc["var"]=perc.index
    perc["avg"] = .5
    perc.columns = ["percentile", "variable","average"]
    perc = perc.sort_values("percentile", ascending=False)

    d = alt.Chart(perc).mark_bar().encode(
        x='percentile', y=alt.Y('variable', sort='-x'),
        color = alt.condition(
        alt.datum.percentile > 0.5,
        alt.value("orange"),
        alt.value("steelblue")))
    st.altair_chart(d, use_container_width=True)
    
    # Sub risk Index
    st.subheader('세부 리스크 지표')
    tickerSymbol = st.selectbox('리스크 지표를 선택하세요', df.columns[1:23])
    chart = alt.Chart(filtered_df).encode(x='Date', y=tickerSymbol).interactive()
    st.altair_chart(chart.mark_line(), use_container_width=True)

    # Charts
    a = alt.Chart(filtered_df)\
        .mark_line()\
        .encode(x='Date', y='MSCI ACWI').interactive()
    b = alt.Chart(filtered_df)\
        .mark_line()\
        .encode(x='Date', y='GRCI', color=alt.value('red')).interactive()
    c = alt.Chart(filtered_df)\
        .mark_line()\
        .encode(x='Date', y='KOSPI').interactive()
    d = alt.Chart(filtered_df)\
        .mark_line()\
        .encode(x='Date', y='KRCI', color=alt.value('red')).interactive()
    e = alt.Chart(filtered_df)\
        .mark_area(opacity=0.5, color="grey")\
        .encode(x='Date', y='GRCI_안정국면').interactive()
    f = alt.Chart(filtered_df)\
        .mark_area(opacity=0.5, color="red")\
        .encode(x='Date', y='GRCI_위기국면', color=alt.value('red')).interactive()
    g = alt.Chart(filtered_df)\
        .mark_area(opacity=0.5, color="grey")\
        .encode(x='Date', y='KRCI_안정국면').interactive()
    h = alt.Chart(filtered_df)\
        .mark_area(opacity=0.5, color="red")\
        .encode(x='Date', y=alt.Y('KRCI_위기국면', scale=alt.Scale(domain=(0, 1))), color=alt.value('red')).interactive()

    # Equity vs RCI
    st.subheader(f'{opt} Equity vs RCI')
    if opt == 'Global':
        st.altair_chart((a + b).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)
    elif opt == 'Korea':
        st.altair_chart((c + d).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)

    # Regime vs RCI
    st.subheader(f'{opt} Risk regime and RCI')
    if opt == 'Global':
        st.altair_chart(b + e + f, use_container_width=True)
    elif opt == 'Korea':
        st.altair_chart(d + g + h, use_container_width=True)


