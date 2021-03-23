import pandas as pd
import streamlit as st
import datetime
from datetime import timedelta
import re
import base64
import os
import altair as alt
import numpy as np
from sklearn.linear_model import LinearRegression

# current_path = os.getcwd()
file_rai = 'https://raw.github.com/HYEONJONG/98_app_building/master/raw_rai_summary.csv'
file_sent = 'https://raw.github.com/HYEONJONG/98_app_building/master/raw_sentiment.csv'
file_heat = 'https://raw.github.com/HYEONJONG/98_app_building/master/raw_heatmap.csv'
file_index = 'https://raw.github.com/HYEONJONG/98_app_building/master/raw_riskboard_index.csv'
file_table = 'https://raw.github.com/HYEONJONG/98_app_building/master/raw_riskboard_table.csv'

def download_csv(name, df):
    csv = short.to_csv(index=False)
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    return file

def df_filter(message, df):
    slider_1, slider_2 = st.slider('%s' % (message), 0, len(df) - 1, [len(df) - 150, len(df) - 1], 1)
    start_date = datetime.datetime.strptime(str(df.iloc[slider_1][0]).replace('.0', ''), '%Y-%m-%d')
    start_date = start_date.strftime('%d %b %Y')
    end_date = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0', ''), '%Y-%m-%d')
    end_date = end_date.strftime('%d %b %Y')
    st.info('Start: **%s** End: **%s**' % (start_date, end_date))
    filtered_df = df.iloc[slider_1:slider_2 + 1][:].reset_index(drop=True)
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    return filtered_df

#====================================
# Side bar Selection
#====================================

st.sidebar.title('**Investment Solution**')
st.sidebar.text("")
option = st.sidebar.radio("Select the option", ('Global Risk Board', 'Sentiment Board','Risk Appetite Index'))
st.sidebar.text("")
st.sidebar.text("")

#====================================
# Global Risk Board
#====================================

if option == 'Global Risk Board':
    st.sidebar.markdown('**Methodology**')
    st.sidebar.markdown('매크로, 주식/금리/FX에 대한 리스크 요인을 감안해 자산시장의 전반적인 위기수준을 판단하기 위한 리스크 종합지표(RCI: Risk Composite Index). RCI는 직전 3년 기간 대비 리스크 요인의 percent rank 변화를 확인해 현재 시점이 어느 수준인지를 측정. 0~1 사이에서 움직이며 0.5가 평균. RCI가 높아질 수록 위기에 가까워짐을 의미')
    st.sidebar.text("")
    st.sidebar.text("")

    df = pd.read_csv(file_index)
    df.columns = ['Date', '단기 매크로리스크', '장기 매크로리스크', 'EM 매크로리스크',
    'G7 OECD 경기선행지수',
    'KR 선행지수 순환변동치', 'KR 광공업 생산(YoY %)', 'KR 수출 증감률(YoY %)', 'KR 소매 판매(YoY %)',
    'KR 소비자 심리지수', '미 장단기 금리 스프레드', '달러인덱스',
    'CVIX', '신흥국 국채 스프레드', '원달러 환율',
    'VIX', 'VKOSPI', 'CDS 프리미엄', '외국인 순매수',
    'DM turbulence 지수', 'EM Turbulence 지수', 'KR Turbulence 지수', 'KR Systemic risk',
    'GRCI', 'KRCI', 'GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면','MSCI ACWI','KOSPI']

    df = df.iloc[157:,:]  # since 2005
    df = df.replace(',', '', regex=True)  # remove comma
    df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']] = df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']].replace("  ", 0)
    df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']] = df[['GRCI_안정국면', 'GRCI_위기국면', 'KRCI_안정국면', 'KRCI_위기국면']].replace("1",1)

    st.header('Global Risk Board')
    st.markdown('리스크 스코어보드는 국내와 글로벌 자산시장의 전반적인 위기수준을 나타냅니다. 매크로와 함께 주식, 금리, 외환시장 등 22개 세부 리스크 요인을 확인할 수 있도록 만든 스코어보드를 작성하고, 세부 리스크 요인을 종합해 산출한 리스크 종합지수를 통해 전반적인 위기 수준을 점검합니다. 글로벌과 국내 리스크를 구분해 GRCI와 KRCI를 주간 단위로 제시하고 있습니다.')
    opt = st.radio("Select the option", ('GRCI (Global Risk Composite Index)', 'KRCI (Korea Risk Composite Index)'))
    filtered_df = df_filter('기간을 선택하세요', df)
    st.subheader(f'1. {opt} 그래프')
    st.markdown('리스크종합지표는 직전 3년 기간 대비 현재 리스크 수준을 측정합니다. 현재 시점이 0~1 사이에서 어느 수준인지를 측정하며 높아질수록 위기에 가까워짐을 의미합니다.')

    short = pd.DataFrame(filtered_df[["Date", "GRCI", "KRCI"]])
    glo = alt.Chart(short).encode(
            x=alt.X('Date', title = None), y="GRCI", tooltip=["Date", "GRCI"]).interactive()

    kor = alt.Chart(short).encode(
            x=alt.X('Date', title = None), y="KRCI", tooltip=["Date", "KRCI"]).interactive()

    if opt == 'GRCI (Global Risk Composite Index)':
        st.altair_chart(glo.mark_line(), use_container_width=True)
    elif opt == 'KRCI (Korea Risk Composite Index)':
        st.altair_chart(kor.mark_line(),use_container_width=True)

    st.markdown(download_csv('Filtered Data Frame', filtered_df), unsafe_allow_html=True)

    # Sub risk Index
    st.subheader('2. 세부 리스크 지표')
    tickerSymbol = st.selectbox('리스크 지표를 선택하세요', df.columns[1:23])
    # dfs = pd.read_csv(current_path + file_table)
    dfs = pd.read_csv(file_table)
    dfs.columns = ['index', 'description']
    txt = dfs[dfs['index'] == tickerSymbol]['description']
    st.markdown(", ".join(txt)) # into String

    chart = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y=tickerSymbol).interactive()
    st.altair_chart(chart.mark_line(), use_container_width=True)

    # Charts
    a = alt.Chart(filtered_df).mark_line().encode(x=alt.X('Date', title = None), y='MSCI ACWI').interactive()
    b = alt.Chart(filtered_df).mark_line().encode(x=alt.X('Date', title = None), y='GRCI', color=alt.value('red')).interactive()
    c = alt.Chart(filtered_df).mark_line().encode(x=alt.X('Date', title = None), y='KOSPI').interactive()
    d = alt.Chart(filtered_df).mark_line().encode(x=alt.X('Date', title = None), y='KRCI', color=alt.value('red')).interactive()
    e = alt.Chart(filtered_df).mark_area(opacity=0.5, color="grey").encode(x=alt.X('Date', title = None), y=alt.Y('GRCI_안정국면', title = None)).interactive()
    f = alt.Chart(filtered_df).mark_area(opacity=0.5, color="red").encode(x=alt.X('Date', title = None), y=alt.Y('GRCI_위기국면', title = None), color=alt.value('red')).interactive()
    g = alt.Chart(filtered_df).mark_area(opacity=0.5, color="grey").encode(x=alt.X('Date', title = None), y=alt.Y('KRCI_안정국면', title = None)).interactive()
    h = alt.Chart(filtered_df).mark_area(opacity=0.5, color="red").encode(x=alt.X('Date', title = None), y=alt.Y('KRCI_위기국면', title = None), scale=alt.Scale(domain=(0, 1)), color=alt.value('red')).interactive()

    # Equity vs RCI
    st.subheader(f'3. {opt}와 주가지수 추이')
    if opt == 'GRCI (Global Risk Composite Index)':
        st.altair_chart((a + b).resolve_scale(y='independent').
                        configure_axisRight(labelColor='red', titleColor='red').
                        configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),
                        use_container_width=True)
    elif opt == 'KRCI (Korea Risk Composite Index)':
        st.altair_chart((c + d).resolve_scale(y='independent').
                        configure_axisRight(labelColor='red', titleColor='red').
                        configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),
                        use_container_width=True)

    # Regime vs RCI
    st.subheader(f'4. {opt}와 리스크 국면')
    st.markdown('통계적인 확률 국면분석 모형(hiddem Markov model)을 활용하여 리스크에 대한 국면을 구분합니다. RCI의 변화에 따라 안정(회색), 전환(흰색), 위기(빨강)의 3국면을 구분하여 나타냅니다')

    if opt == 'GRCI (Global Risk Composite Index)':
        st.altair_chart(b + e + f, use_container_width=True)
    elif opt == 'KRCI (Korea Risk Composite Index)':
        st.altair_chart(d + g + h, use_container_width=True)

    # Risk Percentile Change
    st.subheader('5. Risk Percentile Change')

    def df_filter2(message, df):
        slider_2 = st.slider('%s' % (message), 0, len(df) - 1, len(df) - 1)
        NOW = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0', ''), '%Y-%m-%d')
        NOW = NOW.strftime('%d %b %Y')
        st.info('관찰시점: **%s**' % (NOW))
        filtered_df2 = df.iloc[(slider_2 + 1 -3 * 52):slider_2 + 1][:].reset_index(drop=True)
        return filtered_df2

    filtered_df2 = df_filter2('시점을 선택하면 세부 리스크의 변화를 확인할 수 있습니다. 직전 3년 기간 동안 개별 지표의 percent rank를 나타냅니다.', df)
    perc = filtered_df2.iloc[:,1:25].rank(pct = True).tail(1)*100
    perc = perc.T
    perc["var"]=perc.index
    perc["avg"] = 50
    perc.columns = ["percentile", "variable","average"]
    perc = perc.sort_values("percentile", ascending=False)

    d = alt.Chart(perc).mark_bar().encode(
        x='percentile', y=alt.Y('variable', sort='-x', title = None),
        color = alt.condition(
        alt.datum.percentile > 50,
        alt.value("orange"),
        alt.value("steelblue")))
    st.altair_chart(d, use_container_width=True)

#====================================
# Sentiment Board
#====================================

if option == 'Sentiment Board':
    st.sidebar.markdown('**Methodology**')
    st.sidebar.markdown('2008년 이후 매월마다 국내 증권사의 전체 기업분석 리포트의 텍스트를 분석해 여기에 사용된 단어의 감성을 긍정/부정/중립/불확실성의 4가지 카테고리로 구분합니다. 실적시즌에 따른 계절성을 감안해 발간건수를 가중한 4개월 이동평균을 사용해 감성지수를 산출합니다.')
    st.sidebar.text("")
    st.sidebar.text("")

    # df = pd.read_csv(current_path + file_sent)
    df = pd.read_csv(file_sent)
    df = df.replace(',', '', regex=True)  # remove comma
    df["KOSPI"] = pd.to_numeric(df["KOSPI"])  # series into numeric

    st.header('Sentiment Board')
    st.markdown(
        '센티먼트 보드는 국내 증권사가 발간하는 기업분석 리포트의 텍스트를 분석해 국내 주식시장에 대한 애널리스트의 감성을 지수화한 자료입니다. 증권사의 기업분석 리포트는 표준화된 형식을 가지고 있으며 문법적 완성도가 높아 데이터 분석이 용이합니다. 기업분석 리포트를 자연어 처리해 정형화하고 시계열로 축적해 주식시장 움직임을 예측하는 정보로 활용합니다.')
    st.markdown(
        '기업분석 리포트의 긍정어와 부정어/불확실성 출현 빈도를 파악해 감정을 지수화했습니다. 분기 실적 발표 시즌에 따른 계절성을 감안해 발간건수를 가중한 4개월 이동평균을 사용하여 국내 주식시장의 업종별 감성지수가 산출됩니다. 해당 자료는 시장에 공개되어 있는 리포트의 요약 텍스트를 대상으로 하며, 통계적인 분석을 통해 추출된 감성 정보를 제공합니다.')
    filtered_df = df_filter('기간을 선택하세요', df)
    short = pd.DataFrame(filtered_df, columns=['Date', 'KOSPI', 'VKOSPI', '감성지수'])

    st.subheader('1. 감성지수 그래프')
    st.markdown(
        '기업분석 리포트는 개별 기업의 재무상태를 분석하고 증권에 대한 가치평가를 통해 투자전망을 제시하는 역할을 합니다. 애널리스트는 리포트를 통해 시장참여자의 기대를 변화시키거나 특정 방향으로 투자행위를 유도하는 영향을 미칩니다. 기업분석 리포트 텍스트를 통해 투자의견, 목표주가 외에 추가적인 정보를 확인하는 것이 가능합니다.')

    chart = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='감성지수')
    st.altair_chart(chart.mark_line(), use_container_width=True)
    st.markdown(download_csv('Filtered Data Frame', filtered_df), unsafe_allow_html=True)

    st.subheader('2. 긍정어와 부정어 빈도')
    st.markdown('일반적인 한글 감성어 사전과 기업분석 리포트의 감성어 사전은 다릅니다. 애널리스트가 감성어를 중립적, 긍정적, 부정적, 불확실성의 4가지 카테고리로 구분해 감성어 사전을 구축합니다. 기업분석 리포트의 긍정어와 부정어(또는 불확실)의 차이를 통해 감성 또는 어조의 변화를 확인합니다. 많은 경우 긍정적인 어조를 사용하지만, 불확실성이나 부정적 이베트 시에 사용되는 텍스트가 변화가 나타납니다.')

    a = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='긍정적')
    b = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='부정적', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red', titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),
                    use_container_width=True)

    st.subheader('3. 감성지수와 KOSPI')
    a = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='감성지수')
    b = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='KOSPI', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red',titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),
                    use_container_width=True)

    st.subheader('4. 감성지수와 VKOSPI')
    st.markdown('감성지수는 주식시장의 기대변동성 지표와 역의 관계가 존재합니다. 애널리스트 실적전망은 추세적인데 반해, 애널리스트의 감성은 시장의 센티멘트에 영향을 주고 받기 때문에 민감하게 반응합니다. ')
    a = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='감성지수')
    b = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y='VKOSPI', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red', titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),
                    use_container_width=True)

    # Sector Sentiment Index
    st.subheader('5. 업종별 감성지수 추이')
    tickerSymbol = st.selectbox('업종을 선택하세요', df.columns[6:])
    chart = alt.Chart(filtered_df).encode(x=alt.X('Date', title = None), y=tickerSymbol)
    st.altair_chart(chart.mark_line(), use_container_width=True)

    # Heatmap
    st.subheader('6. 업종별 감성지수 히트맵')
    st.markdown(
        '국내 주식시장을 구성하는 개별 업종의 감성지수를 통해 감성의 변화를 확인합니다. 업종 리포트에 나타난 감성이 부정적일수록 밝게, 긍정적일수록 어둡게 표시됩니다. 감성지수는 시장과열 또는 쏠림(crowdedness)에 대한 지표로 해석하는 것이 가능합니다. ')

    # heat = pd.read_csv(current_path + file_heat)
    heat = pd.read_csv(file_heat)
    heat = pd.DataFrame(heat.iloc[1:, :])
    heat.index = heat["Industry"]
    heat = heat.iloc[:, 1:]
    heat = heat.replace(',', '', regex=True)  # remove comma

    stacked = heat.stack()  # stack from dataframe into Series
    indices = pd.MultiIndex.from_tuples(stacked.index, names=['industry', 'time'])
    df_stacked = pd.DataFrame(stacked, index=indices)
    df_stacked = df_stacked.reset_index()
    df_stacked["time"] = [datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%y/%m') for date in df_stacked["time"]]
    df_stacked.columns = ["industry", "time", "sentiment"]
    num = np.arange(1, len(heat.columns))
    freq = np.percentile(num, np.arange(0, 11) * 10, interpolation='nearest')
    times = df_stacked["time"].values[freq]

    r = alt.Chart(df_stacked).mark_rect().encode(
        alt.X('time', title = None, axis=alt.Axis(values=times)),
        alt.Y('industry', title = None),
        alt.Color('sentiment', scale=alt.Scale(scheme='oranges'))
    ).properties(height=400, width=800)
    st.altair_chart(r)

    # industry comparison
    st.subheader('7. 업종별 감성지수 비교')
    st.markdown(
        '국내 주식시장 업종별로 감성지수를 비교하여 나타냅니다. 또한 개별 업종의 감성지수가 직전 3년 평균과 비교하여 어느 수준인지를 나타냅니다.')
    current = heat.iloc[:, len(heat.columns) - 36:len(heat.columns)]
    now = current.iloc[:, -1:].squeeze()  # dataframe into series
    mean = current.mean(axis=1)  # row means
    stdzed = pd.DataFrame(pd.concat([pd.Series(current.index, index=current.index), mean, now], axis=1))
    stdzed.columns = ["Industry", "3년 평균", "현재"]

    a = alt.Chart(stdzed).encode(x=alt.X('Industry', title = None), y=alt.Y('3년 평균', scale=alt.Scale(domain=(0, 70))))
    b = alt.Chart(stdzed).encode(x=alt.X('Industry', title = None), y=alt.Y('현재', scale=alt.Scale(domain=(0, 70))), color=alt.value('red'))
    st.altair_chart(
        (a.mark_bar() + b.mark_circle(size=60)).resolve_scale(y='independent').
            configure_axisRight(labelColor='red',titleColor='red').
            configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4').properties(height=400, width=750))

#====================================
# Risk Appetite Index
#====================================

if option == 'Risk Appetite Index':
    st.sidebar.markdown('**Methodology**')
    st.sidebar.markdown('''글로벌 주식, 채권, 원자재, 현금 등 11개 자산군의 실현 수익률과 위험을 통해 산출한 CML의 기울기 확인. 자산군의 수익률은 직전 6개월, 변동성은 직전 5년 데이터를 사용.
                        Min-max normalizatoin을 통해 0~100으로 스케일링 (0에 가까울수록 fear 영역, 100에 가까울수록 greed영역으로 해석''')
    st.sidebar.text("")
    st.sidebar.text("")

    # df = pd.read_csv(current_path + file_rai)
    df = pd.read_csv(file_rai)
    st.header('Risk Appetite Index')
    st.markdown(
        '투자자들은 동일 리스크에 대해 서로 다르게 인식하거나 다양한 태로를 취합니다. 글로벌 위험선호지수(RAI)란 자산군의 위험조정성과 변화를 통해 투자자들의 위험 선호도를 측정한 지수입니다. 글로벌 주식, 채권, 원자재, 현금 등의 자산군에서 추출한 리스크 프리미엄의 변동을 확인해 투자자들의 위험선호가 어떻게 변화되는지 확인합니다. ')
    st.markdown(
        '글로벌 위험선호지수의 변화를 통해 투자자들의 공포(fear)와 탐욕(greed)의 정도를 판단합니다. RAI가 극단값에 가까울수록 수익률 반전 가능성이 높아질 수 있어 contrarian 지표로 활용합니다. 위험선호도 외에 펀더멘털 요인이 RAI에 영향을 미칠 수 있어 실제 투자의사결정 시에는 경험과 판단 등 정성적 정보를 함께 고려하는 것이 바람직합니다.')
    filtered_df = df_filter('기간을 선택하세요', df)
    short = filtered_df.iloc[:, 0:5]
    short.columns = ['Date', 'coef', 'conf_L', 'conf_H', 'RAI']
    st.subheader('1. CML 그래프 기울기')
    st.markdown(
        '여기서는 여러 자산군에서 추정한 CML(자본시장선)을 통해 투자자들의 리스크 선호/회피도를 측정합니다. 위험과 초과수익률의 cross-sectional 분석을 통해 선형회귀식의 기울기를 계산하고, 이를 통해 투자자들의 위험자산에 대한 위험선호의 변화를 시계열로 분석합니다. 리스크에 대한 실현된 보상이 너무 크다면 greed 영역, 실현된 보상이 너무 작다면 fear 영역으로 해석합니다.')

    h = alt.Chart(short).encode(x=alt.X('Date', title=None), y=alt.Y('coef', title = "점선: 95% 신뢰구간"))
    i = alt.Chart(short).encode(x=alt.X('Date', title=None), y=alt.Y('conf_L'))
    j = alt.Chart(short).encode(x=alt.X('Date', title=None), y=alt.Y('conf_H'))

    st.altair_chart(h.mark_line() + i.mark_line(strokeDash=[3,3]) + j.mark_line(strokeDash=[3,3]), use_container_width=True)
    st.markdown(download_csv('Filtered Data Frame', filtered_df), unsafe_allow_html=True)

    st.subheader('2. 글로벌 투자자산군 특성치와 CML')
    st.markdown('대부분의 투자자들은 위험에 대해 회피성향을 가지며 효율적인 시장 하에서 추가적인 리스크에 대한 보상인 리스크 프리미엄은 일관되게 우상향하는 자본시장선(CML)을 형성합니다. 그러나 효율적인 시장이라면 어떤 특정한 자산의 높은 위험조정성과는 오래 지속되기 힘들며 자산가격 상승압력을 받아 향후 기대수익률이 낮아지게 됩니다.')
    asset = ["GLOBAL E", "DM E", "EM E", "KOSPI", "US_TRSY", "EM_TRSY", "COMDTY", "GOLD", "HY", "CREDIT", "REIT"]
    assets = pd.DataFrame(asset);
    assets.index = asset
    returns = pd.DataFrame(filtered_df.iloc[len(filtered_df) - 1, 29:40])
    returns.index = asset
    risk = pd.DataFrame(filtered_df.iloc[len(filtered_df) - 1, 41:52])
    risk.index = asset
    profile = pd.concat([assets, risk, returns], axis=1)
    profile.columns = ["asset", "risk(%)", "return(%)"]
    X = profile["risk(%)"];
    y = profile["return(%)"]
    line_fitter = LinearRegression()
    line_fitter.fit(X.values.reshape(-1, 1), y)
    profile["fitted"] = line_fitter.intercept_ + line_fitter.coef_ * X

    a = alt.Chart(profile).mark_area(opacity=1).\
        encode(x='risk(%)', y='return(%)',tooltip=['risk(%)', 'return(%)', 'asset'])
    b = alt.Chart(profile).mark_area(opacity=0.6).\
        encode(x=alt.X('risk(%)', title='Risk(%)'), y=alt.Y('fitted', title='Return(%)'), color=alt.value('red'))
    st.altair_chart((a.mark_circle(size=60) + b.mark_line()), use_container_width=True)

    st.subheader('3. RAI와 MSCI ACWI')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x=alt.X('Date', title = None), y=alt.Y('MIN-MAX norm', title='RAI'))
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x=alt.X('Date', title = None), y='MSCI AC World', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red', titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),use_container_width=True)

    st.subheader('4. RAI와 VIX')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x=alt.X('Date', title = None), y=alt.Y('MIN-MAX norm', title='RAI'))
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x=alt.X('Date', title = None), y='CBOE VIX', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red', titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),use_container_width=True)

    st.subheader('5. RAI와 AAII Sentiment Bullish Index')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x=alt.X('Date', title = None), y=alt.Y('MIN-MAX norm', title='RAI'))
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).\
        encode(x='Date', y='AAII Sentiment Bullish',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').
                    configure_axisRight(labelColor='red', titleColor='red').
                    configure_axisLeft(labelColor='#1f77b4', titleColor='#1f77b4'),use_container_width=True)
    st.markdown('Note: AAII Survey: Americal Association of Individual Investors, Sentiment Survey, Bullish 지수')

#====================================
# Side bar Information
#====================================

st.sidebar.info(''' 
  This app is to give insights about Market risks, Sentiment and risk appetite in Global Equity Market.

  The data considerd for this analysis are from 2005 to 2020.
  Select the different options to vary the visualization.
  All the Charts are interactive. 

  Analysed and designed by: hyeonjong.jung@gmail.com
  ''')
