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

def download_csv(name, df):
    csv = df.to_csv(index=False)
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    return file

def df_filter(message, df):
    slider_1, slider_2 = st.slider('%s' % (message), 0, len(df) - 1, [len(df)-30, len(df) - 1], 1)
    start_date = datetime.datetime.strptime(str(df.iloc[slider_1][0]).replace('.0', ''), '%Y-%m-%d')
    start_date = start_date.strftime('%d %b %Y')
    end_date = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0', ''), '%Y-%m-%d')
    end_date = end_date.strftime('%d %b %Y')

    st.info('Start: **%s** End: **%s**' % (start_date, end_date))
    filtered_df = df.iloc[slider_1:slider_2 + 1][:].reset_index(drop=True)
    return filtered_df

if __name__ == '__main__':

    df = pd.read_csv(current_path + file_sent)
    df = df.replace(',', '', regex=True)        # remove comma
    df["KOSPI"]=pd.to_numeric(df["KOSPI"])      # series into numeric

    st.sidebar.markdown('**Sentiment Dashborad**')
    st.sidebar.markdown(''' 
    This app is to give insights about Sentiment in Korea Stock Market.
    
    The data considerd for this analysis for ... starting from 01-02-2020 to 30-11-2020
    Select the different options to vary the Visualization
    
    All the Charts are interactive. 
    
    Scroll the mouse over the Charts to feel the interactive features like Tool tip, Zoom, Pan

    Analysis by: **Hyeon Jong**  ''')

    st.header('**Sentiment Board**')
    st.markdown('센티먼트 보드는 국내 증권사가 발간하는 기업분석 리포트의 텍스트를 분석해 국내 주식시장에 대한 애널리스트의 감성을 지수화한 자료입니다. 기업분석 리포트의 긍정어와 부정어/불확실성 출현 빈도를 파악해 감정을 지수화했습니다. 분기 실적 발표 시즌에 따른 계절성을 감안해 발간건수를 가중한 4개월 이동평균을 사용하여 국내 주식시장의 업종별 감성지수가 산출됩니다. 해당 자료는 전체 리포트가 아닌 시장에 공개되어 있는 리포트의 요약 텍스트를 대상으로 하며, 통계적인 분석을 통해 추출된 감성 정보만을 제공합니다. ')
    filtered_df = df_filter('기간을 선택하세요',df)
    column_1, column_2 = st.beta_columns(2)

    with column_1:
        st.subheader('**Sentiment Data**')
        short = pd.DataFrame(filtered_df, columns=['Date', 'KOSPI', 'VKOSPI', '감성지수'])
        st.write(short.iloc[:,0:4])
    with column_2:
        st.subheader('**Sentiment Chart**')
        st.line_chart(short["감성지수"])
    st.markdown(download_csv('Filtered Data Frame',filtered_df),unsafe_allow_html=True)

    st.subheader('**긍정어와 부정어 빈도**')
    a = alt.Chart(filtered_df).encode(x='Date', y='긍정적')
    b = alt.Chart(filtered_df).encode(x='Date', y='부정적', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)

    st.subheader('**감성지수와 KOSPI**')
    a = alt.Chart(filtered_df).encode(x='Date', y='감성지수')
    b = alt.Chart(filtered_df).encode(x='Date', y='KOSPI',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)

    st.subheader('**감성지수와 VKOSPI**')
    a = alt.Chart(filtered_df).encode(x='Date', y='감성지수')
    b = alt.Chart(filtered_df).encode(x='Date', y='VKOSPI',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)

    # Sector Sentiment Index
    st.subheader('Search Sector')
    tickerSymbol = st.selectbox('Sector', df.columns[6:])
    chart= alt.Chart(filtered_df).encode(x='Date', y=tickerSymbol)
    st.altair_chart(chart.mark_line(),use_container_width=True)

    # Heatmap
    st.subheader('Industry Sentiment Heatmap')
    st.markdown('국내 주식시장을 구성하는 개별 업종의 감성지수를 통해 감성의 변화를 확인합니다. 업종 리포트에 나타난 감성이 부정적일수록 밝게, 긍정적일수록 어둡게 표시됩니다. 감성지수는 시장과열 또는 쏠림(crowdedness)에 대한 지표로 해석하는 것이 가능합니다. ')

    heat = pd.read_csv(current_path + file_heat)
    heat=pd.DataFrame(heat.iloc[1:,:])
    heat.index=heat["Industry"]
    heat=heat.iloc[:,1:]
    heat = heat.replace(',', '', regex=True)  # remove comma

    stacked = heat.stack()   # stack from dataframe into Series
    indices = pd.MultiIndex.from_tuples(stacked.index,names=['industry','time'])
    df_stacked = pd.DataFrame(stacked,index=indices)
    df_stacked = df_stacked.reset_index()
    df_stacked.columns = ["industry", "time", "sentiment"]

    num = np.arange(1, len(df_stacked["time"]))
    freq = np.percentile(num, np.arange(0, 11) * 10, interpolation='nearest')
    times = df_stacked["time"].values[freq]

    r = alt.Chart(df_stacked).mark_rect().encode(
        alt.X('time', axis=alt.Axis(values=
                                    ['2008-05-30',
                                     '2009-08-31',
                                     '2010-12-30',
                                     '2012-03-30',
                                     '2013-06-28',
                                     '2014-10-31',
                                     '2016-01-29',
                                     '2017-04-28',
                                     '2018-07-31',
                                     '2019-11-29',
                                     '2021-02-26']
                                    )),
        alt.Y('industry:O'),
        alt.Color('sentiment:Q', scale=alt.Scale(scheme='oranges'))
    ).properties(height=300, width=800)
    st.altair_chart(r)

    # check why times(list) does not work
    
    # r = alt.Chart(df_stacked).mark_rect().encode(
    #    alt.X('time', axis=alt.Axis(values=times)),
    #    alt.Y('industry:O'),
    #    alt.Color('sentiment:Q', scale=alt.Scale(scheme='oranges'))
    # ).properties(height=300, width=800)
    # st.altair_chart(r)

    # industry comparison
    st.subheader('Industry Comparison')
    current = heat.iloc[:,len(heat.columns)-36:len(heat.columns)]
    now = current.iloc[:,-1:].squeeze() # dataframe into series
    mean = current.mean(axis=1)  # row means
    stdzed = pd.DataFrame(pd.concat([pd.Series(current.index, index=current.index),mean,now],axis=1))
    stdzed.columns = ["Industry","3년 평균","현재"]

    a = alt.Chart(stdzed).encode(x='Industry', y=alt.Y('3년 평균',scale=alt.Scale(domain=(0, 70))))
    b = alt.Chart(stdzed).encode(x='Industry', y=alt.Y('현재',scale=alt.Scale(domain=(0, 70))), color=alt.value('red'))
    st.altair_chart((a.mark_bar() + b.mark_circle(size=60)).resolve_scale(y='independent').configure_axisRight(labelColor='red', titleColor='red'), use_container_width=True)
