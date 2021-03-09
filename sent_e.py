import pandas as pd
import streamlit as st
import datetime
import re
import base64
import os
import altair as alt

current_path = os.getcwd()
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
    st.header('**Sentiment Board**')
    st.markdown('센티먼트 보드는 국내 증권사가 발간하는 기업분석 리포트의 텍스트를 분석해 국내 주식시장에 대한 애널리스트의 감성을 지수화한 자료입니다. 기업분석 리포트의 긍정어와 부정어/불확실성 출현 빈도를 파악해 감정을 지수화했습니다. 분기 실적 발표 시즌에 따른 계절성을 감안해 발간건수를 가중한 4개월 이동평균을 사용하여 국내 주식시장의 업종별 감성지수가 산출됩니다. 해당 자료는 전체 리포트가 아닌 시장에 공개되어 있는 리포트의 요약 텍스트를 대상으로 하며, 통계적인 분석을 통해 추출된 감성 정보만을 제공합니다. ')
    filtered_df = df_filter('기간을 선택하세요',df)
    column_1, column_2 = st.beta_columns(2)

    with column_1:
        st.subheader('**Sentiment Data**')
        short = pd.DataFrame(filtered_df, columns=['Date', 'KOSPI', '변동성(우)', '감성지수(좌)'])
        short.columns = ['Date', 'KOSPI', 'VKOSPI', 'Sentiment']
        st.write(short.iloc[:,0:4])
    with column_2:
        st.subheader('**Sentiment Chart**')
        st.line_chart(short["Sentiment"])
    st.markdown(download_csv('Filtered Data Frame',filtered_df),unsafe_allow_html=True)

#=========================================================

    st.subheader('**Sentiment and KOSPI**')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x='Date', y='감성지수(우)')
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x='Date', y='KOSPI', color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red'), use_container_width=True)

    st.subheader('**Sentiment and VKOSPI**')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x='Date', y='감성지수(우)')
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x='Date', y='변동성(우)', color=alt.value('red')
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red'), use_container_width=True)

