import pandas as pd
import streamlit as st
import datetime
import re
import base64
import os
import altair as alt
from sklearn.linear_model import LinearRegression

current_path = os.getcwd()
file_rai = '\\datasets\\raw_rai_summary.csv'

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

    df = pd.read_csv(current_path + file_rai)
    st.header('**Global Risk Appetite Index**')
    st.markdown('글로벌 위험선호지수(RAI)란 자산군의 위험조정성과 변화를 통해 투자자들의 위험 선호도를 측정한 지수입니다. 글로벌 주식, 채권, 원자재, 현금 등의 자산군에서 추출한 리스크 프리미엄의 변동을 확인해 투자자들의 위험선호가 어떻게 변화되는지 확인합니다. 글로벌 위험선호지수가 0에 가까울수록 fear 영역, 100에 가까울수록 greed 영역으로 판단합니다. RAI가 극단값에 가까울수록 수익률 반전 가능성이 높아질 수 있어 contrarian 지표로 활용합니다. 위험선호도 외에 펀더멘털 요인이 RAI에 영향을 미칠 수 있어 실제 투자의사결정 시에는 경험과 판단 등 정성적 정보를 함께 고려하는 것이 바람직합니다. ')
    filtered_df = df_filter('기간을 선택하세요',df)
    column_1, column_2 = st.beta_columns(2)

    with column_1:
        st.subheader('**RAI Data**')
        st.write(filtered_df.iloc[:,0:4])
    with column_2:
        st.subheader('**RAI Chart**')
        st.line_chart(filtered_df["coef"])
    st.markdown(download_csv('Filtered Data Frame',filtered_df),unsafe_allow_html=True)


    st.subheader('**Global Asset return & risk profile**')
    asset=["GLOBAL E", "DM E", "EM E", "KOSPI", "US_TRSY", "EM_TRSY", "COMDTY", "GOLD", "HY", "CREDIT", "REIT"]
    assets=pd.DataFrame(asset); assets.index=asset
    returns=pd.DataFrame(filtered_df.iloc[len(filtered_df) - 1, 29:40])
    returns.index=asset
    risk=pd.DataFrame(filtered_df.iloc[len(filtered_df) - 1, 41:52])
    risk.index=asset
    profile=pd.concat([assets, risk,returns], axis=1)
    profile.columns= ["asset","risk(%)","return(%)"]
    X = profile["risk(%)"]; y = profile["return(%)"]
    line_fitter = LinearRegression()
    line_fitter.fit(X.values.reshape(-1, 1), y)
    profile["fitted"] = line_fitter.intercept_ + line_fitter.coef_ * X
    a = alt.Chart(profile).mark_area(opacity=1).encode(x='risk(%)', y='return(%)', tooltip=['risk(%)', 'return(%)','asset'])
    b = alt.Chart(profile).mark_area(opacity=0.6).encode(x='risk(%)', y='fitted',color=alt.value('red'))
    st.altair_chart((a.mark_circle(size=60) + b.mark_line()), use_container_width=True)

    st.subheader('**RAI and MSI ACWI**')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x='DATE', y='coef')
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x='DATE', y='MSCI AC World',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red'), use_container_width=True)

    st.subheader('**RAI and VIX**')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x='DATE', y='coef')
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x='DATE', y='CBOE VIX',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red'), use_container_width=True)

    st.subheader('**RAI and AAII Sentiment Bullish Index**')
    a = alt.Chart(filtered_df).mark_area(opacity=1).encode(x='DATE', y='coef')
    b = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(x='DATE', y='AAII Sentiment Bullish',color=alt.value('red'))
    st.altair_chart((a.mark_line() + b.mark_line()).resolve_scale(y='independent').configure_axisRight(labelColor='red'), use_container_width=True)

