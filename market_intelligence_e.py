# streamlit run market_intelligence_e.py

import streamlit as st
from yahoo_fin.stock_info import *
import yfinance as yf
import pandas as pd
import cufflinks as cf
from datetime import datetime, timedelta
import os

# Retrieving tickers data
current_path = os.getcwd()
file='\\datasets\\list.csv'
ticker=pd.read_csv(current_path + file)
ticker_list=ticker["Security"]

# App title
st.markdown("# Market intelligence")
st.markdown("Shown are the stock price data and fundamentals for US listed companies")
st.markdown("**Notifications**")
st.markdown("- Data retrieved from yahoo finance\n"
            "- Test version 2021-03-08")

# Sidebar
st.sidebar.subheader('Search stock')
ago = datetime.now() - timedelta(365)
yesterday = datetime.now() - timedelta(1)
start_date = st.sidebar.date_input("Start date", ago)
end_date = st.sidebar.date_input("End date", yesterday)

tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker["Security"])
selected = ticker.loc[ticker["Security"]==tickerSymbol,["Symbol"]]
tickerData = yf.Ticker(selected.iloc[0]["Symbol"]) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)
string_name = tickerData.info['longName']
abb_name = selected.iloc[0]["Symbol"]
st.header('**%s**' % string_name)
st.subheader(abb_name)

## Buttons
if st.button("stock description"):
    st.info(tickerData.info['longBusinessSummary'])

# Ticker data
st.header('**Market Price**')
tickerDf.index = tickerDf.index.strftime('%Y-%m-%d')
st.write(tickerDf.iloc[:,0:5])

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Fundamentals
st.header('**Fundamentals**')
income_statement = get_income_statement(abb_name, yearly=False)
income_statement.columns = income_statement.columns.strftime('%Y-%m-%d')
st.write(income_statement)

# News from Yahoo finance RSS feeds
from pprint import pprint as pp
from yahoo_fin import news
news=news.get_yf_rss(abb_name)
df = pd.DataFrame(news,columns = ['title','published','summary'])
st.write(df)

## Error/message text
st.success("Successful")
st.info("Information!")
st.warning("This is a warning")
st.error("This is an error!")
st.exception("NameError('Error name is not defined')")

st.markdown("**Disclaimers**")
st.markdown("- Web application built by Hyeonjong Jung. March 2021.")
st.markdown("- All data and analysis used herein are for illustration purposes only. All data and analysis are intended for these purposes and should not be considered as specific investment advice")
st.sidebar.markdown('<a href="mailto:hyeonjong.jung@gmail.com">Contact me</a>', unsafe_allow_html=True)

