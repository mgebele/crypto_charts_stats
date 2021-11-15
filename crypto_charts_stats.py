import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, timedelta
import plotly.graph_objects as go
import datetime
import pandas as pd
import tweepy
import csv
import os
import time
from dateutil import tz
import glob
import quandl as q
import re
import streamlit as st
st.set_page_config(layout="wide")

quandl_api_key = os.environ['quandl_api_key']


def _max_width_():
    max_width_str = f"max-width: 1300px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()


def main():
    # # # start - read in BTC data # # #
    datasource_btcusd = "BITFINEX/BTCUSD.csv"
    btcusd_data = pd.read_csv("coindata/{}".format(
        datasource_btcusd.replace("/", " ")), index_col=0)
    btcusd_data.index = pd.to_datetime(btcusd_data.index)

    most_recent_stored_btcusd_date = btcusd_data.sort_index().tail(
        1).index[0].strftime("%Y-%m-%d")
    todays_date = datetime.date.today() - timedelta(days=1)
    todays_date = todays_date.strftime("%Y-%m-%d")

    if most_recent_stored_btcusd_date != todays_date:
        data = q.get(datasource_btcusd.split(".")[0],   start_date=most_recent_stored_btcusd_date,
                     end_date='{}'.format(todays_date),
                     api_key=quandl_api_key)
        data.info()
        data["First"] = data.Last.shift(1)
        data.dropna()
        btcusd_data = pd.concat([btcusd_data, data])
        btcusd_data = btcusd_data.sort_index()
        # store current df with up-to-date values
        btcusd_data.to_csv('{}'.format(
            datasource_btcusd.replace("/", " ")), index=True)
    # # # end - read in BTC data # # #

    btcusd_data

    # # # start - chart with tweets # # #
    fig = go.Figure(
        data=[go.Candlestick(x=btcusd_data.index,
                             open=btcusd_data['First'],
                             high=btcusd_data['High'],
                             low=btcusd_data['Low'],
                             close=btcusd_data['Last'],
                             name="{}".format(datasource_btcusd.split("/")
                                              [1].split(".")[0]),
                             )],
    )

    fig.update_layout(
        # title="Plot Title",
        autosize=False,
        width=int(1400/1.3),
        height=int(800/1.3),
    )
    st.plotly_chart(fig)
    # # # end - chart with tweets # # #


run_it = st.sidebar.button('Show visualizations')
st.sidebar.text("")

# TODO: Future integrate ETH
# display_name_all_twitter_user_scraped_csvs, all_twitter_user_scraped_csvs = get_all_stored_crypto_csvs()
# display_name_user_selection_list_containing_twitter_user = st.sidebar.selectbox(
#     "Select existing Twitter-User", list(display_name_all_twitter_user_scraped_csvs), 0)


if run_it:
    main()
