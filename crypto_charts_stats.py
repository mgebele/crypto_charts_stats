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
import plotly.graph_objects as go
import numpy as np
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
    btcusd_data.to_csv('coindata/{}'.format(
        datasource_btcusd.replace("/", " ")), index=True)
# # # end - read in BTC data # # #

# # # start - data processing # # #
btcusd_data = btcusd_data.dropna()
btcusd_data["350_movingaverage"] = pd.Series.rolling(
    btcusd_data["Last"], window=350, min_periods=1).mean()
btcusd_data["111_movingaverage"] = pd.Series.rolling(
    btcusd_data["Last"], window=111, min_periods=1).mean()
# # # end - data processing # # #

st.title("Crypto Charts")

fig = go.Figure(
    data=go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data['Last'],
        mode='lines',
        marker=dict(
            # size=16,
            color="black",  # set color equal to a variable
            # colorscale='Viridis', # one of plotly colorscales
            # showscale=True
        )
    )
)

fig.add_trace(go.Scatter(x=btcusd_data.index, y=btcusd_data["350_movingaverage"]*2,
                         mode='lines',
                         name='350_movingaverage',
                         marker=dict(
    # size=[40, 60, 80, 100],
    color="red"
),
)
)

fig.add_trace(go.Scatter(x=btcusd_data.index, y=btcusd_data["111_movingaverage"],
                         mode='lines',
                         name='111_movingaverage',
                         marker=dict(
    # size=[40, 60, 80, 100],
    color="green"
)
)
)

# btcusd_data["111_movingaverage"]
fig.update_yaxes(type="log")  # , range=[0,5]

fig.update_layout(
    # title="Plot Title",
    autosize=False,
    width=int(1400/1.1),
    height=int(800/1.1),
    title="Pi Cycle Top Indicator BTC/USD"
)

st.plotly_chart(fig)
# # # end - chart with tweets # # #


# run_it = st.sidebar.button('Show visualizations')
st.sidebar.text("")

# TODO: Future integrate ETH
# display_name_all_twitter_user_scraped_csvs, all_twitter_user_scraped_csvs = get_all_stored_crypto_csvs()
# display_name_user_selection_list_containing_twitter_user = st.sidebar.selectbox(
#     "Select existing Twitter-User", list(display_name_all_twitter_user_scraped_csvs), 0)


fed_assets_quandl_key = "FED/RESPPA_N_WW"

fed_assets_data = pd.read_csv("coindata/{}".format(
    fed_assets_quandl_key.replace("/", " ")), index_col=0)
fed_assets_data.index = pd.to_datetime(fed_assets_data.index)

most_recent_stored_fed_assets_date = fed_assets_data.sort_index().tail(
    1).index[0]

# every wednesday we get the data from the fed
todays_date = datetime.date.today() - timedelta(days=6)
todays_date = todays_date

if most_recent_stored_fed_assets_date < todays_date:
    fed_assets_data = q.get(fed_assets_quandl_key,
                            api_key=quandl_api_key
                            )
    fed_assets_data = fed_assets_data.dropna()
    fed_assets_data = fed_assets_data.sort_index()
    fed_assets_data.to_csv('coindata/{}'.format(
        fed_assets_quandl_key.replace("/", " ")), index=True)
# store current df with up-to-date values

# merge fed and btc
btcusd_data_and_fed = pd.merge(
    btcusd_data, fed_assets_data, left_index=True, right_index=True, how='left')
btcusd_data_and_fed["Value"] = btcusd_data_and_fed["Value"].ffill()
btcusd_data_and_fed = btcusd_data_and_fed.dropna()
btcusd_data_and_fed["BTC_per_FedAssets"] = (
    btcusd_data_and_fed["Last"] / btcusd_data_and_fed["Value"]) * 1000
btcusd_data_and_fed["350_movingaverage_per_FedAssets"] = pd.Series.rolling(
    btcusd_data_and_fed["BTC_per_FedAssets"], window=350, min_periods=1).mean()
btcusd_data_and_fed["111_movingaverage_per_FedAssets"] = pd.Series.rolling(
    btcusd_data_and_fed["BTC_per_FedAssets"], window=111, min_periods=1).mean()
fig = go.Figure(
    data=go.Scatter(
        x=btcusd_data_and_fed.index,
        y=btcusd_data_and_fed['BTC_per_FedAssets'],
        name="BTC/FED",
        mode='lines',
        marker=dict(
            color="red",
        )
    )
)

fig.add_trace(go.Scatter(x=btcusd_data_and_fed.index, y=btcusd_data_and_fed["350_movingaverage_per_FedAssets"],
                         mode='lines',
                         name='350_MA_per<br>_FedAssets',
                         marker=dict(
    # size=[40, 60, 80, 100],
    color="gold"
),
)
)

fig.add_trace(go.Scatter(x=btcusd_data_and_fed.index, y=btcusd_data_and_fed["350_movingaverage_per_FedAssets"]*2,
                         mode='lines',
                         name='2*350_MA_per<br>_FedAssets',
                         marker=dict(
    # size=[40, 60, 80, 100],
    color="red"
),
)
)

fig.add_trace(go.Scatter(x=btcusd_data.index, y=btcusd_data_and_fed["111_movingaverage_per_FedAssets"],
                         mode='lines',
                         name='111_MA_per<br>_FedAssets',
                         marker=dict(
    # size=[40, 60, 80, 100],
    color="green"
)
)
)

# btcusd_data["111_movingaverage"]
fig.update_yaxes(type="log")  # , range=[0,5]

fig.update_layout(
    # title="Plot Title",
    autosize=False,
    width=int(1400/1.1),
    height=int(800/1.1),
    # TODO check ezb data summed up with fed
    title="Pi Cycle Top Indicator BTC/FED Total Assets"
)

st.plotly_chart(fig)



st.dataframe(btcusd_data)
