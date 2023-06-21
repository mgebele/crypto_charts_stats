import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import datetime
import pandas as pd
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
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(layout="wide")
from fredapi import Fred

api_key_fred = os.environ['api_key_fred']
fred = Fred(api_key=api_key_fred)

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


# st.dataframe(btcusd_data)


# # # start - get data from fred api and store csv # # #
# # GET S&P DATA
# Billions of Dollars 
data_SP500 = fred.get_series('SP500')
data_SP500 = data_SP500.sort_index()
# store current df with up-to-date values
data_SP500.to_csv('coindata/data_SP500.csv', index=True)

# # Assets: Total Assets: Total Assets: Wednesday Level (RESPPANWW)
#Assets: Total Assets: Total Assets: Wednesday Level (RESPPANWW) Millions of Dollars
data_WALCL = fred.get_series('WALCL')
data_WALCL.dropna()
data_WALCL = data_WALCL.sort_index()
# store current df with up-to-date values
data_WALCL.to_csv('coindata/data_WALCL.csv', index=True)

# # Overnight Reverse Repurchase Agreements Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations (RRPONTSYD)
data_RRPONTSYD = fred.get_series('RRPONTSYD')
#Assets: Total Assets: Total Assets: Wednesday Level (RESPPANWW)
# Billions of U.S. Dollars
# data_FRED_RRPONTSYD.values = data_FRED_RRPONTSYD.values * 1000 
data_RRPONTSYD = data_RRPONTSYD * 1000 
data_RRPONTSYD.dropna()
data_RRPONTSYD = data_RRPONTSYD.sort_index()
# store current df with up-to-date values
data_RRPONTSYD.to_csv('coindata/data_RRPONTSYD.csv', index=True)

# # Deposits with Federal Reserve Banks, other than Reserve Balances: U.S. Treasury, General Account (WTREGEN)
# Billions of Dollars 
data_FRED_WTREGEN = fred.get_series('WTREGEN')
data_FRED_WTREGEN = data_FRED_WTREGEN * 1000

data_FRED_WTREGEN.dropna()
data_FRED_WTREGEN = data_FRED_WTREGEN.sort_index()
# store current df with up-to-date values
data_FRED_WTREGEN.to_csv('coindata/data_FRED_WTREGEN.csv', index=True)

# # net liquidity vs s&p500 weekly - Diagram
datasource_fred_total_assets = "data_WALCL.csv"
fred_total_assets = pd.read_csv("coindata/{}".format(
    datasource_fred_total_assets.replace("/", " ")), index_col=0)
fred_total_assets.index = pd.to_datetime(fred_total_assets.index)

most_recent_stored_fred_rrpontsyd_date = fred_total_assets.sort_index().tail(
    1).index[0].strftime("%Y-%m-%d")
todays_date = datetime.date.today() - timedelta(days=1)
todays_date = todays_date.strftime("%Y-%m-%d")

datasource_FRED_WTREGEN = "data_FRED_WTREGEN.csv"
FRED_WTREGEN_data = pd.read_csv("coindata/{}".format(
    datasource_FRED_WTREGEN.replace("/", " ")), index_col=0)
FRED_WTREGEN_data.index = pd.to_datetime(FRED_WTREGEN_data.index)

most_recent_stored_FRED_WTREGEN_date = FRED_WTREGEN_data.sort_index().tail(
    1).index[0].strftime("%Y-%m-%d")

datasource_FRED_RRPONTSYD = "data_RRPONTSYD.csv"
FRED_RRPONTSYD_data = pd.read_csv("coindata/{}".format(
    datasource_FRED_RRPONTSYD.replace("/", " ")), index_col=0)
FRED_RRPONTSYD_data.index = pd.to_datetime(FRED_RRPONTSYD_data.index)

most_recent_stored_FRED_RRPONTSYD_date = FRED_RRPONTSYD_data.sort_index().tail(
    1).index[0].strftime("%Y-%m-%d")

# # # end - get data from fred api and store csv # # #


# # # start - filter data sources! # # #

FRED_RRPONTSYD_data = FRED_RRPONTSYD_data[(FRED_RRPONTSYD_data.index > '2020-08-11 00:00:00')]
fred_total_assets = fred_total_assets[(fred_total_assets.index > '2020-08-11 00:00:00')]
FRED_WTREGEN_data = FRED_WTREGEN_data[(FRED_WTREGEN_data.index > '2020-08-11 00:00:00')]
data_SP500 = data_SP500[(data_SP500.index > '2020-08-11 00:00:00')]
btcusd_data = btcusd_data[(btcusd_data.index > '2020-08-11 00:00:00')]

# # filter for datetime to allign all fred data sources!
merged_FRED_RRPONTSYD_data = pd.merge(FRED_WTREGEN_data,FRED_RRPONTSYD_data, how='left', left_index=True, right_index=True)
# merged.isnull().sum()
del merged_FRED_RRPONTSYD_data["0_x"]
merged_FRED_RRPONTSYD_data.columns=["0"]

netLiquidity = fred_total_assets-merged_FRED_RRPONTSYD_data-FRED_WTREGEN_data

# shift sp two weeks back cause net liquidity fed predicts sp in two weeks!
# data_SP500_2weeksback = data_SP500.shift(-2,"W")
# data_SP500_2weeksback

netLiquidity = netLiquidity[(netLiquidity.index > '2012-11-18 00:00:00')]
netLiquidity = netLiquidity.squeeze()
netLiquidity = netLiquidity.dropna()

data_SP500_1weekback = data_SP500.shift(-7, "D")
data_SP500_1weekback = data_SP500_1weekback[(data_SP500_1weekback.index > '2013-08-12 00:00:00')]
# data_SP500_1weekback = data_SP500_1weekback.index.drop_duplicates()
data_SP500_1weekback = data_SP500_1weekback[~data_SP500_1weekback.index.duplicated(keep='first')]
data_SP500_1weekback = data_SP500_1weekback.dropna()


dfdiffsp500_netliq = pd.concat([netLiquidity, data_SP500_1weekback], axis=1, keys=('netLiquidity','data_SP500_1weekback'), join='outer').ffill(axis = 0).dropna()
# dfdiffsp500_netliq["netLiquidity"]  = dfdiffsp500_netliq["netLiquidity"] / 1.1 - 1625
dfdiffsp500_netliq["diff"] = dfdiffsp500_netliq["data_SP500_1weekback"]  - dfdiffsp500_netliq["netLiquidity"]  

# # # end - filter data sources! # # #


# Create a scaler object
scaler = MinMaxScaler()

# Fit the scaler to the btcusd_data and transform it
btcusd_data_scaled = scaler.fit_transform(btcusd_data['Last'].values.reshape(-1, 1))

# Now, fit the scaler to the netLiquidity data and transform it
netLiquidity_scaled = scaler.fit_transform(netLiquidity.values.reshape(-1, 1))

# Convert these arrays back into pandas Series, keeping the original indices
btcusd_data_scaled = pd.Series(btcusd_data_scaled.flatten(), index=btcusd_data.index)
netLiquidity_scaled = pd.Series(netLiquidity_scaled.flatten(), index=netLiquidity.index)


# # # start - plot fed net liquidity! # # #

fig_net_liq = make_subplots(specs=[[{"secondary_y": True}]])

# fig_net_liq.add_trace(
#     go.Scatter(
#         x=dfdiffsp500_netliq.index,
#         y=dfdiffsp500_netliq["diff"],
#         name="diffsp500_netliq",
#         mode='lines',
#         yaxis='y3',
#         marker=dict(color="gold"), 
#     )
# )

fig_net_liq.add_trace(
    go.Scatter(
        x=btcusd_data_scaled.index,
        y=btcusd_data_scaled+0.1,
        name="BTC (Scaled)",
        mode='lines',
        marker=dict(color="red"), 
    ), 
)

fig_net_liq.add_trace(
    go.Scatter(
        x=data_SP500.index,
        y=data_SP500.values,
        name="SP500",
        mode='lines',
        yaxis='y5',
        marker=dict(color="blue"), 
    )
)


fig_net_liq.update_layout(yaxis=dict(domain=[0, 0.7]) )

fig_net_liq.add_trace(go.Scatter(x=netLiquidity_scaled.index, y=netLiquidity_scaled, stackgroup='one',
                    mode='lines',
                    name='netLiquidity (Scaled)',
                    marker=dict(color="green"),
                    ),
)


# fig_net_liq.add_trace(go.Scatter(x=fred_total_assets.index, y=fred_total_assets["0"],
#                     mode='lines',
#                     name='fred_total_assets',
#                     marker=dict(color="black"),
#                     )
#                 )

# fig_net_liq.add_trace(go.Scatter(x=merged_FRED_RRPONTSYD_data.index, y=merged_FRED_RRPONTSYD_data["0"], stackgroup='one',
#                     mode='lines',
#                     name='merged_FRED_RRPONTSYD_data',
#                     marker=dict(color="yellow"),
#                     )
#                 )

# fig_net_liq.add_trace(go.Scatter(x=FRED_WTREGEN_data.index, y=FRED_WTREGEN_data["0"], stackgroup='one',
#                     mode='lines',
#                     name='FRED_WTREGEN_data',
#                     marker=dict(color="red"),
#                     )
#                 )

fig_net_liq.update_layout(
    title="Fed net liquidity predicts SP500 for following week!",
    autosize=False,
    width=int(1400/1),
    height=int(800/1),
    yaxis3=dict(anchor="x", overlaying="y", side="left"),
    yaxis4=dict(anchor="x", overlaying="y", side="right"),
    yaxis5=dict(anchor="x", overlaying="y", side="right", position=0.05) # We move this to the left by adjusting the position
)

fig_net_liq.update_yaxes(title_text="BTC & SP500", secondary_y=True)

st.plotly_chart(fig_net_liq)

# # # end - plot fed net liquidity! # # #
