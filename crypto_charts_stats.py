import csv
import datetime
import glob
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quandl as q
import streamlit as st
import requests
import pandas as pd
import json
import time
import warnings
from astral import moon

st.set_page_config(layout="wide")
import requests
import json
from dateutil import tz
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred

api_key_fred = os.environ["api_key_fred"]
fred = Fred(api_key=api_key_fred)

quandl_api_key = os.environ["quandl_api_key"]


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


# # # start - read in BINANCE BTC data # # # binance only goes back to 2017!
# def get_binance_btcusd():
#     URL = "https://api.binance.com/api/v3/klines"
#     start_str = "2014-01-01 00:00:00"
#     fmt = "%Y-%m-%d %H:%M:%S"
#     start_time = int(time.mktime(time.strptime(start_str, fmt)) * 1000)
#     last_open_time = 0  # added this line

#     df = pd.DataFrame()

#     while True:
#         parameters = {
#             "symbol": "BTCUSDT",
#             "interval": "1d",
#             "startTime": start_time,
#             "limit": 1000,  # maximum limit
#         }

#         response = requests.get(URL, params=parameters)
#         data = json.loads(response.text)

#         if len(data) == 0 or last_open_time == start_time:  # updated this line
#             break

#         temp_df = pd.DataFrame(
#             data,
#             columns=[
#                 "Open_time",
#                 "Open",
#                 "High",
#                 "Low",
#                 "Close",
#                 "Volume",
#                 "Close_time",
#                 "Quote_asset_volume",
#                 "Number_of_trades",
#                 "Taker_buy_base",
#                 "Taker_buy_quote",
#                 "Ignore",
#             ],
#         )
#         temp_df["Open_time"] = pd.to_datetime(temp_df["Open_time"], unit="ms")
#         temp_df["Date"] = temp_df["Open_time"].dt.date
#         temp_df["High"] = temp_df["High"].astype(float)
#         temp_df["Low"] = temp_df["Low"].astype(float)
#         temp_df["Last"] = temp_df["Close"].astype(float)
#         temp_df["Volume"] = temp_df["Volume"].astype(float)
#         temp_df["Mid"] = (temp_df["High"] + temp_df["Low"]) / 2
#         temp_df["First"] = temp_df["Last"].shift()

#         df = pd.concat([df, temp_df])
#         last_open_time = start_time  # added this line
#         start_time = (
#             int(temp_df["Open_time"].dt.to_pydatetime()[-1].timestamp() * 1000) + 1
#         )

#     df = df[["Date", "High", "Low", "Mid", "Last", "Volume", "First"]]
#     df.to_csv("coindata/binance_btcusdt.csv", index=False)


# datasource_btcusd = "binance_btcusdt.csv"
# btcusd_data = pd.read_csv("coindata/{}".format(datasource_btcusd), index_col=0)
# btcusd_data.index = pd.to_datetime(btcusd_data.index)

# most_recent_stored_btcusd_date = (
#     btcusd_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
# )

# todays_date = datetime.date.today()
# todays_date = todays_date.strftime("%Y-%m-%d")

# if most_recent_stored_btcusd_date != todays_date:
#     get_binance_btcusd()

#     btcusd_data = pd.read_csv("coindata/{}".format(datasource_btcusd), index_col=0)
#     btcusd_data.index = pd.to_datetime(btcusd_data.index)

# # # # end - read in BINANCE BTC data # # #


# # # start - read in BITFINEX BTC data # # #
datasource_btcusd = "BITFINEX/BTCUSD.csv"
btcusd_data = pd.read_csv(
    "coindata/{}".format(datasource_btcusd.replace("/", " ")), index_col=0
)
btcusd_data.index = pd.to_datetime(btcusd_data.index)

most_recent_stored_btcusd_date = (
    btcusd_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
)
todays_date = datetime.date.today() - datetime.timedelta(days=1)
todays_date = todays_date.strftime("%Y-%m-%d")

if most_recent_stored_btcusd_date != todays_date:
    data = q.get(
        datasource_btcusd.split(".")[0],
        start_date=most_recent_stored_btcusd_date,
        end_date="{}".format(todays_date),
        api_key=quandl_api_key,
    )
    data.info()
    data["First"] = data.Last.shift(1)
    data.dropna()
    btcusd_data = pd.concat([btcusd_data, data])
    btcusd_data = btcusd_data.sort_index()
    # store current df with up-to-date values
    btcusd_data.to_csv(
        "coindata/{}".format(datasource_btcusd.replace("/", " ")), index=True
    )
# # # end - read in BTC data # # #

# # # start - data processing # # #
btcusd_data = btcusd_data.dropna()
btcusd_data["350_movingaverage"] = pd.Series.rolling(
    btcusd_data["Last"], window=350, min_periods=1
).mean()
btcusd_data["111_movingaverage"] = pd.Series.rolling(
    btcusd_data["Last"], window=111, min_periods=1
).mean()
# # # end - data processing # # #

st.title("Crypto Charts")

# diagram - Pi Cycle Top Indicator BTC/USD
fig = go.Figure(
    data=go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data["Last"],
        mode="lines",
        marker=dict(
            # size=16,
            color="black",  # set color equal to a variable
            # colorscale='Viridis', # one of plotly colorscales
            # showscale=True
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data["350_movingaverage"] * 2,
        mode="lines",
        name="350_movingaverage",
        marker=dict(
            # size=[40, 60, 80, 100],
            color="red"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data["111_movingaverage"],
        mode="lines",
        name="111_movingaverage",
        marker=dict(
            # size=[40, 60, 80, 100],
            color="green"
        ),
    )
)
fig.update_yaxes(type="log")
fig.update_layout(
    # title="Plot Title",
    autosize=False,
    width=int(1400 / 1.1),
    height=int(800 / 1.1),
    title="Pi Cycle Top Indicator BTC/USD",
)
st.plotly_chart(fig)

# TODO: Future integrate ETH?

# Get FED data from quandl and store it
fed_assets_quandl_key = "FED/RESPPA_N_WW"
fed_assets_data = pd.read_csv(
    "coindata/{}".format(fed_assets_quandl_key.replace("/", " ")), index_col=0
)
fed_assets_data.index = pd.to_datetime(fed_assets_data.index)
most_recent_stored_fed_assets_date = fed_assets_data.sort_index().tail(1).index[0]
todays_date = datetime.date.today() - datetime.timedelta(days=6)
todays_date = todays_date

# every wednesday we get the data from the fed
if most_recent_stored_fed_assets_date < todays_date:
    fed_assets_data = q.get(fed_assets_quandl_key, api_key=quandl_api_key)
    fed_assets_data = fed_assets_data.dropna()
    fed_assets_data = fed_assets_data.sort_index()
    # store current df with up-to-date values
    fed_assets_data.to_csv(
        "coindata/{}".format(fed_assets_quandl_key.replace("/", " ")), index=True
    )

# merge fed and btc data and create new features
btcusd_data_and_fed = pd.merge(
    btcusd_data, fed_assets_data, left_index=True, right_index=True, how="left"
)
btcusd_data_and_fed["Value"] = btcusd_data_and_fed["Value"].ffill()
btcusd_data_and_fed = btcusd_data_and_fed.dropna()
btcusd_data_and_fed["BTC_per_FedAssets"] = (
    btcusd_data_and_fed["Last"] / btcusd_data_and_fed["Value"]
) * 1000
btcusd_data_and_fed["350_movingaverage_per_FedAssets"] = pd.Series.rolling(
    btcusd_data_and_fed["BTC_per_FedAssets"], window=350, min_periods=1
).mean()
btcusd_data_and_fed["111_movingaverage_per_FedAssets"] = pd.Series.rolling(
    btcusd_data_and_fed["BTC_per_FedAssets"], window=111, min_periods=1
).mean()

# diagram - Pi Cycle Top Indicator BTC/FED Total Assets
fig = go.Figure(
    data=go.Scatter(
        x=btcusd_data_and_fed.index,
        y=btcusd_data_and_fed["BTC_per_FedAssets"],
        name="BTC/FED",
        mode="lines",
        marker=dict(
            color="red",
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=btcusd_data_and_fed.index,
        y=btcusd_data_and_fed["350_movingaverage_per_FedAssets"],
        mode="lines",
        name="350_MA_per<br>_FedAssets",
        marker=dict(
            # size=[40, 60, 80, 100],
            color="gold"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=btcusd_data_and_fed.index,
        y=btcusd_data_and_fed["350_movingaverage_per_FedAssets"] * 2,
        mode="lines",
        name="2*350_MA_per<br>_FedAssets",
        marker=dict(
            # size=[40, 60, 80, 100],
            color="red"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data_and_fed["111_movingaverage_per_FedAssets"],
        mode="lines",
        name="111_MA_per<br>_FedAssets",
        marker=dict(
            # size=[40, 60, 80, 100],
            color="green"
        ),
    )
)
fig.update_yaxes(type="log")  # , range=[0,5]
fig.update_layout(
    # title="Plot Title",
    autosize=False,
    width=int(1400 / 1.1),
    height=int(800 / 1.1),
    # TODO check ezb data summed up with fed
    title="Pi Cycle Top Indicator BTC/FED Total Assets",
)
st.plotly_chart(fig)

# Get Data and Process it
data_SP500 = fred.get_series("SP500")  # Billions of Dollars
data_SP500 = data_SP500.sort_index()
# store current df with up-to-date values
data_SP500.to_csv("coindata/data_SP500.csv", index=True)

# Assets: Total Assets: Total Assets: Wednesday Level (RESPPANWW) Millions of Dollars
data_WALCL = fred.get_series("WALCL")
data_WALCL.dropna()
data_WALCL = data_WALCL.sort_index()
# store current df with up-to-date values
data_WALCL.to_csv("coindata/data_WALCL.csv", index=True)

# # Overnight Reverse Repurchase Agreements Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations (RRPONTSYD)
data_RRPONTSYD = fred.get_series("RRPONTSYD")
# Assets: Total Assets: Total Assets: Wednesday Level (RESPPANWW)
# Billions of U.S. Dollars
# data_FRED_RRPONTSYD.values = data_FRED_RRPONTSYD.values * 1000
data_RRPONTSYD = data_RRPONTSYD * 1000
data_RRPONTSYD.dropna()
data_RRPONTSYD = data_RRPONTSYD.sort_index()
# store current df with up-to-date values
data_RRPONTSYD.to_csv("coindata/data_RRPONTSYD.csv", index=True)

# # Deposits with Federal Reserve Banks, other than Reserve Balances: U.S. Treasury, General Account (WTREGEN)
# Billions of Dollars
data_FRED_WTREGEN = fred.get_series("WTREGEN")
data_FRED_WTREGEN = data_FRED_WTREGEN * 1000

data_FRED_WTREGEN.dropna()
data_FRED_WTREGEN = data_FRED_WTREGEN.sort_index()
# store current df with up-to-date values
data_FRED_WTREGEN.to_csv("coindata/data_FRED_WTREGEN.csv", index=True)

# get fred_total_assets
datasource_fred_total_assets = "data_WALCL.csv"
fred_total_assets = pd.read_csv(
    "coindata/{}".format(datasource_fred_total_assets.replace("/", " ")), index_col=0
)
fred_total_assets.index = pd.to_datetime(fred_total_assets.index)

most_recent_stored_fred_rrpontsyd_date = (
    fred_total_assets.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
)
todays_date = datetime.date.today() - datetime.timedelta(days=1)
todays_date = todays_date.strftime("%Y-%m-%d")

datasource_FRED_WTREGEN = "data_FRED_WTREGEN.csv"
FRED_WTREGEN_data = pd.read_csv(
    "coindata/{}".format(datasource_FRED_WTREGEN.replace("/", " ")), index_col=0
)
FRED_WTREGEN_data.index = pd.to_datetime(FRED_WTREGEN_data.index)

most_recent_stored_FRED_WTREGEN_date = (
    FRED_WTREGEN_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
)

datasource_FRED_RRPONTSYD = "data_RRPONTSYD.csv"
FRED_RRPONTSYD_data = pd.read_csv(
    "coindata/{}".format(datasource_FRED_RRPONTSYD.replace("/", " ")), index_col=0
)
FRED_RRPONTSYD_data.index = pd.to_datetime(FRED_RRPONTSYD_data.index)

most_recent_stored_FRED_RRPONTSYD_date = (
    FRED_RRPONTSYD_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
)

# filter data sources
FRED_RRPONTSYD_data = FRED_RRPONTSYD_data[
    (FRED_RRPONTSYD_data.index > "2020-08-11 00:00:00")
]
fred_total_assets = fred_total_assets[(fred_total_assets.index > "2020-08-11 00:00:00")]
FRED_WTREGEN_data = FRED_WTREGEN_data[(FRED_WTREGEN_data.index > "2020-08-11 00:00:00")]
data_SP500 = data_SP500[(data_SP500.index > "2020-08-11 00:00:00")]
btcusd_data = btcusd_data[(btcusd_data.index > "2020-08-11 00:00:00")]

# filter for datetime to allign all fred data sources!
merged_FRED_RRPONTSYD_data = pd.merge(
    FRED_WTREGEN_data,
    FRED_RRPONTSYD_data,
    how="left",
    left_index=True,
    right_index=True,
)
del merged_FRED_RRPONTSYD_data["0_x"]
merged_FRED_RRPONTSYD_data.columns = ["0"]

netLiquidity = fred_total_assets - merged_FRED_RRPONTSYD_data - FRED_WTREGEN_data

# shift sp two weeks back cause net liquidity fed predicts sp in two weeks!
netLiquidity = netLiquidity[(netLiquidity.index > "2012-11-18 00:00:00")]
netLiquidity = netLiquidity.squeeze()
netLiquidity = netLiquidity.dropna()

data_SP500_1weekback = data_SP500.shift(-7, "D")
data_SP500_1weekback = data_SP500_1weekback[
    (data_SP500_1weekback.index > "2013-08-12 00:00:00")
]
data_SP500_1weekback = data_SP500_1weekback[
    ~data_SP500_1weekback.index.duplicated(keep="first")
]
data_SP500_1weekback = data_SP500_1weekback.dropna()

dfdiffsp500_netliq = (
    pd.concat(
        [netLiquidity, data_SP500_1weekback],
        axis=1,
        keys=("netLiquidity", "data_SP500_1weekback"),
        join="outer",
    )
    .ffill(axis=0)
    .dropna()
)
dfdiffsp500_netliq["diff"] = (
    dfdiffsp500_netliq["data_SP500_1weekback"] - dfdiffsp500_netliq["netLiquidity"]
)

# Scale features to be between 0 and 1 to plot it together in one chart
# Create a scaler object
scaler = MinMaxScaler()
# Fit the scaler to the btcusd_data and transform it
btcusd_data_scaled = scaler.fit_transform(btcusd_data["Last"].values.reshape(-1, 1))
# Now, fit the scaler to the netLiquidity data and transform it
netLiquidity_scaled = scaler.fit_transform(netLiquidity.values.reshape(-1, 1))
# Convert these arrays back into pandas Series, keeping the original indices
btcusd_data_scaled = pd.Series(btcusd_data_scaled.flatten(), index=btcusd_data.index)
netLiquidity_scaled = pd.Series(netLiquidity_scaled.flatten(), index=netLiquidity.index)


# # # start - plot fed net liquidity! # # #
fig_net_liq = make_subplots(specs=[[{"secondary_y": True}]])
fig_net_liq.add_trace(
    go.Scatter(
        x=btcusd_data_scaled.index,
        y=btcusd_data_scaled + 0.4,
        name="BTC (Scaled)",
        mode="lines",
        marker=dict(color="red"),
    )
)
# adjust this value as needed
fig_net_liq.add_trace(
    go.Scatter(
        x=data_SP500.index,
        y=data_SP500.values,  # shift data downwards
        name="SP500 (shifted downwards)",
        mode="lines",
        yaxis="y5",
        marker=dict(color="blue"),
    )
)
fig_net_liq.add_trace(
    go.Scatter(
        x=netLiquidity_scaled.index,
        y=netLiquidity_scaled.values + 0.4,
        stackgroup="one",
        mode="lines",
        name="netLiquidity (Scaled)",
        marker=dict(color="green"),
    )
)
fig_net_liq.update_layout(
    title="Fed net liquidity predicts SP500 for following week!",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
    yaxis3=dict(anchor="x", overlaying="y", side="left"),
    yaxis4=dict(anchor="x", overlaying="y", side="right"),
    yaxis5=dict(
        anchor="x", overlaying="y", side="right"
    ),  # adjust the position downwards # We move this to the left by adjusting the position
)
fig_net_liq.update_yaxes(title_text="BTC & SP500", secondary_y=True)
st.plotly_chart(fig_net_liq)
# # # end - plot fed net liquidity! # # #

# add function to select between different cryptos. and then two charts below change!
# also add function for email notification? Full moon new moon maybe?

# also add diagram for support resistance!
# maybe bring this on own page? where the crypto dependent charts are that can be selected?

# # # start - plot volume bubble # # #
bubble_size = st.slider("Bubble Size", min_value=0, max_value=100, value=30)

# Normalizing the 'Volume'
btcusd_data["norm_volume"] = (btcusd_data["Volume"] - btcusd_data["Volume"].min()) / (
    btcusd_data["Volume"].max() - btcusd_data["Volume"].min()
)
# Calculate daily price change
btcusd_data["daily_change"] = btcusd_data["Last"].diff()
# Assign colors based on positive or negative change
btcusd_data["color"] = np.where(btcusd_data["daily_change"] > 0, "green", "red")
# Create figure with secondary y-axis
fig_vol_bubble = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
fig_vol_bubble.add_trace(
    go.Bar(
        x=btcusd_data.index,
        y=btcusd_data["Volume"],
        name="Volume",
        marker=dict(
            color="black",
            opacity=0.5,
        ),
    ),
    secondary_y=True,
)

fig_vol_bubble.add_trace(
    go.Scatter(
        x=btcusd_data.index,
        y=btcusd_data["Last"],
        mode="markers+lines",
        marker=dict(
            size=btcusd_data["norm_volume"] * bubble_size,  # Scale marker size
            color=btcusd_data["color"],  # Assign color based on 'daily_change'
        ),
        name="Last",
        line=dict(color="black"),
    ),
    secondary_y=False,
)

fig_vol_bubble.update_layout(
    xaxis_title="Date",
    yaxis_title="Last",
    yaxis2_title="Volume",
    xaxis_rangeslider_visible=True,
    title="BTC Bubble Volume Chart",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
)
st.plotly_chart(fig_vol_bubble)
# # # end - plot volume bubble # # #


# # # start - newmoon fullmoon chart # # #

# take the btcusd_data index and iterate over it to get the moon phase for each day
# the make a new df out of it called df_moon_phase
df_moon_phase = pd.DataFrame()
for i in btcusd_data.index:
    df_moon_phase = df_moon_phase.append(
        {"date": i, "moon_phase": moon.phase(i)}, ignore_index=True
    )

df_moon_phase["moon_phase"] = df_moon_phase["moon_phase"].fillna(0).astype(int)
df_moon_phase["date"] = pd.to_datetime(df_moon_phase["date"])
df_moon_phase["date"] = df_moon_phase["date"].dt.date
df_moon_phase = df_moon_phase.set_index("date")

# check the relationship between moon phase and btcusd_data
# merge sp500 with moon phase
btcusd_data_and_moon_phase = pd.merge(
    btcusd_data, df_moon_phase, left_index=True, right_index=True, how="left"
)
btcusd_data_and_moon_phase = btcusd_data_and_moon_phase.dropna()

# 0 .. 6.99	New moon
# 7 .. 13.99	First quarter
# 14 .. 20.99	Full moon
# 21 .. 27.99	Last quarter
# create a new column called moon_phase_category and fill it with the moon_phase_category
btcusd_data_and_moon_phase["moon_phase_category"] = ""
btcusd_data_and_moon_phase.loc[
    (btcusd_data_and_moon_phase["moon_phase"] >= 0)
    & (btcusd_data_and_moon_phase["moon_phase"] <= 6.99),
    "moon_phase_category",
] = 0
btcusd_data_and_moon_phase.loc[
    (btcusd_data_and_moon_phase["moon_phase"] >= 7)
    & (btcusd_data_and_moon_phase["moon_phase"] <= 13.99),
    "moon_phase_category",
] = 7
btcusd_data_and_moon_phase.loc[
    (btcusd_data_and_moon_phase["moon_phase"] >= 14)
    & (btcusd_data_and_moon_phase["moon_phase"] <= 20.99),
    "moon_phase_category",
] = 14
btcusd_data_and_moon_phase.loc[
    (btcusd_data_and_moon_phase["moon_phase"] >= 21)
    & (btcusd_data_and_moon_phase["moon_phase"] <= 27.99),
    "moon_phase_category",
] = 21


# make a new plot and plot the daily_price_change of btc by time on the x-axis
fig_btc_moon = make_subplots(specs=[[{"secondary_y": True}]])
fig_btc_moon.add_trace(
    go.Scatter(
        x=btcusd_data_and_moon_phase.index,
        y=btcusd_data_and_moon_phase["Last"],
        name="Last",
        mode="lines",
        marker=dict(color="red"),
    )
)

last_moon_phase = None

for i in range(len(btcusd_data_and_moon_phase)):
    current_moon_phase = btcusd_data_and_moon_phase["moon_phase_category"].iloc[i]
    if (
        current_moon_phase != last_moon_phase
    ):  # check if the moon phase category changed
        if current_moon_phase == 0:
            fig_btc_moon.add_annotation(
                x=btcusd_data_and_moon_phase.index[i],
                y=0.9,
                text="🌑",
                showarrow=False,
                font=dict(
                    size=16,
                ),
                xref="x",
                yref="paper",  # Position annotation relative to the entire plot
                yanchor="bottom",  # Anchor the bottom of the text at y
            )
        elif current_moon_phase == 14:
            fig_btc_moon.add_annotation(
                x=btcusd_data_and_moon_phase.index[i],
                y=0,
                text="🌕",
                showarrow=False,
                font=dict(
                    size=16,
                ),
                xref="x",
                yref="paper",  # Position annotation relative to the entire plot
                yanchor="bottom",  # Anchor the bottom of the text at y
            )
        last_moon_phase = current_moon_phase  # update the last moon phase


fig_btc_moon.update_layout(
    xaxis_title="Date",
    yaxis_title="Last",
    yaxis2_title="Volume",
    xaxis_rangeslider_visible=True,
    title="BTC moon Chart",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
)
st.plotly_chart(fig_btc_moon)
# # # end - plot volume bubble # # #
