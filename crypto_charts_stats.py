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
import json
import time
import warnings
from astral import moon
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
def get_binance_crypto_usdt(selected_crypto: str = "BTC"):
    URL = "https://api.binance.com/api/v3/klines"
    start_str = "2014-01-01 00:00:00"
    fmt = "%Y-%m-%d %H:%M:%S"
    start_time = int(time.mktime(time.strptime(start_str, fmt)) * 1000)
    last_open_time = 0  # added this line

    df = pd.DataFrame()

    while True:
        parameters = {
            "symbol": f"{selected_crypto}USD",
            "interval": "1d",
            "startTime": start_time,
            "limit": 1000,  # maximum limit
        }

        response = requests.get(URL, params=parameters)
        data = json.loads(response.text)

        if len(data) < 1 or last_open_time == start_time:  # updated this line
            break

        temp_df = pd.DataFrame(
            data,
            columns=[
                "Open_time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_time",
                "Quote_asset_volume",
                "Number_of_trades",
                "Taker_buy_base",
                "Taker_buy_quote",
                "Ignore",
            ],
        )
        temp_df["Open_time"] = pd.to_datetime(temp_df["Open_time"], unit="ms")
        temp_df["Date"] = temp_df["Open_time"].dt.date
        temp_df["High"] = temp_df["High"].astype(float)
        temp_df["Low"] = temp_df["Low"].astype(float)
        temp_df["Last"] = temp_df["Close"].astype(float)
        temp_df["Volume"] = temp_df["Volume"].astype(float)
        temp_df["Mid"] = (temp_df["High"] + temp_df["Low"]) / 2
        temp_df["First"] = temp_df["Last"].shift()

        df = pd.concat([df, temp_df])

        if not temp_df.empty:
            last_open_time = (
                start_time  # This line should only run if temp_df is not empty.
            )
            start_time = (
                int(temp_df["Open_time"].dt.to_pydatetime()[-1].timestamp() * 1000) + 1
            )
        else:
            break  # If temp_df is empty, we should break from the loop.

    df = df[["Date", "High", "Low", "Mid", "Last", "Volume", "First"]]
    df.to_csv(f"coindata/binance {selected_crypto}USD.csv", index=False)


cryptos = [
    "BTC",
    "ETH",
    "DOGE",
    "LINK",
    "OP",
    "MATIC",
    "XRP",
    "LTC",
    "EOS",
    "MANA",
    "SAND",
]
selected_crypto = st.selectbox("Select Cryptocurrency", cryptos)
datasource = f"binance/{selected_crypto}USD.csv"

todays_date = datetime.date.today()
todays_date = todays_date.strftime("%Y-%m-%d")

try:
    xusd_data = pd.read_csv(
        "coindata/{}".format(datasource.replace("/", " ")), index_col=0
    )
except:
    get_binance_crypto_usdt(selected_crypto)
    xusd_data = pd.read_csv(
        "coindata/{}".format(datasource.replace("/", " ")), index_col=0
    )


datasource = f"binance {selected_crypto}USD.csv"
xusd_data = pd.read_csv("coindata/{}".format(datasource), index_col=0)
xusd_data.index = pd.to_datetime(xusd_data.index)

most_recent_stored_btcusd_date = (
    xusd_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")
)


if most_recent_stored_btcusd_date != todays_date:
    get_binance_crypto_usdt(selected_crypto)

    xusd_data = pd.read_csv("coindata/{}".format(datasource), index_col=0)
    xusd_data.index = pd.to_datetime(xusd_data.index)

# # # # end - read in BINANCE BTC data # # #


# # # # start - read in BITFINEX data # # #

# todays_date = datetime.date.today() - datetime.timedelta(days=1)
# todays_date = todays_date.strftime("%Y-%m-%d")


# def store_crypto_csv_from_quandl(datasource, todays_date):
#     data = q.get(
#         datasource.split(".")[0],
#         start_date="2016-01-01",
#         end_date="{}".format(todays_date),
#         api_key=quandl_api_key,
#     )
#     data.info()
#     data["First"] = data.Last.shift(1)
#     data.dropna()
#     data = data.sort_index()
#     # store current df with up-to-date values
#     data.to_csv("coindata/{}".format(datasource.replace("/", " ")), index=True)


# def update_stored_crypto_csv_from_quandl(
#     datasource, xusd_data, most_recent_stored_date, todays_date
# ):
#     data = q.get(
#         datasource.split(".")[0],
#         start_date=most_recent_stored_date,
#         end_date="{}".format(todays_date),
#         api_key=quandl_api_key,
#     )
#     data.info()
#     data["First"] = data.Last.shift(1)
#     data.dropna()
#     xusd_data = pd.concat([xusd_data, data])
#     xusd_data = xusd_data.sort_index()
#     # store current df with up-to-date values
#     xusd_data.to_csv("coindata/{}".format(datasource.replace("/", " ")), index=True)


# cryptos = [
#     "BTC",
#     "ETH",
#     "DOGE",
#     "LINK",
#     "OP",
#     "MATIC",
#     "XRP",
#     "LTC",
#     "EOS",
#     "MANA",
#     "SAND",
# ]
# selected_crypto = st.selectbox("Select Cryptocurrency", cryptos)
# datasource = f"BITFINEX/{selected_crypto}USD.csv"

# try:
#     xusd_data = pd.read_csv(
#         "coindata/{}".format(datasource.replace("/", " ")), index_col=0
#     )
# except:
#     store_crypto_csv_from_quandl(datasource, todays_date)
#     xusd_data = pd.read_csv(
#         "coindata/{}".format(datasource.replace("/", " ")), index_col=0
#     )

# xusd_data.index = pd.to_datetime(xusd_data.index)
# most_recent_stored_date = xusd_data.sort_index().tail(1).index[0].strftime("%Y-%m-%d")

# if most_recent_stored_date != todays_date:
#     update_stored_crypto_csv_from_quandl(
#         datasource, xusd_data, most_recent_stored_date, todays_date
#     )
# # # # end - read in BITFINEX data # # #

# # # start - data processing # # #
xusd_data = xusd_data.dropna()
xusd_data["350_movingaverage"] = pd.Series.rolling(
    xusd_data["Last"], window=350, min_periods=1
).mean()
xusd_data["111_movingaverage"] = pd.Series.rolling(
    xusd_data["Last"], window=111, min_periods=1
).mean()
# # # end - data processing # # #

# diagram - Pi Cycle Top Indicator BTC/USD
fig = go.Figure(
    data=go.Scatter(
        x=xusd_data.index,
        y=xusd_data["Last"],
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
        x=xusd_data.index,
        y=xusd_data["350_movingaverage"] * 2,
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
        x=xusd_data.index,
        y=xusd_data["111_movingaverage"],
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
    title=f"Pi Cycle Top Indicator {selected_crypto}/USD",
)
st.plotly_chart(fig)

# this name is only kept for storing and reading the current csv file
# had to change to the fred api because quandl didnt support it anymore with the following key:
fed_assets_quandl_key = "FED/RESPPA_N_WW"

fed_assets_data = pd.read_csv(
    "coindata/{}".format(fed_assets_quandl_key.replace("/", " ")), index_col=0
)
fed_assets_data.index = pd.to_datetime(fed_assets_data.index)
most_recent_stored_fed_assets_date = fed_assets_data.sort_index().tail(1).index[0]
todays_date = datetime.date.today() - datetime.timedelta(days=6)

# every wednesday we get the data from the fed
# Convert pd.Timestamp to datetime.date for comparison
if most_recent_stored_fed_assets_date.date() < todays_date:
    # Get FED data from fed and store it
    fred = Fred(api_key=api_key_fred)
    fed_assets_data = fred.get_series("RESPPANWW")
    fed_assets_data.index.name = "Date"
    fed_assets_data.name = "Value"
    fed_assets_data = fed_assets_data.dropna()
    fed_assets_data = fed_assets_data.sort_index()
    # store current df with up-to-date values
    fed_assets_data.to_csv(
        "coindata/{}".format(fed_assets_quandl_key.replace("/", " ")), index=True
    )

# merge fed and btc data and create new features
xusd_data_and_fed = pd.merge(
    xusd_data, fed_assets_data, left_index=True, right_index=True, how="left"
)
xusd_data_and_fed["Value"] = xusd_data_and_fed["Value"].ffill()
xusd_data_and_fed = xusd_data_and_fed.dropna()
xusd_data_and_fed["BTC_per_FedAssets"] = (
    xusd_data_and_fed["Last"] / xusd_data_and_fed["Value"]
) * 1000
xusd_data_and_fed["350_movingaverage_per_FedAssets"] = pd.Series.rolling(
    xusd_data_and_fed["BTC_per_FedAssets"], window=350, min_periods=1
).mean()
xusd_data_and_fed["111_movingaverage_per_FedAssets"] = pd.Series.rolling(
    xusd_data_and_fed["BTC_per_FedAssets"], window=111, min_periods=1
).mean()

# diagram - Pi Cycle Top Indicator BTC/FED Total Assets
fig = go.Figure(
    data=go.Scatter(
        x=xusd_data_and_fed.index,
        y=xusd_data_and_fed["BTC_per_FedAssets"],
        name="BTC/FED",
        mode="lines",
        marker=dict(
            color="red",
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=xusd_data_and_fed.index,
        y=xusd_data_and_fed["350_movingaverage_per_FedAssets"],
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
        x=xusd_data_and_fed.index,
        y=xusd_data_and_fed["350_movingaverage_per_FedAssets"] * 2,
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
        x=xusd_data.index,
        y=xusd_data_and_fed["111_movingaverage_per_FedAssets"],
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
    title=f"Pi Cycle Top Indicator {selected_crypto}/FED Total Assets",
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
xusd_data = xusd_data[(xusd_data.index > "2020-08-11 00:00:00")]

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
# Fit the scaler to the xusd_data and transform it
xusd_data_scaled = scaler.fit_transform(xusd_data["Last"].values.reshape(-1, 1))
# Now, fit the scaler to the netLiquidity data and transform it
netLiquidity_scaled = scaler.fit_transform(netLiquidity.values.reshape(-1, 1))
# Convert these arrays back into pandas Series, keeping the original indices
xusd_data_scaled = pd.Series(xusd_data_scaled.flatten(), index=xusd_data.index)
netLiquidity_scaled = pd.Series(netLiquidity_scaled.flatten(), index=netLiquidity.index)


# # # start - plot fed net liquidity! # # #
fig_net_liq = make_subplots(specs=[[{"secondary_y": True}]])
fig_net_liq.add_trace(
    go.Scatter(
        x=xusd_data_scaled.index,
        y=xusd_data_scaled + 0.4,
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
xusd_data["norm_volume"] = (xusd_data["Volume"] - xusd_data["Volume"].min()) / (
    xusd_data["Volume"].max() - xusd_data["Volume"].min()
)
# Calculate daily price change
xusd_data["daily_change"] = xusd_data["Last"].diff()
# Assign colors based on positive or negative change
xusd_data["color"] = np.where(xusd_data["daily_change"] > 0, "green", "red")
# Create figure with secondary y-axis
fig_vol_bubble = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
fig_vol_bubble.add_trace(
    go.Bar(
        x=xusd_data.index,
        y=xusd_data["Volume"],
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
        x=xusd_data.index,
        y=xusd_data["Last"],
        mode="markers+lines",
        marker=dict(
            size=xusd_data["norm_volume"] * bubble_size,  # Scale marker size
            color=xusd_data["color"],  # Assign color based on 'daily_change'
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
    xaxis_rangeslider_visible=False,
    title=f"{selected_crypto} Bubble Volume Chart {datasource.split('/')[0]}",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
)
st.plotly_chart(fig_vol_bubble)
# # # end - plot volume bubble # # #


# # # start - newmoon fullmoon chart # # #

# TODO change it like the Volume Support chart so i can zoom in and have nice candles!

# take the xusd_data index and iterate over it to get the moon phase for each day
# the make a new df out of it called df_moon_phase
# Initialize an empty list to hold the rows.
list_of_rows = []

# Loop to populate the list_of_rows
for i in xusd_data.index:
    list_of_rows.append([i, moon.phase(i)])

# Convert the list of rows to a DataFrame.
df_moon_phase = pd.DataFrame(list_of_rows, columns=["date", "moon_phase"])


df_moon_phase["moon_phase"] = df_moon_phase["moon_phase"].fillna(0).astype(int)
df_moon_phase["date"] = pd.to_datetime(df_moon_phase["date"])
df_moon_phase["date"] = df_moon_phase["date"].dt.date
df_moon_phase = df_moon_phase.set_index("date")

# check the relationship between moon phase and xusd_data
# merge sp500 with moon phase
xusd_data_and_moon_phase = pd.merge(
    xusd_data, df_moon_phase, left_index=True, right_index=True, how="left"
)
xusd_data_and_moon_phase = xusd_data_and_moon_phase.dropna()

# 0 .. 6.99	New moon
# 7 .. 13.99	First quarter
# 14 .. 20.99	Full moon
# 21 .. 27.99	Last quarter
# create a new column called moon_phase_category and fill it with the moon_phase_category
xusd_data_and_moon_phase["moon_phase_category"] = ""
xusd_data_and_moon_phase.loc[
    (xusd_data_and_moon_phase["moon_phase"] >= 0)
    & (xusd_data_and_moon_phase["moon_phase"] <= 6.99),
    "moon_phase_category",
] = 0
xusd_data_and_moon_phase.loc[
    (xusd_data_and_moon_phase["moon_phase"] >= 7)
    & (xusd_data_and_moon_phase["moon_phase"] <= 13.99),
    "moon_phase_category",
] = 7
xusd_data_and_moon_phase.loc[
    (xusd_data_and_moon_phase["moon_phase"] >= 14)
    & (xusd_data_and_moon_phase["moon_phase"] <= 20.99),
    "moon_phase_category",
] = 14
xusd_data_and_moon_phase.loc[
    (xusd_data_and_moon_phase["moon_phase"] >= 21)
    & (xusd_data_and_moon_phase["moon_phase"] <= 27.99),
    "moon_phase_category",
] = 21


# make a new plot and plot the daily_price_change of btc by time on the x-axis
# fig_btc_moon = make_subplots(specs=[[{"secondary_y": True}]])
# Add the scatter plot for the 'Last' data
fig_btc_moon = go.Figure(
    data=[
        go.Candlestick(
            x=xusd_data_and_moon_phase.index,
            open=xusd_data_and_moon_phase["First"],
            high=xusd_data_and_moon_phase["High"],
            low=xusd_data_and_moon_phase["Low"],
            close=xusd_data_and_moon_phase["Last"],
        )
    ]
)
# Create empty lists to hold the x, y, and text values
x_new_moon = []
x_full_moon = []
y_new_moon = []
y_full_moon = []
text_new_moon = []
text_full_moon = []

last_moon_phase = None
# Loop over the dataframe
for i in range(len(xusd_data_and_moon_phase)):
    current_moon_phase = xusd_data_and_moon_phase["moon_phase_category"].iloc[i]
    if current_moon_phase != last_moon_phase:
        if current_moon_phase == 0:
            # Add the date to the x list, 0.9 to the y list, and the hover text to the text list
            x_new_moon.append(xusd_data_and_moon_phase.index[i])
            y_new_moon.append(xusd_data_and_moon_phase.Last[i] * 1.2)
            text_new_moon.append(f"New Moon: {xusd_data_and_moon_phase.index[i]}")
        elif current_moon_phase == 14:
            # Add the date to the x list, 0 to the y list, and the hover text to the text list
            x_full_moon.append(xusd_data_and_moon_phase.index[i])
            y_full_moon.append(xusd_data_and_moon_phase.Last[i] * 0.8)
            text_full_moon.append(f"Full Moon: {xusd_data_and_moon_phase.index[i]}")
        last_moon_phase = current_moon_phase

# Add a scatter plot to the figure for the new moon points
fig_btc_moon.add_trace(
    go.Scatter(
        x=x_new_moon,
        y=y_new_moon,
        mode="markers",
        marker=dict(symbol="circle", size=10, color="gray"),
        text=text_new_moon,
        hoverinfo="text",
        showlegend=False,
    )
)

# Add a scatter plot to the figure for the full moon points
fig_btc_moon.add_trace(
    go.Scatter(
        x=x_full_moon,
        y=y_full_moon,
        mode="markers",
        marker=dict(symbol="circle", size=10, color="yellow"),
        text=text_full_moon,
        hoverinfo="text",
        showlegend=False,
    )
)


fig_btc_moon.add_trace(
    go.Scatter(
        x=xusd_data_and_moon_phase.index,
        y=xusd_data_and_moon_phase.moon_phase,
        mode="lines",
        hoverinfo="text+y",  # It will show the hover text and y-value
        hovertext=xusd_data_and_moon_phase.moon_phase.apply(
            lambda x: f"Moon Phase: {x}"
        ),  # Showing moon phase value as hover text
        yaxis="y2",  # Specifies the usage of secondary y-axis
    )
)

# Update the layout to show the secondary y-axis on the right and other customizations
fig_btc_moon.update_layout(
    xaxis_title="Date",
    yaxis_title="Last",
    yaxis2=dict(
        title="Moon Phase",
        overlaying="y",  # Ensures it overlays on the primary y-axis
        side="right",  # Positioning the y-axis on the right
    ),
    title=f"{selected_crypto} moon Chart {datasource.split('/')[0]}",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig_btc_moon)
# # # end - plot volume bubble # # #


# # # # start - plot Daily Volume Support Resistance Zones # # #
# num_zones = st.number_input("Enter the number of zones", value=12)
# # num_zones_int = int(num_zones)

# # Calculate frequencies of prices
# vol_profile = xusd_data_and_moon_phase["Last"].value_counts().nlargest(num_zones)
# print(vol_profile)

# # Define support and resistance levels
# support_levels = vol_profile.index.sort_values()
# resistance_levels = support_levels[1:]

# # Normalize volumes to range 1-5 for line widths
# vol_normalized = (vol_profile - vol_profile.min()) / (
#     vol_profile.max() - vol_profile.min()
# ) * 4 + 1

# # Plot price and volume profile (support and resistance levels)
# fig_volume_sup_res = go.Figure(
#     data=[
#         go.Candlestick(
#             x=xusd_data_and_moon_phase.index,
#             open=xusd_data_and_moon_phase["First"],
#             high=xusd_data_and_moon_phase["High"],
#             low=xusd_data_and_moon_phase["Low"],
#             close=xusd_data_and_moon_phase["Last"],
#         )
#     ]
# )

# # Add support and resistance lines
# for level in support_levels:
#     width = vol_normalized[level]
#     fig_volume_sup_res.add_trace(
#         go.Scatter(
#             x=xusd_data_and_moon_phase.index,
#             y=[level] * len(xusd_data_and_moon_phase.index),
#             mode="lines",
#             line=dict(width=width),
#             name=f"Vol: {vol_profile[level]}",
#         )
#     )

# fig_volume_sup_res.update_layout(
#     xaxis_title="Date",
#     yaxis_title="Last",
#     # xaxis_rangeslider_visible=True,
#     title=f"{selected_crypto} Daily Volume Support Resistance Zones {datasource.split('/')[0]}",
#     autosize=False,
#     width=int(1400 / 1),
#     height=int(800 / 1),
#     xaxis_rangeslider_visible=False,
# )

# st.plotly_chart(fig_volume_sup_res)
# # # # end - plot Daily Volume Support Resistance Zones # # #


# # # start - slider plot Daily Volume Support Resistance Zones # # #
# Create columns for layout
col1, col2, col3 = st.columns(3)
# Create a slider to select the number of days to go back
with col1:
    days_to_include = st.slider(
        "Days to consider:",
        min_value=5,
        max_value=len(xusd_data_and_moon_phase),
        value=len(xusd_data_and_moon_phase),
    )

with col2:
    num_bars = st.number_input("Bars:", value=12)

with col3:
    bin_percentage = st.number_input("Bin distance percentage:", value=10)

# Filter the data based on the days selected
xusd_data_and_moon_phase = xusd_data_and_moon_phase.iloc[-days_to_include:]

xusd_data_and_moon_phase["Bin_Width"] = (
    xusd_data_and_moon_phase["Last"] * bin_percentage / 100
)
bins = [
    xusd_data_and_moon_phase["Last"].iloc[0]
    - xusd_data_and_moon_phase["Bin_Width"].iloc[0]
]
for i in range(1, len(xusd_data_and_moon_phase)):
    bins.append(
        bins[-1]
        + (
            xusd_data_and_moon_phase["Bin_Width"].iloc[i]
            + xusd_data_and_moon_phase["Bin_Width"].iloc[i - 1]
        )
        / 2
    )

xusd_data_and_moon_phase["Bin"] = pd.cut(
    xusd_data_and_moon_phase["Last"], bins, labels=(bins[:-1]), include_lowest=True
)
binned_volume = xusd_data_and_moon_phase.groupby("Bin").size()
top_bins = binned_volume.nlargest(num_bars)

vol_normalized = (top_bins - top_bins.min()) / (top_bins.max() - top_bins.min()) * 4 + 1

fig_volume_sup_res_slider = go.Figure(
    data=[
        go.Candlestick(
            x=xusd_data_and_moon_phase.index,
            open=xusd_data_and_moon_phase["First"],
            high=xusd_data_and_moon_phase["High"],
            low=xusd_data_and_moon_phase["Low"],
            close=xusd_data_and_moon_phase["Last"],
        )
    ]
)

for level, vol in top_bins.items():
    width = vol_normalized[level]
    fig_volume_sup_res_slider.add_trace(
        go.Scatter(
            x=xusd_data_and_moon_phase.index,
            y=[level] * len(xusd_data_and_moon_phase.index),
            mode="lines",
            line=dict(width=width),
            name=f"Vol: {vol}",
        )
    )

fig_volume_sup_res_slider.update_layout(
    xaxis_title="Date",
    yaxis_title="Last",
    # xaxis_rangeslider_visible=True,
    title=f"{selected_crypto} Daily Volume Support Resistance Zones {datasource.split('/')[0]}",
    autosize=False,
    width=int(1400 / 1),
    height=int(800 / 1),
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig_volume_sup_res_slider)
# # # end - slider plot Daily Volume Support Resistance Zones # # #
