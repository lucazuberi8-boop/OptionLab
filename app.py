import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="OptionLab", page_icon="📈", layout="wide")

DATA_FILE = Path(__file__).parent / "optionlab_data.csv"
PRICE_COLUMNS = {
    "Apple (AAPL)": "AAPL_Adj_Close",
    "Microsoft (MSFT)": "MSFT_Adj_Close",
    "NVIDIA (NVDA)": "NVDA_Adj_Close",
}

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates(subset="Date")
    return data

def historical_volatility(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < 2:
        raise ValueError("Not enough price observations to estimate volatility.")
    return float(returns.std(ddof=1) * np.sqrt(252)), returns

def black_scholes(S, K, T, r, sigma, option_type):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Stock price, strike, maturity and volatility must be positive.")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "Call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.title("📈 OptionLab")
st.caption("An educational European option valuation tool using historical share-price data and the Black–Scholes model.")

if not DATA_FILE.exists():
    st.error("The file optionlab_data.csv is missing. Place it in the same folder as app.py.")
    st.stop()

data = load_data()
st.sidebar.header("1. Select underlying stock")
selected_stock = st.sidebar.selectbox("Stock", list(PRICE_COLUMNS.keys()))
column = PRICE_COLUMNS[selected_stock]
prices = data[column].dropna()
if prices.empty:
    st.error(f"No price data found for {selected_stock}.")
    st.stop()

volatility, returns = historical_volatility(prices)
latest_price = float(prices.iloc[-1])
latest_date = data.loc[prices.index[-1], "Date"]

st.subheader("Historical data")
c1, c2, c3 = st.columns(3)
c1.metric("Latest adjusted close", f"${latest_price:,.2f}", help=f"Last observation: {latest_date:%d %b %Y}")
c2.metric("Annualised historical volatility", f"{volatility:.2%}")
c3.metric("Daily observations", f"{len(prices):,}")

with st.expander("How is historical volatility calculated?"):
    st.latex(r"r_t=\\ln(P_t/P_{t-1})")
    st.latex(r"\\sigma_{annual}=s(r_t)\\sqrt{252}")
    st.write("We calculate daily logarithmic returns from adjusted closing prices, take their sample standard deviation, and annualise using 252 trading days. This is a historical estimate, not a forecast.")

chart_data = pd.DataFrame({"Date": data["Date"], "Adjusted close": data[column]})
fig = px.line(chart_data, x="Date", y="Adjusted close", title=f"{selected_stock} — adjusted closing prices")
st.plotly_chart(fig, use_container_width=True)

ret_chart = pd.DataFrame({"Date": data.loc[returns.index, "Date"], "Daily log return": returns.values})
fig_ret = px.line(ret_chart, x="Date", y="Daily log return", title="Daily logarithmic returns")
st.plotly_chart(fig_ret, use_container_width=True)

st.divider()
st.header("2. Black–Scholes option calculator")
st.write("The model below values a European option. Historical volatility is used as the default volatility assumption; you can override it.")

left, right = st.columns(2)
with left:
    option_type = st.radio("Option type", ["Call", "Put"], horizontal=True)
    S = st.number_input("Current stock price ($)", min_value=0.01, value=latest_price, step=1.0)
    K = st.number_input("Strike price ($)", min_value=0.01, value=round(latest_price / 5) * 5, step=1.0)
    T = st.number_input("Time to maturity (years)", min_value=0.01, max_value=10.0, value=0.5, step=0.1)
with right:
    r_pct = st.number_input("Risk-free interest rate (%)", min_value=-10.0, max_value=30.0, value=3.0, step=0.25)
    use_historical = st.checkbox("Use historical volatility as default", value=True)
    default_vol = volatility * 100 if use_historical else 20.0
    sigma_pct = st.number_input("Volatility (%)", min_value=0.01, max_value=500.0, value=float(default_vol), step=1.0)
    market_price = st.number_input("Observed option market premium ($, optional)", min_value=0.0, value=0.0, step=0.1)

try:
    price = black_scholes(S, K, T, r_pct / 100, sigma_pct / 100, option_type)
    intrinsic = max(S - K, 0.0) if option_type == "Call" else max(K - S, 0.0)
    time_value = max(price - intrinsic, 0.0)
    a, b, c = st.columns(3)
    a.metric(f"Black–Scholes {option_type} value", f"${price:.4f}")
    b.metric("Intrinsic value", f"${intrinsic:.4f}")
    c.metric("Time value", f"${time_value:.4f}")
    if market_price > 0:
        diff = market_price - price
        st.metric("Market premium minus theoretical value", f"${diff:.4f}",
                  help="Positive means the entered market premium is above the model value; negative means it is below.")
    st.caption("A model valuation is not a trading recommendation. Market prices can differ because of dividends, exercise style, transaction costs, liquidity and changing volatility.")
except ValueError as e:
    st.error(str(e))

st.subheader("Sensitivity to volatility")
vol_grid = np.linspace(max(0.01, sigma_pct * 0.5), sigma_pct * 1.5, 31)
sens = []
for v in vol_grid:
    sens.append({"Volatility": v, "Theoretical option value": black_scholes(S, K, T, r_pct / 100, v / 100, option_type)})
sens_df = pd.DataFrame(sens)
fig_sens = px.line(sens_df, x="Volatility", y="Theoretical option value",
                   title="Black–Scholes value under different volatility assumptions")
st.plotly_chart(fig_sens, use_container_width=True)

st.subheader("Option payoff at expiration")
terminal_prices = np.linspace(max(0.01, S * 0.5), S * 1.5, 100)
if option_type == "Call":
    payoffs = np.maximum(terminal_prices - K, 0)
else:
    payoffs = np.maximum(K - terminal_prices, 0)
payoff_df = pd.DataFrame({"Stock price at expiration ($)": terminal_prices, "Payoff ($)": payoffs})
fig_payoff = px.line(payoff_df, x="Stock price at expiration ($)", y="Payoff ($)", title=f"Expiration payoff — {option_type}")
st.plotly_chart(fig_payoff, use_container_width=True)

with st.expander("Data source and limitations"):
    st.write("Dataset supplied for this project: optionlab_data.csv. Record the original provider, download date and exact observation period in the project report.")
    st.write("The dataset contains adjusted daily closing prices. Adjusted prices are used to estimate historical volatility; the latest adjusted close is displayed as a reference price. For a live valuation, replace it with a contemporaneous market spot price.")
    st.write("The standard Black–Scholes formula assumes a European option, constant volatility and risk-free rate, lognormal stock-price dynamics, and no dividends. This implementation does not include a continuous dividend yield.")
