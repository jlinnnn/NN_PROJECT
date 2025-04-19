
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

with open("my_deepar_forecast_results.pkl", "rb") as f:
    forecast_data = pickle.load(f)

st.set_page_config(
    page_title="DeepAR Stock Forecast Dashboard", layout="centered")

selected_ticker = st.selectbox("Choose a Ticker", sorted(
    forecast_data.keys()), index=sorted(forecast_data.keys()).index("AAPL"))
forecast = forecast_data[selected_ticker]

# Determine risk level from first day's VaR
high_risk = forecast["var_99_5day"][0] < -0.03
title_emoji = "📉" if high_risk else "📈"
st.title(f"{title_emoji} DeepAR Stock Forecast Dashboard")


st.markdown("""
This dashboard shows 5-day probabilistic return forecasts for S&P 500 stocks using a DeepAR time series model.
Select a stock to view predicted return ranges and Value-at-Risk (VaR).
""")

# Explanation of metrics
with st.expander("ℹ️ What do these numbers mean?"):
    st.markdown("""
    - **Mean**: The average predicted return for that day.
    - **Median**: The middle prediction (50% of outcomes are above and below).
    - **P10 / P90**: The 10th and 90th percentile — 80% of predicted outcomes fall within this range.
    - **Max Loss (99%)**: The worst-case 1-day loss expected with 99% confidence.
    - 🟣 **Note on Purple Points**: Represent the upper bound of the 80% interval (P90), helping visualize forecast uncertainty.
    """)

# Forecast Summary Table
st.subheader(f"📊 Forecast Summary: {selected_ticker}")


def flatten(x):
    return list(x) if isinstance(x, (np.ndarray, list)) else [x]


metrics_df = pd.DataFrame({
    "Mean": flatten(forecast["mean"]),
    "Median": flatten(forecast["median"]),
    "P10": flatten(forecast["p10"]),
    "P90": flatten(forecast["p90"]),
    "Max Loss (99%)": flatten(forecast["var_99_5day"])
})
metrics_df = metrics_df.apply(pd.to_numeric, errors="coerce")
metrics_df.index = [f"Day {i+1}" for i in range(len(metrics_df))]
st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)

# Download Forecast
csv = metrics_df.to_csv(index=True)
st.download_button(
    label="Download Forecast Data as CSV",
    data=csv,
    file_name=f'{selected_ticker}_forecast.csv',
    mime='text/csv',
)

# Forecast Visualization


def plot_forecast_plotly(forecast, ticker):
    days = np.arange(1, len(forecast["median"]) + 1)

    fig = go.Figure()

    # P90 for interval shading
    fig.add_trace(go.Scatter(
        x=days, y=forecast["p90"],
        mode='lines',
        line=dict(width=0),
        name='',
        hoverinfo='skip',
        showlegend=False,
    ))

    # P10 fill below P90
    fig.add_trace(go.Scatter(
        x=days, y=forecast["p10"],
        fill='tonexty',
        fillcolor='rgba(0, 200, 0, 0.2)',
        line=dict(width=0),
        name='80% Prediction Interval',
        hovertemplate="Day %{x}<br>Return: %{y:.2%}<extra>Lower Bound (P10)</extra>"
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=days, y=forecast["median"],
        mode='lines+markers',
        name='Median Forecast',
        line=dict(color='green', width=2),
        hovertemplate="Day %{x}<br>Return: %{y:.2%}<extra>Median Forecast</extra>"
    ))

    # Max Loss (VaR 99%)
    fig.add_trace(go.Scatter(
        x=days, y=forecast["var_99_5day"],
        mode='lines+markers',
        name='Max Loss (99%)',
        line=dict(color='red', dash='dot'),
        hovertemplate="Day %{x}<br>Return: %{y:.2%}<extra>Max Loss (99%)</extra>"
    ))

    # P90
    fig.add_trace(go.Scatter(
        x=days, y=forecast["p90"],
        mode='markers',
        marker=dict(color='purple'),
        name='90th Percentile (P90)',
        hovertemplate="Day %{x}<br>Return: %{y:.2%}<extra>Upper Bound (P90)</extra>"
    ))

    fig.update_layout(
        title=f"{ticker} 5-Day Return Forecast",
        xaxis_title="Day",
        yaxis_title="Predicted Return",
        yaxis_tickformat=".2%",
        template="plotly_dark",
        height=500,
    )
    return fig


st.plotly_chart(plot_forecast_plotly(
    forecast, selected_ticker), use_container_width=True)

# Risk Banner
if high_risk:
    st.warning(
        "⚠️ High downside risk: The model predicts a potential 5-day loss greater than 3% with 99% confidence.")
else:
    st.success("✅ Forecast suggests relatively low near-term risk.")
