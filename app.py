import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="PGCB Power Generation Decision Support System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ PGCB Power Generation Forecasting Dashboard")
st.markdown("**Web-Based Remote Monitoring & Decision Support System (PGCB Bangladesh Data)**")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('model_results.csv', parse_dates=['datetime'], index_col='datetime')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: `model_results.csv` not found. Run `python train_models.py` first.")
    st.stop()

TARGET_COL = 'generation_mw'

# Sidebar Controls
st.sidebar.header("Dashboard Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["XGBoost (Classical ML)", "LSTM (Deep Learning)"]
)
pred_col = 'XGBoost_Pred' if "XGBoost" in model_choice else 'LSTM_Pred'

min_date = df.index.min().date()
max_date = df.index.max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered_df = df.loc[mask]

# Summary Metrics Panel
st.subheader("Performance Overview")

if len(filtered_df) > 0:
    latest_val = filtered_df[TARGET_COL].iloc[-1]
    peak_val = filtered_df[TARGET_COL].max()
    avg_val = filtered_df[TARGET_COL].mean()

    mae = mean_absolute_error(filtered_df[TARGET_COL], filtered_df[pred_col])
    rmse = np.sqrt(mean_squared_error(filtered_df[TARGET_COL], filtered_df[pred_col]))
    mape = np.mean(np.abs((filtered_df[TARGET_COL] - filtered_df[pred_col]) / filtered_df[TARGET_COL])) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Generation", f"{latest_val:,.1f} MW")
    c2.metric("Peak Generation", f"{peak_val:,.1f} MW")
    c3.metric("Average Generation", f"{avg_val:,.1f} MW")
    c4.metric("MAE", f"{mae:,.1f} MW")
    c5.metric("MAPE Error", f"{mape:.2f}%")

st.markdown("---")

# Interactive Chart with 95% Confidence Band
st.subheader("📈 Historical vs Forecasted Generation")

if len(filtered_df) > 0:
    residuals = filtered_df[TARGET_COL] - filtered_df[pred_col]
    std_err = np.std(residuals)
    upper_bound = filtered_df[pred_col] + (1.96 * std_err)
    lower_bound = filtered_df[pred_col] - (1.96 * std_err)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df.index, y=upper_bound,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df.index, y=lower_bound,
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)', name='95% Confidence Band'
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df.index, y=filtered_df[TARGET_COL],
        mode='lines', name='Actual Generation (MW)', line=dict(color='#1f77b4', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df.index, y=filtered_df[pred_col],
        mode='lines', name=f'Forecast ({model_choice})', line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"Generation Forecast ({start_date} to {end_date})",
        xaxis_title="DateTime",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("View Raw Forecast Data"):
    st.dataframe(filtered_df[[TARGET_COL, 'XGBoost_Pred', 'LSTM_Pred']].tail(100))