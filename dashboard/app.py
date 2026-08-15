"""
§9 — Alpha Flow live dashboard (Streamlit).
Run: streamlit run dashboard/app.py

Panels:
  - KPI row: Sharpe, MFG residual, regime, signal count
  - Price fan chart: 60-day MFG median + 10%-90% CI
  - Capital flow by agent type
  - Dual noise bar (σ_τ vs η by ticker)
  - Lyapunov gauge (RiskIndex, updated every 15min)
  - Live P&L (Alpaca paper account)
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="Alpha Flow", layout="wide", page_icon="📈")
st.title("Alpha Flow — World Model Dashboard")


@st.cache_data(ttl=60)
def load_predictions() -> dict:
    """Load latest model outputs from DB. Replace with real DB query."""
    n = 60
    t_hist = pd.date_range("2024-01-01", periods=120, freq="B")
    price_hist = np.cumprod(1 + np.random.randn(120) * 0.01) * 100
    t_fut_future = pd.date_range(t_hist[-1] + pd.offsets.BDay(1), periods=n, freq="B")
    raw_forecast = np.cumprod(1 + np.random.randn(n) * 0.008)
    p50_future = price_hist[-1] * raw_forecast

    # Last historical point
    t_fut = t_hist[-1:].union(t_fut_future)
    p50 = np.insert(p50_future, 0, price_hist[-1])
    p10 = np.insert(p50_future * 0.92, 0, price_hist[-1])
    p90 = np.insert(p50_future * 1.08, 0, price_hist[-1])
    return {
        "sharpe": 1.72, "sharpe_delta": 0.08,
        "mfg_residual": 0.0073,
        "regime": "Normal", "lyapunov_indicator": "0.31",
        "n_signals": 47, "cvar": 0.028,
        "t_hist": t_hist, "price_hist": price_hist,
        "t_fut": t_fut, "p50": p50, "p10": p10, "p90": p90,
        "flow_df": pd.DataFrame({
            "agent_type": ["Quant HF", "Passive ETF", "Momentum", "Retail"],
            "today":    [+1.2, -0.3, +0.8, -0.5],
            "tomorrow": [+0.9, -0.1, +1.1, -0.3],
        }),
        "noise_df": pd.DataFrame({
            "ticker":    ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
            "sigma_tau": [0.18, 0.16, 0.31, 0.42, 0.22],
            "lambda_eta":[0.04, 0.02, 0.09, 0.14, 0.05],
        }),
    }


preds = load_predictions()

# ── KPI row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Sharpe (30d)", f"{preds['sharpe']:.2f}",
            delta=f"{preds['sharpe_delta']:+.2f}")
col2.metric("MFG Residual", f"{preds['mfg_residual']:.4f}",
            delta="✅ converged" if preds["mfg_residual"] < 0.01 else "⚠️ high")
col3.metric("Market Regime", preds["regime"],
            delta=f"RiskIndex={preds['lyapunov_indicator']}")
col4.metric("Active Signals", f"{preds['n_signals']}",
            delta=f"CVaR={preds['cvar']:.1%}")

st.divider()

# ── Price fan chart ───────────────────────────────────────────────────────────
fig_price = go.Figure()
# 1. CI Band first
fig_price.add_trace(go.Scatter(
    x=list(preds["t_fut"]) + list(reversed(preds["t_fut"])),
    y=list(preds["p10"]) + list(reversed(preds["p90"])),
    fill="toself", fillcolor="rgba(93,173,226,0.18)",
    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="10%–90% CI"))

# 2. Historical line
fig_price.add_trace(go.Scatter(
    x=preds["t_hist"], y=preds["price_hist"],
    name="Historical", line=dict(color="white", width=2)))

# 3. Median line 
fig_price.add_trace(go.Scatter(
    x=preds["t_fut"], y=preds["p50"],
    name="MFG Median", line=dict(color="orange", width=2)))

fig_price.update_layout(template="plotly_dark", title="60-Day Price Forecast (MFG equilibrium paths)")
st.plotly_chart(fig_price, use_container_width=True)

col_left, col_right = st.columns(2)

# ── Agent capital flow ────────────────────────────────────────────────────────
with col_left:
    fig_flow = px.bar(
        preds["flow_df"], x="agent_type", y=["today", "tomorrow"],
        barmode="group",
        color_discrete_map={"today": "#2471A3", "tomorrow": "#F39C12"},
        title="Predicted Capital Flow by Agent Type (% NAV)",
    )
    fig_flow.update_layout(template="plotly_dark")
    st.plotly_chart(fig_flow, use_container_width=True)

# ── Dual noise decomposition ──────────────────────────────────────────────────
with col_right:
    ndf = preds["noise_df"]
    fig_noise = go.Figure()
    fig_noise.add_trace(go.Bar(name="σ_τ (Physical)", x=ndf["ticker"], y=ndf["sigma_tau"],
                                marker_color="#1ABC9C"))
    fig_noise.add_trace(go.Bar(name="λ_η (Behavioral jump intensity)",
                                x=ndf["ticker"], y=ndf["lambda_eta"],
                                marker_color="#E74C3C"))
    fig_noise.update_layout(
        template="plotly_dark", barmode="group",
        title="Dual Noise Decomposition by Ticker (§III Calibration)",
    )
    st.plotly_chart(fig_noise, use_container_width=True)
