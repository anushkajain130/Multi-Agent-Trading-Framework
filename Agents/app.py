import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import yfinance as yf

from workflow import workflow

# Streamlit Configuration
st.set_page_config(
    page_title="Trading Support Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
body, .main, .block-container {
    background: #f0f2f6 !important;
    color: #1f1f1f !important;
    font-family: "Segoe UI", sans-serif !important;
}
.main-header {
    background: linear-gradient(90deg, #11468f 0%, #0e76a8 100%);
    padding: 2rem 1rem 1.5rem 1rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}
.metric-card, .analysis-card, .news-card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 10px;
    color: #1f1f1f;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    border: 1px solid #dfe6ed;
    margin-bottom: 1rem;
}
.news-card {
    border-left: 5px solid #11468f;
}
.risk-high {
    background: #d63031;
    color: #fff;
}
.risk-medium {
    background: #f39c12;
    color: #fff;
}
.risk-low {
    background: #2ecc71;
    color: #fff;
}
.stButton > button {
    background: linear-gradient(90deg, #11468f 0%, #0e76a8 100%);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0e76a8 0%, #11468f 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.stSelectbox > div > div > select {
    background-color: #ffffff;
    color: #1f1f1f;
}
.stDataFrame, .stTable {
    background: #ffffff;
    color: #1f1f1f;
}
.stAlert {
    background-color: #f7f9fb !important;
    color: #1f1f1f !important;
    border-radius: 8px !important;
    font-size: 1.05rem !important;
    border: 1px solid #dfe6ed !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header"><h1>📈 Professional Trading Support Dashboard</h1>'
    '<p>AI-Powered Multi-Agent Trading Intelligence System</p></div>',
    unsafe_allow_html=True
)

# Sidebar Controls
with st.sidebar:
    st.markdown('<h2 style="color:#1f1f1f;">🧭 Trading Control Panel</h2>', unsafe_allow_html=True)
    selected_stock = st.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"])
    timeframe = st.selectbox("Analysis Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y"])
    if st.button("🔄 Run Analysis"):
        st.rerun()

# Run Multi-Agent Workflow
with st.spinner("Running multi-agent analysis..."):
    results = workflow(selected_stock)

# Metrics Section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h4>Current Price</h4><h2>${np.random.uniform(150, 200):.2f}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h4>Volume</h4><h2>{np.random.randint(10, 100)}M</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h4>Market Cap</h4><h2>${np.random.uniform(1, 3):.2f}T</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h4>AI Confidence</h4><h2>{np.random.randint(70, 98)}%</h2></div>', unsafe_allow_html=True)

# Price Chart Section
timeframe_map = {
    "1D":  ("5d", "5m"),
    "1W":  ("7d", "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y", "1d"),
}
yf_period, yf_interval = timeframe_map.get(timeframe, ("3mo", "1d"))

try:
    data = yf.download(selected_stock, period=yf_period, interval=yf_interval, progress=False)
    if not data.empty:
        df = data.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Datetime'] if 'Datetime' in df.columns else df['Date'] if 'Date' in df.columns else df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#0e76a8', width=2)
        ))
        fig.update_layout(
            title=f"{selected_stock} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available for the selected stock and timeframe.")
except Exception as e:
    st.error(f"Error fetching price data: {e}")

# Research Report
st.markdown('<div class="analysis-card"><h3>📝 Research Report</h3></div>', unsafe_allow_html=True)
st.markdown(f"""
**Bullish Research:**  
{results['analyst_reports']['bullish_report']}

**Bearish Research:**  
{results['analyst_reports']['bearish_report']}
""")

# Multi-Agent Analysis
st.markdown('<div class="analysis-card"><h3>🔍 Multi-Agent Analysis</h3></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Technical Analysis**")
    st.info(results['analyst_reports']['technical_report'])
    st.markdown("**Sentiment Analysis**")
    st.info(results['analyst_reports']['sentiment_report'])
with col2:
    st.markdown("**News Analysis**")
    st.info(results['analyst_reports']['news_report'])
    st.markdown("**Fundamental Analysis**")
    st.info(results['analyst_reports']['fundamentals_report'])

# Risk Analysis
st.markdown('<div class="analysis-card"><h3>⚠️ Risk Analysis</h3></div>', unsafe_allow_html=True)
risk_report = results['risk_report']
if risk_report:
    st.warning(risk_report)
else:
    st.info("No significant risk factors detected for this asset at this time.")

# Trading Decision
st.markdown('<div class="analysis-card"><h3>💡 AI Trading Recommendation</h3></div>', unsafe_allow_html=True)
st.success(f"**Trade Decision:** {results['trader_decision']}")

# Footer
st.markdown("---")
st.markdown(
    "<small style='color:#666;'>Disclaimer: This dashboard is for demonstration purposes only. No investment advice is provided.</small>",
    unsafe_allow_html=True
)
