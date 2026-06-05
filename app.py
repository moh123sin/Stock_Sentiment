import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide", page_icon="📈")

st.markdown("""
<style>
.bull  {color:#27ae60;font-weight:bold;}
.bear  {color:#e74c3c;font-weight:bold;}
.neutral {color:#f39c12;font-weight:bold;}
h1 {color:#2E74B5;}
.card {background:#f8f9fa;border-radius:8px;padding:15px;border-left:4px solid #2E74B5;margin:5px 0;}
</style>""", unsafe_allow_html=True)

# ── Simulated data (mimics real Yahoo Finance + NewsAPI output) ───────────────
@st.cache_data
def generate_stock_data(ticker, days=180):
    np.random.seed(hash(ticker) % 1000)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")

    # Price simulation (GBM)
    start_prices = {"AAPL":175, "MSFT":380, "GOOGL":140, "TSLA":220, "AMZN":185, "NVDA":650}
    S0 = start_prices.get(ticker, 100)
    mu, sigma = 0.0003, 0.018
    returns = np.random.normal(mu, sigma, days)
    prices  = S0 * np.exp(np.cumsum(returns))

    # Volume
    volume = np.random.randint(20_000_000, 80_000_000, days)

    # Sentiment scores (-1 to 1) correlated with next-day returns
    sentiment = np.clip(returns * 30 + np.random.normal(0, 0.3, days), -1, 1)

    # News headlines simulation
    headlines_pos = [
        "Strong earnings beat analyst expectations",
        "Company announces record revenue quarter",
        "New product launch drives investor optimism",
        "Partnership deal boosts stock outlook",
        "Analyst upgrades to Strong Buy with price target raised",
    ]
    headlines_neg = [
        "Revenue misses estimates amid macro headwinds",
        "Regulatory probe weighs on investor sentiment",
        "Supply chain disruptions impact guidance",
        "CEO departure raises governance concerns",
        "Competition intensifies in core market segment",
    ]
    headlines_neu = [
        "Company maintains full-year guidance",
        "Board approves share buyback program",
        "Quarterly dividend announced as expected",
        "Management presents at investor conference",
        "Annual report filed with SEC",
    ]

    news = []
    for i, s in enumerate(sentiment):
        if s > 0.2:
            news.append(np.random.choice(headlines_pos))
        elif s < -0.2:
            news.append(np.random.choice(headlines_neg))
        else:
            news.append(np.random.choice(headlines_neu))

    df = pd.DataFrame({
        "date": dates, "close": prices.round(2),
        "volume": volume, "sentiment": sentiment.round(3),
        "headline": news
    })
    df["return_1d"]  = df["close"].pct_change().round(4)
    df["sma_20"]     = df["close"].rolling(20).mean().round(2)
    df["sma_50"]     = df["close"].rolling(50).mean().round(2)
    df["volatility"] = df["return_1d"].rolling(20).std().round(4)
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: "Bullish" if x > 0.1 else ("Bearish" if x < -0.1 else "Neutral"))
    return df

@st.cache_data
def sentiment_correlation(df):
    """Compute correlation between sentiment and next-day return."""
    df2 = df.copy()
    df2["next_return"] = df2["return_1d"].shift(-1)
    return df2[["sentiment","next_return"]].corr().iloc[0,1]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
ticker   = st.sidebar.selectbox("Select Stock", ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA"])
period   = st.sidebar.selectbox("Time Period", ["30 Days","90 Days","180 Days"], index=1)
days_map = {"30 Days":30, "90 Days":90, "180 Days":180}
days     = days_map[period]

df = generate_stock_data(ticker, 180).tail(days).reset_index(drop=True)
corr = sentiment_correlation(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Market Sentiment Dashboard")
st.markdown(f"*Real-time sentiment analysis and price intelligence for **{ticker}***")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
latest_price   = df["close"].iloc[-1]
price_change   = df["close"].pct_change().iloc[-1] * 100
avg_sentiment  = df["sentiment"].mean()
bullish_days   = (df["sentiment_label"] == "Bullish").sum()
bearish_days   = (df["sentiment_label"] == "Bearish").sum()

k1.metric(f"{ticker} Price",    f"${latest_price:.2f}", f"{price_change:+.2f}%")
k2.metric("Avg Sentiment",      f"{avg_sentiment:+.3f}")
k3.metric("Bullish Days",       f"{bullish_days}")
k4.metric("Bearish Days",       f"{bearish_days}")
k5.metric("Sentiment-Return ρ", f"{corr:.3f}")

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price & Sentiment", "📰 News Feed", "🔬 Analysis", "📋 Data Table"])

# ════════════════════════════════════════════════════════
# TAB 1 — Price & Sentiment
# ════════════════════════════════════════════════════════
with tab1:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                         gridspec_kw={"height_ratios":[3,1.5,1]})

    # Price + MAs
    ax1.plot(df["date"], df["close"],  color="#2E74B5", lw=1.5, label="Price")
    ax1.plot(df["date"], df["sma_20"], color="#e67e22", lw=1,   linestyle="--", label="SMA 20", alpha=0.8)
    ax1.plot(df["date"], df["sma_50"], color="#8e44ad", lw=1,   linestyle="--", label="SMA 50", alpha=0.8)
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{ticker} Price with Moving Averages")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # Sentiment
    colors = ["#27ae60" if s > 0.1 else "#e74c3c" if s < -0.1 else "#f39c12"
              for s in df["sentiment"]]
    ax2.bar(df["date"], df["sentiment"], color=colors, alpha=0.7, width=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Daily News Sentiment")
    ax2.grid(alpha=0.3)

    # Volume
    ax3.bar(df["date"], df["volume"]/1e6, color="#5BA3E0", alpha=0.6)
    ax3.set_ylabel("Volume (M)")
    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 2 — News Feed
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📰 Recent News Headlines — {ticker}")
    filter_sent = st.selectbox("Filter by Sentiment", ["All","Bullish","Bearish","Neutral"])

    news_df = df[["date","headline","sentiment","sentiment_label"]].tail(30).sort_values("date", ascending=False)
    if filter_sent != "All":
        news_df = news_df[news_df["sentiment_label"] == filter_sent]

    for _, row in news_df.iterrows():
        icon  = "🟢" if row["sentiment_label"] == "Bullish" else "🔴" if row["sentiment_label"] == "Bearish" else "🟡"
        color = "bull" if row["sentiment_label"] == "Bullish" else "bear" if row["sentiment_label"] == "Bearish" else "neutral"
        st.markdown(f"""<div class='card'>
        {icon} <strong>{row['date'].strftime('%b %d, %Y')}</strong> &nbsp;
        <span class='{color}'>[{row['sentiment_label']} | Score: {row['sentiment']:+.2f}]</span><br>
        {row['headline']}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Analysis
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 Sentiment vs Price Return Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment Distribution**")
        fig, ax = plt.subplots(figsize=(5,3.5))
        sent_counts = df["sentiment_label"].value_counts()
        colors_pie  = ["#27ae60","#f39c12","#e74c3c"]
        ax.pie(sent_counts.values, labels=sent_counts.index, colors=colors_pie,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Sentiment Breakdown")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Sentiment vs Next-Day Return (Scatter)**")
        df2 = df.copy()
        df2["next_return"] = df2["return_1d"].shift(-1) * 100
        df2 = df2.dropna()
        fig, ax = plt.subplots(figsize=(5,3.5))
        scatter_colors = ["#27ae60" if s > 0.1 else "#e74c3c" if s < -0.1 else "#f39c12"
                          for s in df2["sentiment"]]
        ax.scatter(df2["sentiment"], df2["next_return"], c=scatter_colors, alpha=0.5, s=20)
        z = np.polyfit(df2["sentiment"], df2["next_return"], 1)
        p = np.poly1d(z)
        ax.plot(sorted(df2["sentiment"]), p(sorted(df2["sentiment"])), "b--", lw=1.5)
        ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Sentiment Score"); ax.set_ylabel("Next-Day Return (%)")
        ax.set_title(f"Correlation: {corr:.3f}")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Average Return by Sentiment Category**")
    df["next_return"] = df["return_1d"].shift(-1) * 100
    avg_ret = df.groupby("sentiment_label")["next_return"].agg(["mean","std","count"]).round(3)
    avg_ret.columns = ["Avg Return (%)","Std Dev (%)","Days"]
    st.dataframe(avg_ret, use_container_width=True)

    st.markdown("**30-Day Rolling Sentiment Trend**")
    fig, ax = plt.subplots(figsize=(11,3))
    rolling = df["sentiment"].rolling(10).mean()
    ax.plot(df["date"], rolling, color="#2E74B5", lw=2)
    ax.fill_between(df["date"], rolling, 0,
                    where=(rolling > 0), alpha=0.2, color="#27ae60", label="Positive")
    ax.fill_between(df["date"], rolling, 0,
                    where=(rolling < 0), alpha=0.2, color="#e74c3c", label="Negative")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("10-Day Rolling Average Sentiment")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 4 — Data Table
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Full Dataset")
    display_df = df[["date","close","return_1d","sma_20","sma_50","volume","sentiment","sentiment_label","headline"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True)
    st.download_button("📥 Download CSV",
                       data=display_df.to_csv(index=False),
                       file_name=f"{ticker}_sentiment_data.csv",
                       mime="text/csv")
