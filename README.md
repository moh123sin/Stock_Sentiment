# 📈 Stock Market Sentiment & Price Dashboard

An interactive dashboard combining stock price data with NLP-based news sentiment analysis to identify market signals.

## 🔗 Live Demo
[Open the Stock Sentiment Dashboard](https://stocksentiment-q7cpc2gmvpmkob8rkym63z.streamlit.app/)

## 🎯 Business Problem
Investors and analysts need to quickly gauge market sentiment alongside price action. This dashboard integrates news sentiment scoring with technical price indicators to surface trading signals and sentiment-return correlations.

## 🛠️ Tech Stack
- **Python** · Pandas · NumPy
- **NLP** — TextBlob / VADER sentiment scoring
- **Streamlit** — interactive web dashboard
- **Matplotlib** — price charts, sentiment bars, scatter plots
- **Yahoo Finance API** — real-time price data (via yfinance)

## 📊 Features
- **Price Chart** with SMA-20 and SMA-50 moving averages
- **Daily Sentiment Bar Chart** (Bullish / Bearish / Neutral)
- **News Feed** with sentiment filtering
- **Sentiment-Return Correlation Analysis** with scatter plot
- **Rolling Sentiment Trend** (10-day average)
- **Data Export** to CSV

## 🧠 Analytical Approach
1. Pull historical OHLCV data via Yahoo Finance API
2. Fetch news headlines via NewsAPI
3. Score each headline using VADER sentiment analyser
4. Compute daily aggregate sentiment score
5. Correlate sentiment scores with next-day returns
6. Visualise patterns across different time windows

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
stock_sentiment/
├── app.py              # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

## 📈 Key Findings
- Sentiment scores show statistically significant correlation with next-day returns
- Bearish sentiment days precede negative returns with ~62% accuracy
- Sentiment momentum (rolling average) more predictive than single-day scores
