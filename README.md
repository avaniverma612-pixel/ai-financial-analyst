# AI Financial Analyst 📈

An AI-powered stock market analysis web application that combines historical market data, machine learning, technical indicators, and AI-assisted explanations to help users understand stock performance.

## Overview

Analysing stocks manually can be time-consuming. Investors often need to collect price data, interpret market indicators, study trends, and compare different signals.

**AI Financial Analyst** brings these steps together in one interactive dashboard. Users can enter a company name or ticker symbol, select a forecast horizon, and explore the stock's historical performance and experimental machine-learning predictions.

The application is designed for educational analysis and decision support. Its predictions are not guaranteed and should not be treated as financial advice.

## Features

- **Stock Analysis:** Retrieve historical stock-market data using Yahoo Finance.
- **Price Trend Visualisation:** View stock prices alongside a five-session moving average.
- **Technical Indicators:** Calculate moving-average ratio, momentum, RSI, daily returns, and volatility.
- **Machine Learning:** Compare three models:
  - Logistic Regression — predicts UP or DOWN.
  - Decision Tree — classifies BUY, HOLD, or SELL.
  - Random Forest — classifies BUY, HOLD, or SELL.
- **AI Financial Analyst:** Generate natural-language interpretations of the available analysis.
- **Financial Chatbot:** Ask questions about stock analysis and financial indicators.
- **Global Market Pulse:** Explore daily movements across selected markets in the United States, United Kingdom, Germany, India, Hong Kong, and Japan.
- **Analysis Report:** Download a report containing analysis and methodology information.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive web interface |
| Yahoo Finance / yfinance | Historical market data |
| Pandas and NumPy | Data processing and indicator calculations |
| Scikit-learn | Machine-learning models and evaluation |
| Altair, Plotly, Matplotlib | Data visualisation |
| CrewAI and Groq | AI-assisted analysis and explanations |

## How It Works

1. The user enters a stock ticker and selects a forecast horizon.
2. Historical market data is retrieved from Yahoo Finance.
3. The application calculates technical indicators.
4. Machine-learning models analyse historical patterns and generate classifications.
5. The dashboard displays the stock trend, indicators, and model outputs.
6. AI-assisted tools help users interpret the results in natural language.

## Machine-Learning Methodology

The application uses historical daily price data and features such as moving-average ratio, momentum, volatility, RSI, and daily returns.

The models are evaluated using chronological training and testing data to better respect the time order of market observations. Performance is compared with a majority-class baseline.

Model performance may be limited, and predictions can be incorrect. The application presents experimental results rather than guaranteed future outcomes.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/avaniverma612-pixel/ai-financial-analyst.git
cd ai-financial-analyst
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

If using the AI features, configure your Groq API key through Streamlit secrets.

Create `.streamlit/secrets.toml` and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Never commit your real API key or other secrets to GitHub.

### 5. Run the application

```bash
streamlit run web_app.py
```

Open the local URL shown in the terminal.

## Limitations

- Stock-market movements are uncertain and influenced by many factors.
- Machine-learning predictions are experimental and may perform below simple baselines.
- Historical patterns do not guarantee future performance.
- Global Market Pulse displays selected market indices, not every stock or trade.
- AI-generated explanations may contain errors and should be checked against the displayed data.

## Disclaimer

This project is intended for educational and informational purposes only. It does not provide guaranteed investment predictions or personalised financial advice. Always conduct independent research before making investment decisions.

## Author

**Avani Verma**

GitHub: [avaniverma612-pixel](https://github.com/avaniverma612-pixel)
