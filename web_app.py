import streamlit as st
import yfinance as yf
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai import LLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

.stApp {
    background-color: #0b1020;
}

.block-container {
    max-width: 1200px;
    padding-top: 45px;
    padding-bottom: 50px;
}


/* HEADER */

.main-title {
    font-size: 42px;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 6px;
}

.subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 30px;
}


/* COMPANY CARD */

.company-card {
    background: #151d33;
    border: 1px solid #293756;
    border-radius: 18px;
    padding: 28px;
    margin-top: 25px;
    margin-bottom: 20px;
}

.company-name {
    font-size: 30px;
    font-weight: 700;
    color: #f8fafc;
}

.company-ticker {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 7px;
}


/* METRICS */

.metric-card {
    background: #151d33;
    border: 1px solid #293756;
    border-radius: 15px;
    padding: 22px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 10px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 700;
}


/* CHART */

.chart-card {
    background: #151d33;
    border: 1px solid #293756;
    border-radius: 18px;
    padding: 22px;
    margin-top: 25px;
    margin-bottom: 25px;
}

.chart-title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
}

.chart-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 15px;
}


/* RECOMMENDATION */

.recommendation-card {
    border-radius: 18px;
    padding: 28px;
    margin-top: 25px;
    margin-bottom: 30px;
    text-align: center;
}

.recommendation-label {
    font-size: 14px;
    font-weight: 600;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.recommendation-value {
    font-size: 46px;
    font-weight: 800;
    margin: 8px 0;
}

.recommendation-text {
    font-size: 17px;
    color: #e2e8f0;
    max-width: 800px;
    margin: auto;
}


/* BUY */

.buy {
    background: #123524;
    border: 1px solid #287a50;
}

.buy .recommendation-value {
    color: #4ade80;
}


/* HOLD */

.hold {
    background: #352b12;
    border: 1px solid #8a6b1d;
}

.hold .recommendation-value {
    color: #facc15;
}


/* SELL */

.sell {
    background: #3a171b;
    border: 1px solid #8a3038;
}

.sell .recommendation-value {
    color: #f87171;
}


/* SECTION */

.section-title {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 18px;
}


/* INFO / METHODOLOGY */

.info-strip {
    background: #11182b;
    border: 1px solid #293756;
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 18px;
    margin-bottom: 24px;
}

.info-label {
    color: #64748b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 5px;
}

.info-value {
    color: #e2e8f0;
    font-size: 14px;
}

.signal-card {
    background: #151d33;
    border: 1px solid #293756;
    border-radius: 18px;
    padding: 24px;
    margin-top: 28px;
    margin-bottom: 24px;
}

.signal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 20px;
}

.signal-title {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
}

.signal-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 5px;
}

.signal-badge {
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    white-space: nowrap;
}

.positive-signal {
    color: #4ade80;
    background: #123524;
    border: 1px solid #287a50;
}

.negative-signal {
    color: #f87171;
    background: #3a171b;
    border: 1px solid #8a3038;
}

.neutral-signal {
    color: #facc15;
    background: #352b12;
    border: 1px solid #8a6b1d;
}

.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.signal-item {
    background: #11182b;
    border: 1px solid #293756;
    border-radius: 13px;
    padding: 16px;
}

.signal-item-label {
    color: #64748b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.signal-item-value {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 700;
    margin-top: 7px;
}

.signal-item-note {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 5px;
}

@media (max-width: 800px) {
    .signal-grid {
        grid-template-columns: 1fr;
    }

    .signal-header {
        flex-direction: column;
    }
}

.methodology-card {
    background: #11182b;
    border: 1px solid #293756;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 24px;
    margin-bottom: 28px;
}

.methodology-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
}

.methodology-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.65;
}

.methodology-step {
    color: #cbd5e1;
    margin: 7px 0;
}

/* TECHNICAL INDICATORS */

.technical-card {
    background: #151d33;
    border: 1px solid #293756;
    border-radius: 18px;
    padding: 24px;
    margin-top: 28px;
    margin-bottom: 28px;
}

.technical-title {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 5px;
}

.technical-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 18px;
}

.technical-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.technical-item {
    background: #11182b;
    border: 1px solid #293756;
    border-radius: 13px;
    padding: 17px;
}

.technical-label {
    color: #64748b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.technical-value {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-top: 7px;
}

.technical-note {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 5px;
}

@media (max-width: 800px) {
    .technical-grid {
        grid-template-columns: 1fr;
    }
}

/* REPORT EXPORT */

.report-card {
    background: #11182b;
    border: 1px solid #293756;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 28px;
    margin-bottom: 24px;
}

.report-title {
    color: #f8fafc;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 6px;
}

.report-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.55;
    margin-bottom: 16px;
}

.stack-strip {
    color: #64748b;
    font-size: 12px;
    text-align: center;
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid #1f2a43;
}

/* DISCLAIMER */

.disclaimer {
    color: #64748b;
    font-size: 12px;
    margin-top: 40px;
    text-align: center;
}

</style>
""")


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="main-title">📈 AI Financial Analyst</div>

<div class="subtitle">
Enter a company name or stock ticker to receive an AI-powered financial analysis.
</div>
""")


# ============================================================
# FIND TICKER
# ============================================================

def find_ticker(company_input):

    company_input = company_input.strip()

    if not company_input:
        return None

    direct_ticker = company_input.upper()

    try:
        stock = yf.Ticker(direct_ticker)
        history = stock.history(period="5d")

        if not history.empty:
            return direct_ticker

    except Exception:
        pass

    try:

        search = yf.Search(
            company_input,
            max_results=10
        )

        quotes = search.quotes

        if quotes:

            for quote in quotes:

                symbol = quote.get("symbol")

                if symbol:
                    return symbol

    except Exception:
        pass

    return None


# ============================================================
# GET STOCK DATA
# ============================================================

def get_stock_data(ticker):

    stock = yf.Ticker(ticker)

    try:

        history = stock.history(
            period="1mo",
            auto_adjust=True
        )

    except Exception:

        return None

    if history.empty:
        return None

    history = history.dropna(subset=["Close"])

    if len(history) < 5:
        return None

    # 5-DAY MOVING AVERAGE

    history["MA5"] = history["Close"].rolling(5).mean()

    latest_price = float(history["Close"].iloc[-1])
    ma5 = float(history["MA5"].iloc[-1])

    if (
        latest_price != latest_price
        or ma5 != ma5
        or ma5 == 0
    ):
        return None

    difference = latest_price - ma5

    percentage = (difference / ma5) * 100

    # ADDITIONAL TECHNICAL INDICATORS
    recent_close = history["Close"].tail(5)

    five_day_momentum = (
        (float(recent_close.iloc[-1]) / float(recent_close.iloc[0])) - 1
    ) * 100

    daily_returns = history["Close"].pct_change().dropna().tail(5)
    five_day_volatility = float(daily_returns.std() * 100) if len(daily_returns) >= 2 else 0.0

    delta = history["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean().iloc[-1]
    avg_loss = losses.rolling(14).mean().iloc[-1]

    if avg_loss == 0:
        rsi14 = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi14 = 100 - (100 / (1 + rs))


    # SHORT-TERM TREND

    if percentage >= 1:

        trend = "Bullish"

    elif percentage <= -1:

        trend = "Bearish"

    else:

        trend = "Neutral"


    # RECOMMENDATION

    if percentage >= 2:

        recommendation = "BUY"

    elif percentage <= -2:

        recommendation = "SELL"

    else:

        recommendation = "HOLD"


    # COMPANY INFORMATION

    try:

        info = stock.info

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )

        exchange = info.get("exchange", "")

        currency = info.get("currency", "USD")

    except Exception:

        company_name = ticker
        exchange = ""
        currency = "USD"


    # KEEP ONLY THE DATA NEEDED FOR THE CHART

    chart_data = history[["Close", "MA5"]].copy()

    chart_data.columns = [
        "Stock Price",
        "5-Day Moving Average"
    ]


    return {

        "ticker": ticker,

        "company_name": company_name,

        "exchange": exchange,

        "currency": currency,

        "price": latest_price,

        "ma5": ma5,

        "difference": difference,

        "percentage": percentage,

        "five_day_momentum": five_day_momentum,

        "five_day_volatility": five_day_volatility,

        "rsi14": float(rsi14),

        "trend": trend,

        "recommendation": recommendation,

        "chart_data": chart_data
    }


# ============================================================
# AI ANALYSIS
# ============================================================

def generate_analysis(data):

    local_llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.22
    )


    financial_analyst = Agent(

        role="Senior Financial Analyst",

        goal=(
            "Produce a clear, concise and professional "
            "financial analysis from verified market data."
        ),

        backstory="""

You are a professional financial analyst.

Your job is to interpret financial and market information
and explain it clearly to an ordinary investor.

Do NOT simply repeat raw numbers.

Explain what the numbers mean.

Your analysis must contain:

### Executive Summary

Explain the overall situation in simple language.

### Technical View

Explain the current price compared with the
5-day moving average.

### Strengths

Give 2 or 3 relevant strengths.

### Risks

Give 2 or 3 relevant risks.

### Final Assessment

Give a clear recommendation:

BUY
HOLD
or
SELL

The recommendation must agree with the system
recommendation provided in the market data.

Do not invent financial figures.

Do not claim that a stock is guaranteed to rise or fall.

Keep everything understandable to an ordinary investor.

""",

        llm=local_llm,

        verbose=False
    )


    task = Task(

        description=f"""

Analyze the following verified market information.

Company:
{data['company_name']}

Ticker:
{data['ticker']}

Current Price:
{data['price']:.2f} {data['currency']}

5-Day Moving Average:
{data['ma5']:.2f} {data['currency']}

Difference:
{data['difference']:.2f}

Percentage Difference:
{data['percentage']:.2f}%

5-Day Price Momentum:
{data['five_day_momentum']:.2f}%

5-Day Volatility:
{data['five_day_volatility']:.2f}%

14-Day RSI:
{data['rsi14']:.1f}

Short-Term Trend:
{data['trend']}

System Recommendation:
{data['recommendation']}


Write a readable financial analysis.

Use these sections:

### Executive Summary

Explain what is happening and what it means.

### Technical View

Explain the relationship between the current
price and the 5-day moving average. Also interpret
the provided 5-day momentum, 5-day volatility,
and 14-day RSI where useful. Do not invent any
additional indicators or financial figures.

### Strengths

Give 2 or 3 concise points.

### Risks

Give 2 or 3 concise points.

### Final Assessment

Start with:

Recommendation: {data['recommendation']}

Then explain clearly why.

Do not invent numbers.

Do not simply repeat the data.

Make the conclusion obvious to the reader.

""",

        expected_output=(
            "A clear and professional financial analysis "
            "with an explicit BUY, HOLD or SELL conclusion."
        ),

        agent=financial_analyst
    )


    crew = Crew(
        agents=[financial_analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )


    result = crew.kickoff()

    return str(result)


# ============================================================
# USER INPUT
# ============================================================

company_input = st.text_input(
    "Company name or stock ticker",
    placeholder="e.g. Apple, NVIDIA, Tesla, Microsoft, AAPL"
)


# ============================================================
# BUTTON
# ============================================================

if st.button("Analyze Company", type="primary"):

    if not company_input.strip():

        st.warning(
            "Please enter a company name or stock ticker."
        )

        st.stop()


    # FIND COMPANY

    with st.spinner("Finding company..."):

        ticker = find_ticker(company_input)


    if ticker is None:

        st.error(
            "I couldn't find that company. "
            "Try entering its stock ticker instead."
        )

        st.stop()


    # GET DATA

    with st.spinner("Getting market data..."):

        stock_data = get_stock_data(ticker)


    if stock_data is None:

        st.error(
            "No recent market data could be retrieved "
            "for this company."
        )

        st.stop()


    # ========================================================
    # COMPANY HEADER
    # ========================================================

    exchange_text = ""

    if stock_data["exchange"]:

        exchange_text = " · " + stock_data["exchange"]


    st.html(f"""

    <div class="company-card">

        <div class="company-name">
            {stock_data['company_name']}
        </div>

        <div class="company-ticker">
            {stock_data['ticker']}{exchange_text}
        </div>

    </div>

    """)


    # ========================================================
    # DATA SOURCE / UPDATE INFO
    # ========================================================

    latest_market_date = None
    try:
        latest_market_date = (
            yf.Ticker(stock_data["ticker"])
            .history(period="5d", auto_adjust=True)
            .dropna(subset=["Close"])
            .index[-1]
        )
        latest_market_date = latest_market_date.strftime("%b %d, %Y")
    except Exception:
        latest_market_date = "Latest available session"

    analysis_time = datetime.now().astimezone().strftime("%b %d, %Y · %I:%M %p")

    st.html(f"""
    <div class="info-strip">
        <div style="display:flex; gap:28px; flex-wrap:wrap;">
            <div>
                <div class="info-label">Market data source</div>
                <div class="info-value">Yahoo Finance</div>
            </div>
            <div>
                <div class="info-label">Market data through</div>
                <div class="info-value">{latest_market_date}</div>
            </div>
            <div>
                <div class="info-label">Analysis generated</div>
                <div class="info-value">{analysis_time}</div>
            </div>
        </div>
    </div>
    """)


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.html(f"""

        <div class="metric-card">

            <div class="metric-label">
                Current Price
            </div>

            <div class="metric-value">
                {stock_data['price']:.2f}
                {stock_data['currency']}
            </div>

        </div>

        """)


    with col2:

        st.html(f"""

        <div class="metric-card">

            <div class="metric-label">
                5-Day Average
            </div>

            <div class="metric-value">
                {stock_data['ma5']:.2f}
                {stock_data['currency']}
            </div>

        </div>

        """)


    with col3:

        percentage = stock_data["percentage"]

        if percentage >= 0:

            percentage_display = f"+{percentage:.2f}%"

        else:

            percentage_display = f"{percentage:.2f}%"


        st.html(f"""

        <div class="metric-card">

            <div class="metric-label">
                vs. 5-Day Average
            </div>

            <div class="metric-value">
                {percentage_display}
            </div>

        </div>

        """)


    with col4:

        st.html(f"""

        <div class="metric-card">

            <div class="metric-label">
                Short-Term Trend
            </div>

            <div class="metric-value">
                {stock_data['trend']}
            </div>

        </div>

        """)


    # ========================================================
    # 5-DAY STOCK PRICE CHART
    # ========================================================
    st.html("""
    <div class="chart-card">
        <div class="chart-title">📊 5-Day Stock Price Trend</div>
        <div class="chart-subtitle">
            Closing price compared with the 5-day moving average
        </div>
    </div>
    """)

    # Use the most recent 5 trading sessions from the data we already fetched.
    chart_history = yf.Ticker(stock_data["ticker"]).history(
        period="5d",
        auto_adjust=True
    )

    if chart_history is None or chart_history.empty:
        st.warning("The 5-day chart data could not be retrieved right now.")
    else:
        chart_history = chart_history.dropna(subset=["Close"]).tail(5).copy()

        if len(chart_history) < 2:
            st.warning("Not enough recent trading data to display the chart.")
        else:
            # The 5-day average is calculated from the same five displayed sessions.
            five_day_average = float(chart_history["Close"].mean())
            chart_history["5-Day Moving Average"] = five_day_average

            chart_display = chart_history[["Close", "5-Day Moving Average"]].copy()
            chart_display.columns = ["Stock Price", "5-Day Moving Average"]
            chart_display.index = chart_display.index.strftime("%b %d")
            chart_display.index.name = "Trading Day"

            # Altair gives us a cleaner, more presentation-ready chart than st.line_chart.
            import altair as alt

            chart_long = (
                chart_display
                .reset_index()
                .melt(
                    id_vars="Trading Day",
                    var_name="Metric",
                    value_name="Price"
                )
            )

            price_line = (
                alt.Chart(chart_long[chart_long["Metric"] == "Stock Price"])
                .mark_line(point=alt.OverlayMarkDef(size=75), strokeWidth=3)
                .encode(
                    x=alt.X(
                        "Trading Day:N",
                        sort=list(chart_display.index),
                        title="Trading Day",
                        axis=alt.Axis(labelAngle=0)
                    ),
                    y=alt.Y(
                        "Price:Q",
                        title=f"Price ({stock_data['currency']})",
                        scale=alt.Scale(zero=False)
                    ),
                    tooltip=[
                        alt.Tooltip("Trading Day:N", title="Day"),
                        alt.Tooltip("Price:Q", title="Stock Price", format=".2f")
                    ]
                )
            )

            average_line = (
                alt.Chart(chart_long[chart_long["Metric"] == "5-Day Moving Average"])
                .mark_line(strokeWidth=2, strokeDash=[7, 5])
                .encode(
                    x=alt.X(
                        "Trading Day:N",
                        sort=list(chart_display.index),
                        title="Trading Day"
                    ),
                    y=alt.Y(
                        "Price:Q",
                        title=f"Price ({stock_data['currency']})",
                        scale=alt.Scale(zero=False)
                    ),
                    tooltip=[
                        alt.Tooltip("Trading Day:N", title="Day"),
                        alt.Tooltip("Price:Q", title="5-Day Average", format=".2f")
                    ]
                )
            )

            st.altair_chart(
                (price_line + average_line)
                .properties(height=390)
                .configure_view(strokeOpacity=0)
                .configure_axis(
                    labelColor="#cbd5e1",
                    titleColor="#94a3b8",
                    gridColor="#293756",
                    domainColor="#293756"
                )
                .configure_legend(
                    title=None,
                    labelColor="#e2e8f0",
                    orient="bottom"
                ),
                use_container_width=True
            )

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.metric(
                    "5-Day Average",
                    f"{five_day_average:.2f} {stock_data['currency']}"
                )
            with chart_col2:
                change = (
                    (float(chart_history["Close"].iloc[-1]) /
                     float(chart_history["Close"].iloc[0]) - 1) * 100
                )
                st.metric(
                    "5-Day Price Change",
                    f"{change:+.2f}%"
                )

    # ========================================================
    # SIGNAL BREAKDOWN
    # ========================================================

    if stock_data["percentage"] > 0:
        position_text = "above"
        position_class = "positive-signal"
    elif stock_data["percentage"] < 0:
        position_text = "below"
        position_class = "negative-signal"
    else:
        position_text = "at"
        position_class = "neutral-signal"

    if stock_data["trend"] == "Bullish":
        trend_class = "positive-signal"
        trend_explanation = "Price is meaningfully above the 5-day average."
    elif stock_data["trend"] == "Bearish":
        trend_class = "negative-signal"
        trend_explanation = "Price is meaningfully below the 5-day average."
    else:
        trend_class = "neutral-signal"
        trend_explanation = "The difference from the 5-day average is relatively small."

    st.html(f"""
    <div class="signal-card">
        <div class="signal-header">
            <div>
                <div class="signal-title">What is driving the signal?</div>
                <div class="signal-subtitle">The recommendation is based on the verified short-term indicators below.</div>
            </div>
            <div class="signal-badge {trend_class}">{stock_data['trend']}</div>
        </div>

        <div class="signal-grid">
            <div class="signal-item">
                <div class="signal-item-label">Price position</div>
                <div class="signal-item-value">{position_text} 5-day average</div>
                <div class="signal-item-note">{stock_data['percentage']:+.2f}% vs. moving average</div>
            </div>

            <div class="signal-item">
                <div class="signal-item-label">Trend classification</div>
                <div class="signal-item-value">{stock_data['trend']}</div>
                <div class="signal-item-note">{trend_explanation}</div>
            </div>

            <div class="signal-item">
                <div class="signal-item-label">System decision</div>
                <div class="signal-item-value">{stock_data['recommendation']}</div>
                <div class="signal-item-note">Rule-based short-term signal</div>
            </div>
        </div>
    </div>
    """)


    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    rsi = stock_data["rsi14"]

    if rsi >= 70:
        rsi_note = "Above 70, indicating an overbought reading."
    elif rsi <= 30:
        rsi_note = "Below 30, indicating an oversold reading."
    else:
        rsi_note = "Within the commonly used neutral range."

    momentum = stock_data["five_day_momentum"]

    if momentum > 0:
        momentum_note = "The stock gained over the five displayed trading sessions."
    elif momentum < 0:
        momentum_note = "The stock declined over the five displayed trading sessions."
    else:
        momentum_note = "The stock was essentially unchanged over the five displayed trading sessions."

    volatility = stock_data["five_day_volatility"]

    st.html(f"""
    <div class="technical-card">
        <div class="technical-title">Technical Indicators</div>
        <div class="technical-subtitle">Additional quantitative context used by the AI analyst.</div>

        <div class="technical-grid">
            <div class="technical-item">
                <div class="technical-label">5-Day Momentum</div>
                <div class="technical-value">{momentum:+.2f}%</div>
                <div class="technical-note">{momentum_note}</div>
            </div>

            <div class="technical-item">
                <div class="technical-label">14-Day RSI</div>
                <div class="technical-value">{rsi:.1f}</div>
                <div class="technical-note">{rsi_note}</div>
            </div>

            <div class="technical-item">
                <div class="technical-label">5-Day Volatility</div>
                <div class="technical-value">{volatility:.2f}%</div>
                <div class="technical-note">Standard deviation of recent daily returns.</div>
            </div>
        </div>
    </div>
    """)


    # ========================================================
    # ANALYSIS METHODOLOGY
    # ========================================================

    st.html("""
    <div class="methodology-card">
        <div class="methodology-title">How the analysis works</div>
        <div class="methodology-text">
            The application separates <strong>data retrieval</strong> from <strong>AI interpretation</strong>.
            Verified market data is retrieved first, quantitative indicators are calculated, and only then is the
            resulting information passed to the AI financial analyst.
            <div class="methodology-step"><strong>1.</strong> Retrieve recent market data for the selected company.</div>
            <div class="methodology-step"><strong>2.</strong> Calculate the 5-day moving average and price difference.</div>
            <div class="methodology-step"><strong>3.</strong> Classify the short-term signal as Bullish, Neutral, or Bearish.</div>
            <div class="methodology-step"><strong>4.</strong> Generate a BUY, HOLD, or SELL system recommendation.</div>
            <div class="methodology-step"><strong>5.</strong> Pass the verified indicators to the AI agent for explanation and context.</div>
        </div>
    </div>
    """)


    # ========================================================
    # RECOMMENDATION
    # ========================================================
    recommendation = stock_data["recommendation"]

    if recommendation == "BUY":
        recommendation_class = "buy"
        recommendation_text = (
            "The available short-term indicators are positive. "
            "The system detects enough upward momentum to produce a BUY signal."
        )
    elif recommendation == "SELL":
        recommendation_class = "sell"
        recommendation_text = (
            "The available short-term indicators are negative. "
            "The system detects enough downward pressure to produce a SELL signal."
        )
    else:
        recommendation_class = "hold"
        recommendation_text = (
            "The available short-term indicators are mixed or too weak to support "
            "a strong directional signal. The system therefore recommends HOLD."
        )

    st.html(f"""
    <div class="recommendation-card {recommendation_class}">
        <div class="recommendation-label">AI Investment Recommendation</div>
        <div class="recommendation-value">{recommendation}</div>
        <div class="recommendation-text">{recommendation_text}</div>
    </div>
    """)

    # ========================================================
    # AI ANALYSIS
    # ========================================================
    st.html("""
    <div class="section-title">AI Financial Analysis</div>
    """)

    with st.spinner("AI analyst is preparing the financial analysis..."):
        try:
            analysis = generate_analysis(stock_data)
            st.markdown(analysis)

            # ====================================================
            # DOWNLOADABLE FINANCIAL BRIEF
            # ====================================================

            recommendation = stock_data["recommendation"]
            price = stock_data["price"]
            ma5 = stock_data["ma5"]
            percentage = stock_data["percentage"]
            momentum = stock_data["five_day_momentum"]
            rsi = stock_data["rsi14"]
            volatility = stock_data["five_day_volatility"]
            trend = stock_data["trend"]
            currency = stock_data["currency"]

            if percentage > 0:
                price_position = (
                    f"The stock is trading {percentage:.2f}% above its "
                    f"5-day moving average, which indicates slightly stronger "
                    f"short-term price positioning."
                )
            elif percentage < 0:
                price_position = (
                    f"The stock is trading {abs(percentage):.2f}% below its "
                    f"5-day moving average, which indicates slightly weaker "
                    f"short-term price positioning."
                )
            else:
                price_position = (
                    "The stock is trading almost exactly at its 5-day "
                    "moving average, so there is little short-term separation."
                )

            if momentum > 0:
                momentum_text = (
                    f"Over the five-day period, the stock gained {momentum:.2f}%. "
                    "That points to positive recent momentum, although the move "
                    "should be considered alongside the other indicators."
                )
            elif momentum < 0:
                momentum_text = (
                    f"Over the five-day period, the stock declined {abs(momentum):.2f}%. "
                    "That points to weaker recent momentum, although the move "
                    "should be considered alongside the other indicators."
                )
            else:
                momentum_text = (
                    "The stock was essentially unchanged over the five-day period, "
                    "suggesting limited recent directional momentum."
                )

            if rsi >= 70:
                rsi_text = (
                    f"The 14-day RSI is {rsi:.1f}, which is in a relatively "
                    "high range and may indicate stronger buying momentum."
                )
            elif rsi <= 30:
                rsi_text = (
                    f"The 14-day RSI is {rsi:.1f}, which is in a relatively "
                    "low range and may indicate stronger selling pressure."
                )
            else:
                rsi_text = (
                    f"The 14-day RSI is {rsi:.1f}, which sits in a broadly "
                    "neutral range rather than indicating an extreme condition."
                )

            volatility_text = (
                f"Recent five-day volatility is {volatility:.2f}%, giving an "
                "indication of how much the stock's daily returns have been moving."
            )

            if recommendation == "BUY":
                decision_text = (
                    "The system identifies a positive short-term setup. "
                    "The available indicators provide enough upward evidence "
                    "for the model's BUY threshold to be reached."
                )
            elif recommendation == "SELL":
                decision_text = (
                    "The system identifies a negative short-term setup. "
                    "The available indicators provide enough downward evidence "
                    "for the model's SELL threshold to be reached."
                )
            else:
                decision_text = (
                    "The system does not detect a strong enough short-term signal "
                    "to justify a BUY or SELL. The indicators are therefore treated "
                    "as mixed or relatively neutral, resulting in a HOLD."
                )

            generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

            report_lines = [
                "AI FINANCIAL ANALYST",
                f"{stock_data['company_name']} ({stock_data['ticker']})",
                "=" * 64,
                "",
                f"Investment Assessment: {recommendation}",
                "",
                "AT A GLANCE",
                "-" * 64,
                decision_text,
                "",
                "MARKET SNAPSHOT",
                "-" * 64,
                f"Current price       : {price:.2f} {currency}",
                f"5-day moving avg.   : {ma5:.2f} {currency}",
                f"Price vs. average   : {percentage:+.2f}%",
                f"5-day momentum      : {momentum:+.2f}%",
                f"14-day RSI          : {rsi:.1f}",
                f"5-day volatility    : {volatility:.2f}%",
                f"Short-term trend    : {trend}",
                "",
                "WHAT THE INDICATORS SUGGEST",
                "-" * 64,
                "",
                "Price position",
                price_position,
                "",
                "Momentum",
                momentum_text,
                "",
                "RSI",
                rsi_text,
                "",
                "Volatility",
                volatility_text,
                "",
                "AI ANALYST ASSESSMENT",
                "-" * 64,
                "",
                analysis,
                "",
                "RECENT PRICE HISTORY",
                "-" * 64,
            ]

            for idx, row in stock_data["chart_data"].tail(5).iterrows():
                date_label = (
                    idx.strftime("%b %d, %Y")
                    if hasattr(idx, "strftime")
                    else str(idx)
                )
                report_lines.append(
                    f"{date_label:<16} {float(row['Stock Price']):.2f} {currency}"
                )

            report_lines.extend([
                "",
                "METHODOLOGY",
                "-" * 64,
                "Market data is retrieved through Yahoo Finance. The system",
                "calculates short-term technical indicators and applies a",
                "rule-based BUY/HOLD/SELL signal. The verified indicators are",
                "then provided to the AI financial analyst for interpretation.",
                "",
                f"Generated: {generated_at}",
                "Data source: Yahoo Finance",
                "",
                "DISCLAIMER",
                "-" * 64,
                "This report is intended for educational purposes only.",
                "It is not financial advice and should not be used as the",
                "sole basis for investment decisions.",
            ])

            report_text = "\n".join(report_lines)

            st.html("""
            <div class="report-card">
                <div class="report-title">Download financial brief</div>
                <div class="report-text">
                    Save a clean, shareable summary of the market indicators,
                    AI assessment, methodology and recent price history.
                </div>
            </div>
            """)

            st.download_button(
                label="⬇ Download Financial Brief",
                data=report_text,
                file_name=f"{stock_data['ticker']}_financial_brief.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(
                "The market data was retrieved successfully, "
                "but the AI analyst could not complete the analysis."
            )
            st.caption(str(e))

    # ========================================================
    # PROJECT FOOTER
    # ========================================================
    st.html("""
    <div class="stack-strip">
        Built with Python · Streamlit · yFinance · CrewAI · Ollama
    </div>
    """)

    # ========================================================
    # DISCLAIMER
    # ========================================================
    st.html("""
    <div class="disclaimer">
        This project provides a short-term, data-based analysis for educational
        purposes. It is not financial advice and should not be used as the sole
        basis for investment decisions.
    </div>
    """)
