import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# 1. Connect CrewAI to your local Ollama AI
local_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2
)


# 2. Create the stock-data tool
@tool("fetch_stock_data")
def fetch_stock_data(ticker: str) -> str:
    """Fetch the last month of stock data and calculate a 5-day moving average."""
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")

        if hist.empty:
            return f"No stock data found for {ticker}."

        # Calculate 5-day moving average
        hist["MA_5"] = hist["Close"].rolling(window=5).mean()

        latest_close = hist["Close"].iloc[-1]
        ma_5_latest = hist["MA_5"].iloc[-1]

        trend = "UPWARD" if latest_close > ma_5_latest else "DOWNWARD"

        return (
            f"Ticker: {ticker}\n"
            f"Latest Closing Price: ${latest_close:.2f}\n"
            f"5-Day Moving Average: ${ma_5_latest:.2f}\n"
            f"Calculated Short-term Trend: {trend}"
        )

    except Exception as e:
        return f"Error fetching data: {str(e)}"


# 3. Create the autonomous financial analyst
financial_analyst_agent = Agent(
    role="Senior Financial Analyst",
    goal="Analyze live market metrics and provide a clear stock trend analysis.",
    backstory=(
        "You are an expert financial analyst who specializes in "
        "technical analysis and turning financial data into clear "
        "investment insights."
    ),
    verbose=True,
    llm=local_llm,
    tools=[fetch_stock_data]
)


# 4. Give the agent its task
analysis_task = Task(
    description=(
        "Analyze the stock performance of NVIDIA (NVDA). "
        "Use the fetch_stock_data tool to obtain the latest real "
        "stock data. Explain the current price, compare it with "
        "the 5-day moving average, identify whether the short-term "
        "trend is upward or downward, and provide a final "
        "Buy, Hold, or Sell recommendation."
    ),
    expected_output=(
        "A clear 3-paragraph investment memo containing the latest "
        "stock price, 5-day moving average, calculated trend, "
        "reasoning, and a final Buy/Hold/Sell recommendation."
    ),
    agent=financial_analyst_agent,
    output_file="investment_memo.txt"
)


# 5. Assemble the autonomous agent system
financial_crew = Crew(
    agents=[financial_analyst_agent],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)


# 6. Start the system
if __name__ == "__main__":
    print("Starting the Agentic AI Financial Analyst system...")

    financial_crew.kickoff()

    print("Project complete! Check your folder for 'investment_memo.txt'")