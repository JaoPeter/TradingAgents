from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4-mini"  # Use a different model
config["quick_think_llm"] = "gpt-5.4-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (stocks use yfinance by default; get_stock_data prefers crypto-native feeds first)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance, binance, coinmarketcap
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance, coinmarketcap, defilama
    \"news_data\": \"yfinance\",                 # Options: alpha_vantage, yfinance, cryptocompare, crypto_rss, x_sentiment
}
config["tool_vendors"] = {
    "get_stock_data": "binance,coinmarketcap,yfinance,alpha_vantage",
    "get_fundamentals": "coinmarketcap,defilama,yfinance,alpha_vantage",
    "get_news": "cryptocompare,x_sentiment,crypto_rss,yfinance,alpha_vantage",
    "get_global_news": "cryptocompare,crypto_rss,yfinance,alpha_vantage",
    "get_sentiment_summary": "x_sentiment",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
