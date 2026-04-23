import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": "low",  # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Output token limits per LLM role
    "quick_max_tokens": 700,
    "deep_max_tokens": 1100,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Timeout settings (seconds) — override via env vars API_TIMEOUT_SECONDS / LLM_TIMEOUT_SECONDS
    "api_timeout": int(os.getenv("API_TIMEOUT_SECONDS", "30")),
    "llm_timeout": int(os.getenv("LLM_TIMEOUT_SECONDS", "240")),
    # Trading style / timeframe defaults (overridden by CLI selections)
    "trading_style": "swing",
    "primary_tf": "1d",
    "confirm_tf": "",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "binance",         # Options: binance, coinmarketcap, yfinance, alpha_vantage
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "defilama",      # Options: defilama, coinmarketcap, yfinance, alpha_vantage
        "news_data": "cryptocompare",        # Options: cryptocompare, x_sentiment, crypto_rss, yfinance, alpha_vantage
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        "get_stock_data": "binance,coinmarketcap,yfinance,alpha_vantage",
        "get_fundamentals": "defilama,coinmarketcap,yfinance,alpha_vantage",
        "get_news": "cryptocompare,x_sentiment,crypto_rss,yfinance,alpha_vantage",
        "get_global_news": "cryptocompare,crypto_rss,yfinance,alpha_vantage",
        "get_sentiment_summary": "x_sentiment",
    },
}
