from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .binance import get_stock_data as get_binance_stock
from .coinmarketcap import (
    get_fundamentals as get_coinmarketcap_fundamentals,
    get_stock_data as get_coinmarketcap_stock,
)
from .cryptocompare import (
    get_news as get_cryptocompare_news,
    get_global_news as get_cryptocompare_global_news,
)
from .defilama import get_fundamentals as get_defilama_fundamentals
from .x_sentiment import (
    get_sentiment_summary as get_x_sentiment,
    get_news as get_x_news,
)
from .crypto_rss import (
    get_news as get_crypto_rss_news,
    get_global_news as get_crypto_rss_global_news,
)
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError

# Configuration and routing logic
from .config import get_config


def _is_vendor_failure(result) -> bool:
    """Detect common vendor failure payloads so fallback vendors can run."""
    if not isinstance(result, str):
        return False

    normalized = result.strip().lower()
    failure_prefixes = (
        "error",
        "no data",
        "invalid",
    )

    return normalized.startswith(failure_prefixes) or "api_key not set" in normalized

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_sentiment_summary",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "binance",
    "coinmarketcap",
    "cryptocompare",
    "crypto_rss",
    "defilama",
    "x_sentiment",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "binance": get_binance_stock,
        "coinmarketcap": get_coinmarketcap_stock,
        "yfinance": get_YFin_data_online,
        "alpha_vantage": get_alpha_vantage_stock,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "coinmarketcap": get_coinmarketcap_fundamentals,
        "defilama": get_defilama_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "alpha_vantage": get_alpha_vantage_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "cryptocompare": get_cryptocompare_news,
        "x_sentiment": get_x_news,
        "crypto_rss": get_crypto_rss_news,
        "yfinance": get_news_yfinance,
        "alpha_vantage": get_alpha_vantage_news,
    },
    "get_global_news": {
        "cryptocompare": get_cryptocompare_global_news,
        "crypto_rss": get_crypto_rss_global_news,
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_sentiment_summary": {
        "x_sentiment": get_x_sentiment,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    last_error = None

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            result = impl_func(*args, **kwargs)
            if _is_vendor_failure(result):
                last_error = f"{vendor}: {result}"
                continue
            return result
        except AlphaVantageRateLimitError:
            last_error = f"{vendor}: rate limited"
            continue  # Only rate limits trigger fallback
        except Exception as exc:
            last_error = f"{vendor}: {exc}"
            continue

    if last_error:
        raise RuntimeError(f"No available vendor for '{method}'. Last failure: {last_error}")

    raise RuntimeError(f"No available vendor for '{method}'")