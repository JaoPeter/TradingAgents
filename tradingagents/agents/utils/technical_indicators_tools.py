from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


_INDICATOR_ALIAS_MAP = {
    # Common shorthand aliases that models frequently emit
    "ema": "close_10_ema",
    "sma": "close_50_sma",
    "50sma": "close_50_sma",
    "200sma": "close_200_sma",
    "ema10": "close_10_ema",
    "rsi14": "rsi",
    "macd_signal": "macds",
    "macd_hist": "macdh",
    "bb_mid": "boll",
    "bb_upper": "boll_ub",
    "bb_lower": "boll_lb",
}


def _normalize_indicator_name(indicator: str) -> str:
    key = (indicator or "").strip().lower().replace("-", "_")
    key = key.replace(" ", "_")
    return _INDICATOR_ALIAS_MAP.get(key, key)

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [_normalize_indicator_name(i) for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except (ValueError, RuntimeError) as e:
            results.append(str(e))

    if not results:
        return "No technical indicator result available."
    return "\n\n".join(results)