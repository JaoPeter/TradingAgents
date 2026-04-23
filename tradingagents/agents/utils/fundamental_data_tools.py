from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.defilama import get_fundamentals as get_defilama_fundamentals
from tradingagents.dataflows.coinmarketcap import get_fundamentals as get_coinmarketcap_fundamentals


_CRYPTO_SUFFIXES = ("-USD", "-USDT", "-BTC", "-ETH", "-EUR", "-GBP")
_CRYPTO_BASE_TICKERS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "LINK", "UNI", "AAVE", "CRV", "MKR"
}


def _is_crypto_ticker(ticker: str) -> bool:
    upper = (ticker or "").upper().strip()
    if not upper:
        return False
    if upper.endswith(_CRYPTO_SUFFIXES):
        return True
    return upper in _CRYPTO_BASE_TICKERS


def _is_failure_payload(text: str) -> bool:
    if not isinstance(text, str):
        return True
    normalized = text.strip().lower()
    return (
        normalized.startswith("error")
        or normalized.startswith("no data")
        or normalized.startswith("invalid")
        or "api_key not set" in normalized
        or "not set in .env" in normalized
    )


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    if _is_crypto_ticker(ticker):
        # For crypto, combine DeFi/on-chain context (DefiLlama) with market metadata (CoinMarketCap)
        defilama_report = get_defilama_fundamentals(ticker, curr_date)
        coinmarketcap_report = get_coinmarketcap_fundamentals(ticker, curr_date)

        sections = []
        if not _is_failure_payload(defilama_report):
            sections.append(defilama_report)
        if not _is_failure_payload(coinmarketcap_report):
            sections.append(coinmarketcap_report)

        if sections:
            return "\n\n---\n\n".join(sections)

        # If both failed, keep existing routing fallback behavior as a last resort.
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)