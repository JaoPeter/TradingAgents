import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)

# Common crypto symbols and patterns
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "DOGE", "SHIB", "SOL", "XRP", "ADA", "AVAX", "MATIC", "LINK",
    "LITECOIN", "BITCOIN", "ETHEREUM", "CARDANO", "SOLANA", "RIPPLE", "POLKADOT",
    "UNI", "SUSHI", "AAVE", "CURVE", "PANCAKE", "UNISWAP",
}

def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol is likely a crypto asset (24/7 markets).
    
    Crypto indicators include: known crypto tickers, USDT/BUSD/USDC suffixes,
    -USD suffix, or uppercase patterns like BTC-USDT.
    """
    if not symbol:
        return False
    
    upper_sym = symbol.upper().strip()
    
    # Check against known crypto symbols
    base = upper_sym.split("-")[0].split("USDT")[0].split("BUSD")[0].split("USDC")[0].split("USD")[0]
    if base in CRYPTO_SYMBOLS:
        return True
    
    # Check for stablecoin suffixes
    if any(upper_sym.endswith(suffix) for suffix in ["USDT", "BUSD", "USDC", "-USD"]):
        return True
    
    # Check for crypto pairs like BTC-USDT, ETH-BUSD
    if "-" in upper_sym:
        parts = upper_sym.split("-")
        if parts[0] in CRYPTO_SYMBOLS:
            return True
    
    return False


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    
    For crypto symbols (24/7 markets), uses Binance API directly.
    For stocks, uses yfinance (markets closed on weekends).
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Check if this is a crypto symbol
    if is_crypto_symbol(symbol):
        return _load_crypto_ohlcv(symbol, curr_date_dt)
    else:
        return _load_stock_ohlcv(symbol, curr_date_dt)


def _load_crypto_ohlcv(symbol: str, curr_date_dt) -> pd.DataFrame:
    """Load crypto OHLCV from Binance (24/7 markets, all days included)."""
    from datetime import timedelta
    from . import binance
    
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")
    
    config = get_config()
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-Binance-data-{start_str}-{end_str}.csv",
    )
    
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        # Fetch from Binance
        csv_str = binance.get_stock_data(symbol, start_str, end_str)
        if csv_str.startswith("Error") or csv_str.startswith("No"):
            logger.warning(f"Failed to fetch crypto data for {symbol} from Binance: {csv_str}")
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        
        from io import StringIO
        data = pd.read_csv(StringIO(csv_str))
        data.to_csv(data_file, index=False)
    
    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    return data


def _load_stock_ohlcv(symbol: str, curr_date_dt) -> pd.DataFrame:
    """Load stock OHLCV from yfinance (market hours only, weekdays)."""
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")
    
    config = get_config()
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            # For crypto (24/7 markets), use the most recent data point available
            # For stocks, no data = not a trading day
            if is_crypto_symbol(symbol) and not df.empty:
                latest_value = df[indicator].iloc[-1]
                latest_date = df["Date"].iloc[-1]
                return f"{latest_value} (last available: {latest_date})"
            else:
                return "N/A: Not a trading day (weekend or holiday)"
