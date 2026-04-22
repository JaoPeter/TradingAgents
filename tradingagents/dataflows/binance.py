"""Binance dataflow vendor for crypto market OHLCV."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests


# Crypto is always priced against USDC — a fully-reserved USD stablecoin.
# Unlike USDT, USDC is audited and provides a transparent USD reference.
_DEFAULT_QUOTE = "USDC"


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbols like BTC, BTC-USD, BTCUSDC into Binance USDC-quoted format.

    All ambiguous or USD-denominated inputs are mapped to the USDC pair:
      BTC        → BTCUSDC
      BTC-USD    → BTCUSDC
      BTC-USDT   → BTCUSDC   (re-quoted to USDC)
      BTCUSDC    → BTCUSDC   (unchanged)
      BTCUSDT    → BTCUSDT   (explicit USDT kept as-is)
    """
    raw = (symbol or "").upper().strip().replace("/", "").replace("_", "")
    if not raw:
        return raw

    if "-" in raw:
        base, quote = raw.split("-", 1)
        # Map generic USD / stablecoin quotes to USDC
        if quote in ("USD", "USDT", "BUSD"):
            quote = _DEFAULT_QUOTE
        return f"{base}{quote}"

    # Already has an explicit stablecoin suffix — keep it unchanged
    if raw.endswith("USDC") or raw.endswith("USDT") or raw.endswith("BUSD"):
        return raw

    # Strip a bare USD suffix and re-quote to USDC
    if raw.endswith("USD"):
        return raw[:-3] + _DEFAULT_QUOTE

    # Plain base ticker like BTC / ETH / SOL → default USDC pair
    return f"{raw}{_DEFAULT_QUOTE}"


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Return Binance OHLCV klines as CSV-compatible text."""
    pair = _normalize_symbol(symbol)

    try:
        start_ms = _to_ms(start_date)
        end_ms = _to_ms(end_date)
    except ValueError:
        return "Invalid date format. Use yyyy-mm-dd."

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": pair,
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    headers = {}
    api_key = ""
    try:
        import os

        api_key = os.getenv("BINANCE_API_KEY", "").strip()
    except Exception:
        api_key = ""

    if api_key:
        headers["X-MBX-APIKEY"] = api_key

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=int(os.getenv("API_TIMEOUT_SECONDS", "30")))
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        return f"Error fetching Binance klines for {pair}: {exc}"

    if not rows:
        return f"No Binance data found for symbol '{pair}' between {start_date} and {end_date}"

    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    frame["Date"] = pd.to_datetime(frame["open_time"], unit="ms").dt.strftime("%Y-%m-%d")
    out = frame[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values("Date")

    header = f"# Binance OHLCV data for {pair} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(out)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + out.to_csv(index=False)
