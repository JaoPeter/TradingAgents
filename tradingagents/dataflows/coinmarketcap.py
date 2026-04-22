"""CoinMarketCap dataflow vendor for crypto market data and fundamentals."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

_BASE = "https://pro-api.coinmarketcap.com/v1"

import os
_API_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "30"))


def _normalize_symbol(symbol: str) -> str:
    raw = (symbol or "").upper().strip()
    if not raw:
        return raw
    if "-" in raw:
        return raw.split("-", 1)[0]
    if raw.endswith("USDT"):
        return raw[:-4]
    if raw.endswith("USD"):
        return raw[:-3]
    return raw


def _headers() -> dict[str, str]:
    import os

    key = os.getenv("COINMARKETCAP_API_KEY", "").strip()
    if not key:
        return {}
    return {"X-CMC_PRO_API_KEY": key}


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Return CoinMarketCap market info and fundamentals."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    key = os.getenv("COINMARKETCAP_API_KEY", "").strip()
    if not key:
        return f"COINMARKETCAP_API_KEY not set in .env"

    try:
        resp = requests.get(
            f"{_BASE}/cryptocurrency/info",
            params={"symbol": symbol},
            headers=_headers(),
            timeout=_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching CoinMarketCap info for {symbol}: {exc}"

    if "data" not in data or symbol not in data["data"]:
        return f"No CoinMarketCap data found for symbol '{symbol}'."

    coin_info = data["data"][symbol]

    lines = [
        f"Name: {coin_info.get('name')}",
        f"Symbol: {coin_info.get('symbol')}",
        f"Category: {coin_info.get('category')}",
        f"Description: {(coin_info.get('description') or '')[:200]}",
        f"Date Added: {coin_info.get('date_added')}",
        f"Website: {', '.join(coin_info.get('urls', {}).get('website', []))}",
        f"Technical Docs: {', '.join(coin_info.get('urls', {}).get('technical_docs', []))}",
        f"Explorer: {', '.join(coin_info.get('urls', {}).get('explorer', []))}",
        f"Source Code: {', '.join(coin_info.get('urls', {}).get('source_code', []))}",
        f"Message Board: {', '.join(coin_info.get('urls', {}).get('message_board', []))}",
        f"Chat: {', '.join(coin_info.get('urls', {}).get('chat', []))}",
        f"Announcement: {', '.join(coin_info.get('urls', {}).get('announcement', []))}",
        f"Reddit: {', '.join(coin_info.get('urls', {}).get('reddit', []))}",
    ]

    cleaned = [l for l in lines if not l.endswith("[]")]
    header = f"# Crypto Fundamentals for {symbol} (CoinMarketCap)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += "\n"

    return header + "\n".join(cleaned)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Return CoinMarketCap historical price data."""
    sym = _normalize_symbol(symbol)
    if not sym:
        return "Invalid ticker symbol."

    key = os.getenv("COINMARKETCAP_API_KEY", "").strip()
    if not key:
        return f"COINMARKETCAP_API_KEY not set in .env"

    try:
        resp = requests.get(
            f"{_BASE}/cryptocurrency/quotes/historical",
            params={"symbol": sym, "time_start": start_date, "time_end": end_date},
            headers=_headers(),
            timeout=_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching CoinMarketCap historical data for {sym}: {exc}"

    if "data" not in data or sym not in data["data"]:
        return f"No CoinMarketCap historical data found for symbol '{sym}'."

    import pandas as pd

    quotes = data["data"][sym]
    rows = []
    for quote in quotes:
        ts = quote.get("timestamp", "")
        usd_data = quote.get("quote", {}).get("USD", {})
        rows.append({
            "Date": ts[:10] if ts else "",
            "Open": usd_data.get("open"),
            "High": usd_data.get("high"),
            "Low": usd_data.get("low"),
            "Close": usd_data.get("price"),
            "Volume": usd_data.get("volume_24h"),
        })

    if not rows:
        return f"No quote data for {sym} in range {start_date} to {end_date}"

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("Date")

    header = f"# CoinMarketCap OHLCV for {sym} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(frame)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + frame.to_csv(index=False)
