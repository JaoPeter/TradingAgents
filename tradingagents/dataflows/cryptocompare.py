"""CryptoCompare dataflow vendor for crypto news, sentiment and market analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import os

import requests

_BASE = "https://min-api.cryptocompare.com/data"
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

    key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    if not key:
        return {}
    return {"authorization": f"Apikey {key}"}


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch news for a crypto asset from CryptoCompare."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    try:
        resp = requests.get(
            f"{_BASE}/v1/news/",
            params={
                "lang": "EN",
                "feeds": "cointelegraph,coindesk",
                "sortOrder": "latest",
            },
            headers=_headers(),
            timeout=_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching CryptoCompare news for {symbol}: {exc}"

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use yyyy-mm-dd."

    articles = data.get("Data", [])
    filtered: list[dict[str, Any]] = []
    q = symbol.lower()

    for article in articles:
        published = article.get("published_on", 0)
        if published:
            pub_dt = datetime.fromtimestamp(published)
        else:
            pub_dt = datetime.now()

        if not (start_dt <= pub_dt <= end_dt):
            continue

        haystack = " ".join([
            article.get("title", ""),
            article.get("body", ""),
            article.get("tags", ""),
        ]).lower()

        if q in haystack:
            filtered.append(article)

    lines = []
    for article in filtered[:25]:
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        url = article.get("url", "")
        published_on = article.get("published_on", 0)
        if published_on:
            pub_str = datetime.fromtimestamp(published_on).strftime("%Y-%m-%d")
        else:
            pub_str = "Unknown"

        lines.append(f"### {title} (source: {source})")
        lines.append(f"Published: {pub_str}")
        if url:
            lines.append(f"Link: {url}")
        lines.append("")

    header = f"## CryptoCompare News for {symbol}, from {start_date} to {end_date}:"
    if not lines:
        return f"{header}\n\nNo news found."
    return f"{header}\n\n" + "\n".join(lines)


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 20) -> str:
    """Fetch broad crypto news from CryptoCompare."""
    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use yyyy-mm-dd."

    start_dt = curr_dt - timedelta(days=max(0, look_back_days))

    try:
        resp = requests.get(
            f"{_BASE}/v1/news/",
            params={
                "lang": "EN",
                "feeds": "cointelegraph,coindesk,decrypt",
                "sortOrder": "latest",
            },
            headers=_headers(),
            timeout=_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching CryptoCompare global news: {exc}"

    articles = data.get("Data", [])
    filtered: list[dict[str, Any]] = []

    for article in articles:
        published = article.get("published_on", 0)
        if published:
            pub_dt = datetime.fromtimestamp(published)
        else:
            pub_dt = datetime.now()

        if start_dt <= pub_dt <= curr_dt + timedelta(days=1):
            filtered.append(article)

    lines = []
    for article in filtered[: max(1, limit)]:
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        url = article.get("url", "")
        published_on = article.get("published_on", 0)
        if published_on:
            pub_str = datetime.fromtimestamp(published_on).strftime("%Y-%m-%d")
        else:
            pub_str = "Unknown"

        lines.append(f"### {title} (source: {source})")
        lines.append(f"Published: {pub_str}")
        if url:
            lines.append(f"Link: {url}")
        lines.append("")

    header = (
        f"## Global Crypto News from CryptoCompare, "
        f"from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:"
    )
    if not lines:
        return f"{header}\n\nNo news found."
    return f"{header}\n\n" + "\n".join(lines)
