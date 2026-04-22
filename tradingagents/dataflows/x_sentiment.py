"""X (Twitter) sentiment dataflow vendor for crypto sentiment analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests

# Note: This is a placeholder for X/Twitter sentiment integration.
# Actual X API requires authentication and special access.
# For now, this uses public sentiment aggregators or alternative sources.


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


def get_sentiment_summary(ticker: str, curr_date: str = None) -> str:
    """Return X/Twitter sentiment for a crypto asset using aggregated data."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    api_key = os.getenv("X_API_KEY", "").strip()
    if not api_key:
        return (
            f"X_API_KEY not set in .env. "
            f"For X (Twitter) sentiment, apply at https://developer.twitter.com/"
        )

    # Placeholder: Sentiment APIs often require custom endpoint setup
    # This is a template for when X API integration is available
    try:
        # Example structure - adjust per actual API
        resp = requests.get(
            "https://api.example.com/sentiment",
            params={"symbol": symbol, "source": "twitter"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return (
            f"Error fetching X sentiment for {symbol}: {exc}\n"
            f"Ensure X_API_KEY is configured and has appropriate permissions."
        )

    sentiment = data.get("sentiment", "neutral")
    score = data.get("score", 0)
    mention_count = data.get("mentions", 0)
    engagement = data.get("engagement_score", 0)

    header = f"# X (Twitter) Sentiment for {symbol}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += "\n"

    lines = [
        f"Sentiment: {sentiment}",
        f"Sentiment Score: {score}",
        f"Mention Count (24h): {mention_count}",
        f"Engagement Score: {engagement}",
    ]

    return header + "\n".join(lines)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Placeholder: Fetch crypto discussion trends from X for a ticker."""
    symbol = _normalize_symbol(ticker)

    return (
        f"## X (Twitter) Discussion Trends for {symbol}\n\n"
        f"X API integration requires authentication at https://developer.twitter.com/\n"
        f"Set X_API_KEY in .env after obtaining API credentials.\n"
        f"Once configured, this will aggregate tweets and sentiment for {symbol}.\n"
    )


# Re-export as generic news interface
get_global_news = get_news

import os
