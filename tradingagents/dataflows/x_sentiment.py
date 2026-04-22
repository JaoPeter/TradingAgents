"""X (Twitter) sentiment dataflow vendor for crypto sentiment analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import re

import requests

_API_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
_X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_RECENT_SEARCH_MAX_DAYS = 7
_DEFAULT_SENTIMENT_LOOKBACK_HOURS = 24

_SYMBOL_ALIASES = {
    "BTC": ["Bitcoin"],
    "ETH": ["Ethereum", "Ether"],
    "SOL": ["Solana"],
    "XRP": ["Ripple"],
    "ADA": ["Cardano"],
    "DOGE": ["Dogecoin"],
    "BNB": ["Binance Coin", "BNB Chain"],
    "AVAX": ["Avalanche"],
    "LINK": ["Chainlink"],
    "MATIC": ["Polygon", "POL"],
}

_POSITIVE_TERMS = {
    "bullish",
    "breakout",
    "buy",
    "long",
    "accumulate",
    "strength",
    "outperform",
    "support",
    "rebound",
    "momentum",
    "uptrend",
    "moon",
}

_NEGATIVE_TERMS = {
    "bearish",
    "breakdown",
    "sell",
    "short",
    "distribute",
    "weakness",
    "underperform",
    "resistance",
    "dump",
    "downtrend",
    "rug",
    "capitulation",
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "just",
    "your",
    "about",
    "into",
}


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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower())
    return [word for word in words if word not in _STOPWORDS]


def _get_auth_headers() -> dict[str, str]:
    token = os.getenv("X_BEARER_TOKEN", "").strip() or os.getenv("X_API_KEY", "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


# Minimum follower count a post author must have to be included in quality filtering.
# Accounts with fewer followers contribute noise rather than signal.
_MIN_FOLLOWERS_FOR_INCLUSION = 1000

# Minimum character length of a post body (excluding URLs and cashtags) to be considered.
_MIN_MEANINGFUL_CHARS = 60


def _build_query(symbol: str) -> str:
    """Build a high-quality X search query for a crypto symbol.

    Uses cashtag and full-name terms only — hashtags are excluded because they
    attract low-signal promotional noise. Replies and retweets are excluded to
    focus on original, substantive opinions.
    """
    aliases = _SYMBOL_ALIASES.get(symbol, [])
    # Cashtag is the strongest signal: $BTC is an explicit financial reference.
    # Full-name aliases add recall for analysts who write prose rather than tags.
    # Hashtags (#BTC) are intentionally omitted — they appear heavily in spam/promotions.
    search_terms = [f'"${symbol}"']
    search_terms.extend(f'"{alias}"' for alias in aliases)
    symbol_clause = "(" + " OR ".join(search_terms) + ")"
    return f"{symbol_clause} lang:en -is:retweet -is:reply"


def _resolve_recent_window(start_date: str, end_date: str) -> tuple[datetime, datetime] | tuple[None, None]:
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date) + timedelta(days=1) - timedelta(seconds=1)
    except ValueError:
        return None, None

    if start_dt > end_dt:
        return None, None

    now = _utc_now()
    min_allowed = now - timedelta(days=_RECENT_SEARCH_MAX_DAYS)
    if end_dt < min_allowed:
        return None, None

    return max(start_dt, min_allowed), min(end_dt, now)


def _resolve_sentiment_window(curr_date: str | None) -> tuple[datetime, datetime] | tuple[None, None]:
    now = _utc_now()
    if not curr_date:
        end_dt = now
    else:
        try:
            end_dt = _parse_date(curr_date) + timedelta(days=1) - timedelta(seconds=1)
        except ValueError:
            return None, None

    min_allowed = now - timedelta(days=_RECENT_SEARCH_MAX_DAYS)
    if end_dt < min_allowed:
        return None, None

    end_dt = min(end_dt, now)
    start_dt = max(end_dt - timedelta(hours=_DEFAULT_SENTIMENT_LOOKBACK_HOURS), min_allowed)
    return start_dt, end_dt


def _engagement_score(tweet: dict, user: dict | None) -> float:
    """Composite engagement score that rewards reach, discussion, and author authority.

    Weights:
    - Quotes (1.5×): deliberate commentary — higher signal than a passive like.
    - Retweets (2×): amplification indicates the opinion resonated.
    - Replies (1×): engagement, but can be noise (bots/arguments).
    - Likes (0.5×): weakest signal — cheap to give.
    - Author followers (capped at 50): authority proxy without dominating the score.
    - Verified bonus (+15): verified accounts are less likely to be bot farms.
    """
    public_metrics = tweet.get("public_metrics", {})
    likes = public_metrics.get("like_count", 0)
    retweets = public_metrics.get("retweet_count", 0)
    replies = public_metrics.get("reply_count", 0)
    quotes = public_metrics.get("quote_count", 0)
    followers = (user or {}).get("public_metrics", {}).get("followers_count", 0)
    verified_bonus = 15 if (user or {}).get("verified") else 0
    return (
        (0.5 * likes)
        + (2.0 * retweets)
        + (1.5 * quotes)
        + (1.0 * replies)
        + min(followers / 1000, 50)
        + verified_bonus
    )


def _sentiment_score(text: str) -> float:
    keywords = set(_extract_keywords(text))
    positive_hits = len(keywords & _POSITIVE_TERMS)
    negative_hits = len(keywords & _NEGATIVE_TERMS)
    raw_score = positive_hits - negative_hits

    lowered = (text or "").lower()
    if "$" in lowered and "buy" in lowered:
        raw_score += 0.5
    if "$" in lowered and "sell" in lowered:
        raw_score -= 0.5

    return raw_score


def _sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "bullish"
    if score <= -0.35:
        return "bearish"
    return "neutral"


def _meaningful_text_length(text: str) -> int:
    """Return character count after stripping URLs, cashtags, and @-mentions."""
    stripped = re.sub(r"https?://\S+", "", text)
    stripped = re.sub(r"[$#@]\w+", "", stripped)
    return len(stripped.strip())


def _dedupe_and_rank_tweets(payload: dict) -> list[dict]:
    """Filter low-quality posts, deduplicate, and rank by engagement then sentiment.

    Quality gates (applied in order):
    1. Minimum meaningful character count — excludes link-only or tag-only posts.
    2. Minimum author follower count — excludes new/bot accounts.
    3. Exact-text deduplication — excludes copy-paste spam.
    4. Minimum engagement threshold — at least one meaningful interaction OR
       a non-zero sentiment signal from an established author.

    Ranking: primary key is engagement score (reach and discussion depth);
    secondary key is |sentiment score| so strongly opinionated posts surface
    when two posts have identical engagement.
    """
    users_by_id = {
        user["id"]: user
        for user in payload.get("includes", {}).get("users", [])
        if "id" in user
    }

    ranked = []
    seen_texts: set[str] = set()

    for tweet in payload.get("data", []):
        text = _normalize_text(tweet.get("text", ""))

        # Gate 1: minimum substantive text length
        if _meaningful_text_length(text) < _MIN_MEANINGFUL_CHARS:
            continue

        # Gate 2: author quality
        user = users_by_id.get(tweet.get("author_id"))
        followers = (user or {}).get("public_metrics", {}).get("followers_count", 0)
        if followers < _MIN_FOLLOWERS_FOR_INCLUSION:
            continue

        # Gate 3: deduplication
        dedupe_key = text.lower()
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)

        engagement = _engagement_score(tweet, user)
        sentiment = _sentiment_score(text)

        # Gate 4: require at least some signal (engagement or clear directional opinion)
        if engagement < 2 and abs(sentiment) < 1.0:
            continue

        ranked.append(
            {
                "id": tweet.get("id", ""),
                "text": text,
                "created_at": tweet.get("created_at", ""),
                "author": user or {},
                "public_metrics": tweet.get("public_metrics", {}),
                "engagement_score": engagement,
                "sentiment_score": sentiment,
                "sentiment_label": _sentiment_label(sentiment),
                "keywords": _extract_keywords(text)[:8],
            }
        )

    return sorted(
        ranked,
        key=lambda item: (item["engagement_score"], abs(item["sentiment_score"])),
        reverse=True,
    )


def _fetch_recent_search(query: str, start_dt: datetime, end_dt: datetime, max_results: int = 25) -> dict:
    headers = _get_auth_headers()
    if not headers:
        return {"error": "X_API_KEY not set in .env or X_BEARER_TOKEN is empty."}

    params = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),
        "sort_order": "relevancy",
        "start_time": _to_rfc3339(start_dt),
        "end_time": _to_rfc3339(end_dt),
        "tweet.fields": "created_at,lang,public_metrics,text,author_id",
        "expansions": "author_id",
        "user.fields": "name,username,verified,public_metrics",
    }

    response = requests.get(_X_SEARCH_URL, params=params, headers=headers, timeout=_API_TIMEOUT)
    if response.status_code == 401:
        return {"error": "X authentication failed. Ensure X_API_KEY contains a valid bearer token or set X_BEARER_TOKEN."}
    if response.status_code == 403:
        return {"error": "X API access forbidden for recent search. Ensure your app has X API v2 recent search access."}
    if response.status_code == 429:
        return {"error": "X API rate limit reached."}

    response.raise_for_status()
    return response.json()


def _format_tweet_line(tweet: dict) -> str:
    author = tweet.get("author", {})
    username = author.get("username", "unknown")
    metrics = tweet.get("public_metrics", {})
    return (
        f"- @{username} | {tweet['sentiment_label']} | engagement={tweet['engagement_score']:.1f} | "
        f"likes={metrics.get('like_count', 0)} retweets={metrics.get('retweet_count', 0)} replies={metrics.get('reply_count', 0)}\n"
        f"  {tweet['text']}"
    )


def get_sentiment_summary(ticker: str, curr_date: str = None) -> str:
    """Return X/Twitter sentiment for a crypto asset using recent search results."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    if not _get_auth_headers():
        return (
            f"X_API_KEY not set in .env. "
            f"For X (Twitter) sentiment, set a bearer token in X_API_KEY or X_BEARER_TOKEN."
        )

    try:
        start_dt, end_dt = _resolve_sentiment_window(curr_date)
        if not start_dt or not end_dt:
            return (
                f"Error fetching X sentiment for {symbol}: requested date is outside X recent-search retention. "
                f"The standard X recent search API only supports approximately the last {_RECENT_SEARCH_MAX_DAYS} days."
            )

        query = _build_query(symbol)
        payload = _fetch_recent_search(query, start_dt, end_dt, max_results=30)
        if "error" in payload:
            return f"Error fetching X sentiment for {symbol}: {payload['error']}"

        ranked_tweets = _dedupe_and_rank_tweets(payload)
    except Exception as exc:
        return (
            f"Error fetching X sentiment for {symbol}: {exc}\n"
            f"Ensure X_API_KEY is configured as a bearer token and has X API v2 recent-search permissions."
        )

    if not ranked_tweets:
        return (
            f"# X (Twitter) Sentiment for {symbol}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Search criteria: {_build_query(symbol)}\n"
            f"# Considered window: {_to_rfc3339(start_dt)} to {_to_rfc3339(end_dt)}\n\n"
            f"Sentiment: neutral\n"
            f"Sentiment Score: 0.00\n"
            f"Mention Count (considered): 0\n"
            f"Engagement Score (aggregate): 0.0\n"
            f"Criteria Applied: English only, no retweets, no replies, deduplicated posts, low-signal posts filtered."
        )

    weighted_score_total = sum(tweet["sentiment_score"] * max(tweet["engagement_score"], 1.0) for tweet in ranked_tweets)
    engagement_total = sum(tweet["engagement_score"] for tweet in ranked_tweets)
    normalized_score = weighted_score_total / max(engagement_total, 1.0)
    sentiment = _sentiment_label(normalized_score)
    top_tweets = ranked_tweets[:5]

    header = f"# X (Twitter) Sentiment for {symbol}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += f"# Search criteria: {_build_query(symbol)}\n"
    header += f"# Considered window: {_to_rfc3339(start_dt)} to {_to_rfc3339(end_dt)}\n"
    header += "\n"

    lines = [
        f"Sentiment: {sentiment}",
        f"Sentiment Score: {normalized_score:.2f}",
        f"Mention Count (considered): {len(ranked_tweets)}",
        f"Engagement Score (aggregate): {engagement_total:.1f}",
        "Criteria Applied: English only, no retweets, no replies, deduplicated posts, low-signal posts filtered.",
        "Top Posts Considered:",
    ]
    lines.extend(_format_tweet_line(tweet) for tweet in top_tweets)

    return header + "\n".join(lines)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch crypto discussion trends from X recent search for a ticker."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    if not _get_auth_headers():
        return "Error fetching X discussion trends: X_API_KEY not set in .env or X_BEARER_TOKEN is empty."

    start_dt, end_dt = _resolve_recent_window(start_date, end_date)
    if not start_dt or not end_dt:
        return (
            f"Error fetching X discussion trends for {symbol}: requested date window is outside X recent-search retention "
            f"or has an invalid date range."
        )

    try:
        query = _build_query(symbol)
        payload = _fetch_recent_search(query, start_dt, end_dt, max_results=25)
        if "error" in payload:
            return f"Error fetching X discussion trends for {symbol}: {payload['error']}"

        ranked_tweets = _dedupe_and_rank_tweets(payload)
    except Exception as exc:
        return f"Error fetching X discussion trends for {symbol}: {exc}"

    if not ranked_tweets:
        return f"No data found for X discussion trends for '{symbol}' between {start_date} and {end_date}"

    header = f"## X (Twitter) Discussion Trends for {symbol}\n"
    header += f"Search criteria: {_build_query(symbol)}\n"
    header += f"Window: {_to_rfc3339(start_dt)} to {_to_rfc3339(end_dt)}\n"
    header += "Criteria applied: English only, no retweets, no replies, deduplicated posts, low-signal posts filtered.\n\n"

    body = []
    for tweet in ranked_tweets[:10]:
        body.append(_format_tweet_line(tweet))

    return header + "\n".join(body)


# Re-export as generic news interface
get_global_news = get_news
