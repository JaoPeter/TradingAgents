"""Crypto RSS news dataflow vendor (no API key required)."""

from __future__ import annotations

from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
import urllib.parse
import xml.etree.ElementTree as ET

import requests

_RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
]


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def _extract_items(feed_name: str, xml_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    root = ET.fromstring(xml_text)

    # RSS 2.0
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = (item.findtext("title") or "No title").strip()
            link = (item.findtext("link") or "").strip()
            pub = _parse_date((item.findtext("pubDate") or "").strip())
            desc = (item.findtext("description") or "").strip()
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published_at": pub,
                    "source": feed_name,
                    "summary": desc,
                }
            )
        return items

    # Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="No title", namespaces=ns) or "No title").strip()
        link_elem = entry.find("atom:link", ns)
        link = ""
        if link_elem is not None:
            link = (link_elem.get("href") or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        pub = None
        if updated:
            try:
                pub = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except Exception:
                pub = None
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        items.append(
            {
                "title": title,
                "link": link,
                "published_at": pub,
                "source": feed_name,
                "summary": summary,
            }
        )
    return items


def _fetch_all_items() -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    for name, url in _RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            all_items.extend(_extract_items(name, resp.text))
        except Exception:
            continue

    all_items.sort(
        key=lambda x: x.get("published_at") or datetime.min,
        reverse=True,
    )
    return all_items


def _clean_html(text: str) -> str:
    # Keep this dependency-free: minimal cleanup for common RSS description tags.
    replacements = {
        "<p>": "",
        "</p>": "",
        "<br>": " ",
        "<br/>": " ",
        "<br />": " ",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def _format(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No RSS news found.\n\n{title}"

    lines = [title, ""]
    for row in rows:
        lines.append(f"### {row['title']} (source: {row['source']})")
        if row.get("published_at"):
            lines.append(f"Published: {row['published_at']}")
        if row.get("summary"):
            lines.append(_clean_html(row["summary"])[:500])
        if row.get("link"):
            lines.append(f"Link: {row['link']}")
        lines.append("")
    return "\n".join(lines)


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


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch ticker-related news from curated crypto RSS feeds."""
    symbol = _normalize_symbol(ticker)
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use yyyy-mm-dd."

    items = _fetch_all_items()

    out: list[dict[str, Any]] = []
    q = symbol.lower()
    for item in items:
        published = item.get("published_at")
        if published is None:
            continue
        published_naive = published.replace(tzinfo=None)
        if not (start_dt <= published_naive <= end_dt + timedelta(days=1)):
            continue

        haystack = " ".join(
            [
                item.get("title", ""),
                item.get("summary", ""),
                urllib.parse.unquote(item.get("link", "")),
            ]
        ).lower()
        if q in haystack:
            out.append(item)

    header = f"## Crypto RSS News for {symbol}, from {start_date} to {end_date}:"
    return _format(header, out[:25])


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 20) -> str:
    """Fetch broad crypto market news from curated RSS feeds."""
    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use yyyy-mm-dd."

    start_dt = curr_dt - timedelta(days=max(0, look_back_days))
    items = _fetch_all_items()

    out: list[dict[str, Any]] = []
    for item in items:
        published = item.get("published_at")
        if published is None:
            continue
        published_naive = published.replace(tzinfo=None)
        if start_dt <= published_naive <= curr_dt + timedelta(days=1):
            out.append(item)

    header = (
        f"## Global Crypto RSS News, from {start_dt.strftime('%Y-%m-%d')} "
        f"to {curr_date}:"
    )
    return _format(header, out[: max(1, limit)])
