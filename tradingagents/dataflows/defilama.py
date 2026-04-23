"""DefiLlama dataflow vendor for DeFi market and protocol data (no API key required)."""

from __future__ import annotations

from datetime import datetime
import os
import re

import requests

_BASE = "https://api.llama.fi"
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


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug


def _resolve_protocol_match(protocols: list[dict], symbol: str) -> dict | None:
    symbol_key = _norm_key(symbol)

    # 1) Strict match on protocol symbol/name
    for proto in protocols:
        if _norm_key(proto.get("symbol", "")) == symbol_key:
            return proto
        if _norm_key(proto.get("name", "")) == symbol_key:
            return proto

    # 2) Partial name hint for common token vs protocol naming differences
    for proto in protocols:
        name_key = _norm_key(proto.get("name", ""))
        if symbol_key and symbol_key in name_key:
            return proto

    return None


def _try_protocol_endpoint(base_path: str, candidates: list[str]) -> dict | None:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resp = requests.get(f"{_BASE}/{base_path}/{candidate}", timeout=_API_TIMEOUT)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if isinstance(payload, dict) and payload:
                return payload
        except Exception:
            continue
    return None


def get_defi_metrics(curr_date: str = None) -> str:
    """Return global DeFi metrics: total TVL, top chains, stablecoin market cap."""
    header = "# Global DeFi Metrics (DefiLlama)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += "\n"

    # Total TVL across all chains
    try:
        tvl_resp = requests.get(f"{_BASE}/v2/historicalChainTvl", timeout=_API_TIMEOUT)
        tvl_resp.raise_for_status()
        tvl_history = tvl_resp.json()
        if tvl_history:
            latest = tvl_history[-1]
            lines = [f"Total DeFi TVL (USD): {latest.get('tvl')}"]
        else:
            lines = []
    except Exception as exc:
        lines = [f"TVL fetch error: {exc}"]

    # Top chains by TVL
    try:
        chains_resp = requests.get(f"{_BASE}/v2/chains", timeout=_API_TIMEOUT)
        chains_resp.raise_for_status()
        chains = sorted(chains_resp.json(), key=lambda c: c.get("tvl") or 0, reverse=True)
        top = chains[:5]
        lines.append("\nTop 5 Chains by TVL:")
        for c in top:
            lines.append(f"  {c.get('name')}: ${c.get('tvl'):,.0f}" if c.get("tvl") else f"  {c.get('name')}: -")
    except Exception as exc:
        lines.append(f"Chains fetch error: {exc}")

    # Stablecoin market cap
    try:
        sc_resp = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=true", timeout=_API_TIMEOUT)
        if sc_resp.status_code == 200:
            sc_data = sc_resp.json()
            total_mcap = sum(
                sum(v for v in (s.get("circulating") or {}).values() if isinstance(v, (int, float)))
                for s in sc_data.get("peggedAssets", [])
            )
            if total_mcap:
                lines.append(f"\nTotal Stablecoin Market Cap (USD): ${total_mcap:,.0f}")
    except Exception:
        pass

    return header + "\n".join(l for l in lines if l is not None)


def get_chain_tvl(chain: str = "ethereum") -> str:
    """Return TVL for a specific blockchain."""
    try:
        resp = requests.get(f"{_BASE}/chains", timeout=_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama chain data: {exc}"

    chain_norm = chain.lower()
    for chain_data in data:
        if chain_data.get("name", "").lower() == chain_norm:
            return (
                f"# Chain TVL: {chain}\n"
                f"Total TVL (USD): {chain_data.get('tvl')}\n"
                f"24h Change: {chain_data.get('change_1d')}%\n"
            )

    return f"No data found for chain '{chain}'."


def get_protocol_tvl(protocol: str) -> str:
    """Return TVL for a specific DeFi protocol."""
    try:
        resp = requests.get(f"{_BASE}/protocols", timeout=_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama protocol data: {exc}"

    proto_norm = protocol.lower()
    for proto_data in data:
        if proto_data.get("name", "").lower() == proto_norm:
            lines = [
                f"Protocol: {proto_data.get('name')}",
                f"Total TVL (USD): {proto_data.get('tvl')}",
                f"Category: {proto_data.get('category')}",
                f"Chain: {proto_data.get('chains')}",
            ]
            header = f"# Protocol TVL (DefiLlama)\n"
            header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            return header + "\n".join(lines)

    return f"No data found for protocol '{protocol}'."


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """For DeFi tokens/chains, return TVL, fees, DEX volume and ecosystem metrics."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    header = f"# DeFi Fundamentals for {symbol} (DefiLlama)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += "\n"

    # 1. Try protocol lookup by symbol or name
    try:
        resp = requests.get(f"{_BASE}/protocols", timeout=_API_TIMEOUT)
        resp.raise_for_status()
        protocols = resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama protocols: {exc}"

    matched = _resolve_protocol_match(protocols, symbol)

    if matched:
        proto_name = matched.get("name", symbol)
        slug_candidates = []
        for raw in (
            matched.get("slug", ""),
            matched.get("name", ""),
            matched.get("symbol", ""),
            proto_name,
            symbol,
        ):
            slug = _slugify(raw)
            if slug and slug not in slug_candidates:
                slug_candidates.append(slug)
        lines = [
            f"Protocol: {proto_name}",
            f"Symbol: {matched.get('symbol', '-')}",
            f"Category: {matched.get('category', '-')}",
            f"Chains: {', '.join(matched.get('chains') or [])}",
            f"TVL (USD): {matched.get('tvl')}",
            f"TVL 24h Change: {matched.get('change_1d')}%",
            f"TVL 7d Change: {matched.get('change_7d')}%",
            f"TVL 30d Change: {matched.get('change_1m')}%",
        ]
        if matched.get("mcap"):
            lines.append(f"Market Cap (USD): {matched['mcap']}")
        if matched.get("fdv"):
            lines.append(f"FDV (USD): {matched['fdv']}")

        # Fees & revenue
        fee_data = _try_protocol_endpoint("summary/fees", slug_candidates)
        if fee_data:
            if fee_data.get("total24h") is not None:
                lines.append(f"Fees 24h (USD): {fee_data['total24h']}")
            if fee_data.get("totalRevenue24h") is not None:
                lines.append(f"Revenue 24h (USD): {fee_data['totalRevenue24h']}")
            if fee_data.get("total7d") is not None:
                lines.append(f"Fees 7d (USD): {fee_data['total7d']}")

        # DEX volume
        dex_data = _try_protocol_endpoint("summary/dexs", slug_candidates)
        if dex_data:
            if dex_data.get("total24h") is not None:
                lines.append(f"DEX Volume 24h (USD): {dex_data['total24h']}")
            if dex_data.get("total7d") is not None:
                lines.append(f"DEX Volume 7d (USD): {dex_data['total7d']}")

        return header + "\n".join(
            l for l in lines if l.split(": ", 1)[-1] not in ("None", "-", "None%")
        )

    # 2. Fall back to chain lookup
    try:
        chain_resp = requests.get(f"{_BASE}/chains", timeout=_API_TIMEOUT)
        chain_resp.raise_for_status()
        chains = chain_resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama chain data: {exc}"

    for chain in chains:
        if chain.get("name", "").upper() == symbol or chain.get("symbol", "").upper() == symbol:
            lines = [
                f"Blockchain: {chain.get('name')}",
                f"Total TVL (USD): {chain.get('tvl')}",
                f"24h Change: {chain.get('change_1d')}%",
                f"7d Change: {chain.get('change_7d')}%",
                f"30d Change: {chain.get('change_1m')}%",
            ]
            return header + "\n".join(l for l in lines if not l.endswith("None") and not l.endswith("None%"))

    return f"No DeFi data found for '{symbol}'. DefiLlama supports DeFi protocol tokens (e.g. UNI, AAVE) and chain names (ethereum, polygon, arbitrum)."
