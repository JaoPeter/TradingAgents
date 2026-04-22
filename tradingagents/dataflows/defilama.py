"""DefiLlama dataflow vendor for DeFi market and protocol data (no API key required)."""

from __future__ import annotations

from datetime import datetime
import os

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


def get_defi_metrics(curr_date: str = None) -> str:
    """Return global DeFi metrics and overview."""
    try:
        resp = requests.get(f"{_BASE}/data/defiSnapshot", timeout=_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama global metrics: {exc}"

    tvl_data = data.get("tvl", {})
    stablecoin = data.get("stablecoins", {})
    mcap = data.get("mcap", {})

    lines = [
        f"Total DeFi TVL (USD): {tvl_data.get('All', {}).get('total')}",
        f"Top DeFi Chain TVL: {tvl_data.get('All', {}).get('Ethereum')}",
        f"Total Stablecoin Market Cap (USD): {stablecoin.get('total')}",
        f"Total DeFi Market Cap (USD): {mcap.get('DeFi')}",
        f"Total Crypto Market Cap (USD): {mcap.get('total')}",
    ]

    header = "# Global DeFi Metrics (DefiLlama)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if curr_date:
        header += f"# Trading date context: {curr_date}\n"
    header += "\n"

    return header + "\n".join([l for l in lines if not l.endswith("None")])


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
    """For DeFi tokens/chains, return relevant TVL and ecosystem metrics."""
    symbol = _normalize_symbol(ticker)
    if not symbol:
        return "Invalid ticker symbol."

    # Try to interpret as a chain name first
    try:
        resp = requests.get(f"{_BASE}/chains", timeout=_API_TIMEOUT)
        resp.raise_for_status()
        chains = resp.json()
    except Exception as exc:
        return f"Error fetching DefiLlama data: {exc}"

    for chain in chains:
        if chain.get("name", "").upper() == symbol:
            lines = [
                f"Blockchain: {chain.get('name')}",
                f"Total TVL (USD): {chain.get('tvl')}",
                f"24h Change: {chain.get('change_1d')}%",
                f"7d Change: {chain.get('change_7d')}%",
            ]
            header = f"# DeFi Metrics for {symbol} (DefiLlama)\n"
            header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            if curr_date:
                header += f"# Trading date context: {curr_date}\n"
            header += "\n"
            return header + "\n".join([l for l in lines if not l.endswith("None")])

    return f"No DeFi data found for '{symbol}'. Try a chain name (ethereum, polygon, arbitrum, etc.)."
