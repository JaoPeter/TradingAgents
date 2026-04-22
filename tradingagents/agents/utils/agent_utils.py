from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
    get_sentiment_summary,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


# Lookback days per primary timeframe
_TF_LOOKBACK_DAYS: dict = {
    "15m": 3,
    "1h":  7,
    "4h":  14,
    "1d":  30,
    "1w":  180,
}

# Human-readable time horizon per timeframe
_TF_HORIZON: dict = {
    "15m": "intraday (1-8 hours)",
    "1h":  "short-term (hours to 2 days)",
    "4h":  "short-term (2-7 days)",
    "1d":  "medium-term (1-4 weeks)",
    "1w":  "long-term (1-6 months)",
}

# Preferred indicators per timeframe
_TF_INDICATORS: dict = {
    "15m": "EMA(9), EMA(21), VWAP, ATR, RSI(14), MACD, Bollinger Bands",
    "1h":  "EMA(21), EMA(50), ATR, RSI(14), MACD, Bollinger Bands, VWMA",
    "4h":  "EMA(50), SMA(200), ATR, RSI(14), MACD, Bollinger Bands, VWMA",
    "1d":  "SMA(50), SMA(200), ATR, RSI(14), MACD, Bollinger Bands, VWMA",
    "1w":  "SMA(50), SMA(200), RSI(14), MACD, Bollinger Bands",
}

# Dependent timeframes for multi-timeframe analysis (higher TFs for context/confirmation)
# news and social media are excluded as they only look back max 2 weeks
_TF_DEPENDENCIES: dict = {
    "15m": ["1h", "4h", "1d"],      # 15m confirms on 1h, 4h, daily
    "1h":  ["4h", "1d"],            # 1h confirms on 4h, daily
    "4h":  ["1d", "1w", "1M"],      # 4h confirms on daily, weekly, monthly
    "1d":  ["1w", "1M"],            # daily confirms on weekly, monthly
    "1w":  ["1M"],                  # weekly confirms on monthly
}


def get_lookback_days(primary_tf: str) -> int:
    """Return number of days to look back for news/sentiment based on timeframe."""
    return _TF_LOOKBACK_DAYS.get(primary_tf, 30)


def get_dependent_timeframes(primary_tf: str, include_primary: bool = False) -> list:
    """Get all higher timeframes that should be analyzed for multi-timeframe confirmation.
    
    Args:
        primary_tf: Primary trading timeframe (15m, 1h, 4h, 1d, 1w)
        include_primary: If True, include primary_tf in the returned list
    
    Returns:
        List of higher timeframes for confirmation/context (news/social excluded)
    
    Example:
        get_dependent_timeframes("15m") → ["1h", "4h", "1d"]
        get_dependent_timeframes("4h")  → ["1d", "1w", "1M"]
        get_dependent_timeframes("1d")  → ["1w", "1M"]
    """
    deps = _TF_DEPENDENCIES.get(primary_tf, [])
    if include_primary:
        return [primary_tf] + deps
    return deps


def build_multi_timeframe_context(primary_tf: str) -> str:
    """Build a context string for analyzing multiple timeframes.
    
    Returns information about which higher timeframes should be analyzed
    for confirmation and context.
    """
    deps = get_dependent_timeframes(primary_tf)
    if not deps:
        return f"Single timeframe analysis: {primary_tf} (no higher timeframes available)"
    
    return (
        f"Multi-timeframe analysis: {primary_tf} (primary) + "
        f"{', '.join(deps)} (confirmation/context)"
    )



def build_timeframe_context(state: dict) -> str:
    """Build a concise timeframe/style context string for agent prompts."""
    style = state.get("trading_style", "swing")
    tf = state.get("primary_tf", "1d")
    conf = state.get("confirm_tf", "")
    lookback = state.get("lookback_days", get_lookback_days(tf))
    horizon = _TF_HORIZON.get(tf, "medium-term")
    indicators = _TF_INDICATORS.get(tf, "standard indicators")

    ctx = (
        f"Trading context: {style.upper()} strategy | "
        f"Primary timeframe: {tf} | "
        f"Time horizon: {horizon} | "
        f"Preferred indicators: {indicators} | "
        f"Data lookback: {lookback} days"
    )
    if conf:
        ctx += f" | Confirmation timeframe: {conf}"
    return ctx


def _is_ollama_tool_support_error(exc: Exception) -> bool:
    """Detect Ollama errors where the selected model cannot use tool calling."""
    message = str(exc).lower()
    return "does not support tools" in message and (
        "registry.ollama.ai" in message or "ollama" in message
    )


def invoke_with_optional_tools(prompt, llm, tools, messages):
    """Invoke with tools first and gracefully fall back when Ollama model lacks tool support."""
    chain = prompt | llm.bind_tools(tools)
    try:
        return chain.invoke(messages)
    except Exception as exc:
        if _is_ollama_tool_support_error(exc):
            print("[warning] Selected Ollama model does not support tools. Retrying without tools.")
            return (prompt | llm).invoke(messages)
        raise

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
