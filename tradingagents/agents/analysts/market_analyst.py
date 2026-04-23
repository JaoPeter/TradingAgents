from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_autonomous_evidence_instruction,
    get_indicators,
    invoke_with_optional_tools,
    get_language_instruction,
    get_lookback_days,
    get_stock_data,
    _TF_INDICATORS,
    get_dependent_timeframes,
    build_multi_timeframe_context,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tf_context = build_timeframe_context(state)
        primary_tf = state.get("primary_tf", "1d")
        lookback = state.get("lookback_days", get_lookback_days(primary_tf))
        preferred_indicators = _TF_INDICATORS.get(primary_tf, "SMA(50), SMA(200), ATR, RSI(14), MACD, Bollinger Bands, VWMA")
        
        # Multi-timeframe context
        dependent_tfs = get_dependent_timeframes(primary_tf)
        mtf_context = build_multi_timeframe_context(primary_tf)

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            f"""{tf_context}\n\n{mtf_context}\n\nAnalyze market structure with a multi-timeframe approach.

Use tools in this order:
1. Call get_stock_data with lookback={lookback} on {primary_tf}.
2. Call get_indicators with up to 8 relevant indicators, prioritizing: {preferred_indicators}.
3. If available, use higher-timeframe confirmation from: {', '.join(dependent_tfs) if dependent_tfs else 'N/A'}.

Output format (max 220 words, concise):
- Bias: bullish / bearish / neutral
- 3 key evidence bullets
- 2 key risks
- 1 actionable setup aligned to {state.get('trading_style', 'swing')} on {primary_tf}
- 1 markdown table with exactly 4 rows: Signal | Evidence | Risk | Action

Tool-call rule: when calling get_indicators, use exact supported keys only:
close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma.
"""
            + """ Do not ask the user what to do next. Do not ask follow-up questions. You must make decisions from available evidence and deliver a clear directional view, key risks, and concrete next action."""
            + get_autonomous_evidence_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result = invoke_with_optional_tools(prompt, llm, tools, state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
