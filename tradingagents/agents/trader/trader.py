import functools
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_autonomous_evidence_instruction,
    build_instrument_context,
    build_timeframe_context,
    get_lookback_days,
    get_stock_data,
    invoke_with_optional_tools,
)


def create_trader(llm, memory):
    def _clip(text: str, limit: int = 1200) -> str:
        return (text or "")[:limit]

    def _detect_regime(market_report: str) -> str:
        text = (market_report or "").lower()
        high_signals = (
            "high volatility",
            "moderate-high volatility",
            "extreme volatility",
            "high atr",
            "risk-off",
            "choppy",
        )
        low_signals = (
            "low volatility",
            "compressed volatility",
            "range compression",
            "low atr",
        )
        if any(token in text for token in high_signals):
            return "high"
        if any(token in text for token in low_signals):
            return "low"
        return "moderate"

    def _extract_setup_score(report: str) -> int | None:
        match = re.search(r"Setup Score \(0-100\)\s*:\s*(\d{1,3})", report or "", flags=re.IGNORECASE)
        if not match:
            return None
        return max(0, min(100, int(match.group(1))))

    def _extract_trigger_active(report: str) -> bool | None:
        match = re.search(r"Trigger Active Now\s*:\s*(YES|NO)", report or "", flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).upper() == "YES"

    def _extract_final_proposal(report: str) -> str | None:
        match = re.search(
            r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*",
            report or "",
            flags=re.IGNORECASE,
        )
        return match.group(1).upper() if match else None

    def _force_hold(report: str, note: str) -> str:
        updated = re.sub(
            r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*",
            "FINAL TRANSACTION PROPOSAL: **HOLD**",
            report,
            flags=re.IGNORECASE,
        )
        if "FINAL TRANSACTION PROPOSAL:" not in updated:
            updated += "\n\nFINAL TRANSACTION PROPOSAL: **HOLD**"
        updated += f"\n\nExecution Gate Override: {note}"
        return updated

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        tf_context = build_timeframe_context(state)
        primary_tf = state.get("primary_tf", "1d")
        trading_style = state.get("trading_style", "swing")
        stop_loss_options = "2%, 3%, 5%"
        current_date = state["trade_date"]
        lookback_days = state.get("lookback_days", get_lookback_days(primary_tf))
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = (
            f"{_clip(market_research_report)}\n\n"
            f"{_clip(sentiment_report)}\n\n"
            f"{_clip(news_report)}\n\n"
            f"{_clip(fundamentals_report)}"
        )
        past_memories = memory.get_memories(
            curr_situation,
            n_matches=1,
            primary_tf=primary_tf,
        )

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        tools = [
            get_stock_data,
        ]

        system_message = f"""You are a trading agent analyzing market data to make investment decisions for a {trading_style} strategy on the {primary_tf} timeframe.

{instrument_context}
{tf_context}

Before giving your final entry recommendation, you MUST call get_stock_data to perform a live price alignment check.

Required live-price tool call rule:
- Use get_stock_data(symbol, start_date, end_date) with symbol={company_name}.
- end_date must be {current_date}.
- start_date should be close enough to validate current entry context (typically {lookback_days} days back).
- Do not finalize any setup until this tool call has returned.

You are given a proposed plan from the research team:
{investment_plan}

Setup evaluation rules:
- Evaluate BOTH a potential long setup AND a potential short setup independently based on the available data.
- Select the setup with the highest conviction given the current evidence (trend, momentum, structure, sentiment, fundamentals).
- If both setups are valid but conflicting, prefer the one aligned with the dominant trend on the {primary_tf} timeframe.
- If neither setup has sufficient conviction, output FLAT / NO TRADE with reasoning.
- A "setup" requires: a defined entry condition, a stop-loss level, and at least one profit target.

Execution scorecard and gate (MANDATORY):
- Compute and report a numeric Setup Score (0-100) using:
    - Trend Alignment (0-25)
    - Momentum Quality (0-20)
    - Structure Clarity (0-20)
    - Risk/Reward Quality (0-20)
    - Catalyst/Flow Support (0-15)
- Apply this gate to actionability:
    - Score >= 70: actionable setup allowed (LONG or SHORT if criteria met)
    - Score 55-69: conditional only, default HOLD unless trigger is already active
    - Score < 55: no-trade, output HOLD

After your analysis, include a structured trade parameters section:

**Trade Parameters ({primary_tf} timeframe):**
- **Direction**: LONG / SHORT / FLAT
- **Entry Condition**: [exact price level, breakout trigger, or pullback zone that activates the trade]
- **Entry Zone**: [price range for entry]
- **Stop-Loss %**: [MUST be exactly one of: {stop_loss_options}]
- **Stop-Loss Price**: [price derived from the selected Stop-Loss % and entry — above entry for SHORT, below for LONG]
- **Target 1**: [first profit target]
- **Target 2**: [second profit target, optional]
- **Invalidation**: [condition that cancels this setup entirely]
- **Expected Holding Period**: [duration appropriate for {trading_style} on {primary_tf}]
- **Position Sizing Note**: [risk guidance, e.g. \"risk max 1-2% of portfolio\"]

**Setup Scorecard:**
- Trend Alignment: [0-25]
- Momentum Quality: [0-20]
- Structure Clarity: [0-20]
- Risk/Reward Quality: [0-20]
- Catalyst/Flow Support: [0-15]
- Setup Score (0-100): [sum]

**Trigger Checklist:**
- Trigger Active Now: YES/NO
- Long Trigger Condition: [one line]
- Short Trigger Condition: [one line]
- Gate Result: ACTIONABLE / CONDITIONAL / NO-TRADE

If a valid opposing setup exists, briefly describe it and explain why it was ranked lower.

Hard constraints:
- You must choose Stop-Loss % from exactly this fixed set: {stop_loss_options}.
- Do not output any other stop-loss percentage.
- Stop-Loss Price must be on the correct side of entry (SHORT: above entry, LONG: below entry).
- Respect the setup-score gate before final action.
- BUY/SELL is only valid when Trigger Active Now = YES; otherwise final action must be HOLD.
- Always conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**

Apply lessons from past decisions to strengthen your analysis. Use this memory context briefly: {_clip(past_memory_str, 800)}"""
        system_message += get_autonomous_evidence_instruction()

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
                    " For your reference, the current date is {current_date}."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)

        result = invoke_with_optional_tools(prompt, llm, tools, state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content
            score = _extract_setup_score(report)
            trigger_active = _extract_trigger_active(report)
            current_proposal = _extract_final_proposal(report)

            regime = _detect_regime(market_research_report)
            threshold_by_regime = {
                "low": 62,
                "moderate": 70,
                "high": 78,
            }
            threshold = threshold_by_regime[regime]

            if current_proposal in {"BUY", "SELL"}:
                if score is None:
                    report = _force_hold(report, "Missing Setup Score (0-100), actionable trade not allowed.")
                elif score < threshold:
                    report = _force_hold(
                        report,
                        f"Score below regime threshold ({score}/100 < {threshold}) for {regime}-volatility regime.",
                    )
                elif trigger_active is not True:
                    report = _force_hold(
                        report,
                        "Trigger Active Now is not YES; entry confirmation missing.",
                    )

        return {
            "messages": [result],
            "trader_investment_plan": report,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
