from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_autonomous_evidence_instruction,
    get_language_instruction,
)


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])
        tf_context = build_timeframe_context(state)
        primary_tf = state.get("primary_tf", "1d")
        trading_style = state.get("trading_style", "swing")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(
            curr_situation,
            n_matches=2,
            primary_tf=primary_tf,
        )

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

    {tf_context}

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**
- Trading Style: **{trading_style}** on **{primary_tf}** timeframe

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Executive Summary**: A concise action plan covering entry strategy, position sizing (calibrated for {trading_style} risk tolerance), key risk levels, and time horizon (appropriate for {primary_tf}).
3. **Investment Thesis**: Detailed reasoning anchored in the analysts' debate and past reflections.
4. **Position Sizing Guidance**: Specific guidance on position size relative to portfolio (e.g., max 1-2% risk per trade for {trading_style}).

---

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""
        prompt += get_autonomous_evidence_instruction()

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
