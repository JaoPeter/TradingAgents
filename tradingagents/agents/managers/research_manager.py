
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_autonomous_evidence_instruction,
)


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        tf_context = build_timeframe_context(state)
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        primary_tf = state.get("primary_tf", "1d")
        trading_style = state.get("trading_style", "swing")

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(
            curr_situation,
            n_matches=2,
            primary_tf=primary_tf,
        )

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the research manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified.

    {tf_context}

    Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable for a {trading_style} trader on the {primary_tf} timeframe. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

    Additionally, develop a detailed investment plan for the trader. This should include:
    1. Your Recommendation: A decisive stance supported by the most convincing arguments.
    2. Rationale: An explanation of why these arguments lead to your conclusion.
    3. Strategic Actions: Concrete steps for implementing the recommendation, including entry timing relevant to the {primary_tf} timeframe.
    4. Holding Period: Expected holding duration based on the {trading_style} style and {primary_tf} timeframe.
    5. Key Risk Factors: Main risks to monitor that could invalidate the thesis.

    Take into account your past mistakes on similar situations. Use these insights to refine your decision-making. Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

{instrument_context}

Here is the debate:
Debate History:
{history}"""
        prompt += get_autonomous_evidence_instruction()
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
