import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_lookback_days,
    get_stock_data,
    invoke_with_optional_tools,
)


def create_trader(llm, memory):
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

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(
            curr_situation,
            n_matches=2,
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
- Do not finalize Entry/Stop-Loss/Targets until this tool call has returned.

You are given a proposed plan from the research team:
{investment_plan}

After your analysis, include a structured trade parameters section:

**Trade Parameters ({primary_tf} timeframe):**
- **Action**: BUY / SELL / HOLD
- **Entry Zone**: [price level or condition]
- **Stop-Loss %**: [MUST be exactly one of: {stop_loss_options}]
- **Stop-Loss Price**: [price derived from the selected Stop-Loss % and entry]
- **Target 1**: [first profit target]
- **Target 2**: [second profit target, optional]
- **Expected Holding Period**: [duration appropriate for {trading_style} on {primary_tf}]
- **Position Sizing Note**: [risk guidance, e.g. \"risk max 1-2% of portfolio\"]

Hard constraints:
- You must choose Stop-Loss % from exactly this fixed set: {stop_loss_options}.
- Do not output any other stop-loss percentage.
- Always conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**

Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}"""

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

        return {
            "messages": [result],
            "trader_investment_plan": report,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
