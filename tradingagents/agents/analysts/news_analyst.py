from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_autonomous_evidence_instruction,
    get_global_news,
    invoke_with_optional_tools,
    get_language_instruction,
    get_lookback_days,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tf_context = build_timeframe_context(state)
        primary_tf = state.get("primary_tf", "1d")
        lookback = state.get("lookback_days", get_lookback_days(primary_tf))

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            f"{tf_context}\n\nAnalyze instrument and macro news over the last {lookback} days for a {state.get('trading_style', 'swing')} trader on {primary_tf}."
            + """ Use get_news for instrument-specific coverage and get_global_news for macro context."""
            + """ Output format (max 200 words): Bias, 3 evidence bullets, 2 risks, 1 actionable implication."""
            + """ Add one markdown table with exactly 4 rows: Theme | Evidence | Impact | Action."""
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
            "news_report": report,
        }

    return news_analyst_node
