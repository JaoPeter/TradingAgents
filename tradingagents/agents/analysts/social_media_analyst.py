from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_autonomous_evidence_instruction,
    invoke_with_optional_tools,
    get_language_instruction,
    get_lookback_days,
    get_news,
    get_sentiment_summary,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tf_context = build_timeframe_context(state)
        primary_tf = state.get("primary_tf", "1d")
        lookback = state.get("lookback_days", get_lookback_days(primary_tf))

        tools = [
            get_news,
            get_sentiment_summary,
        ]

        system_message = (
            f"{tf_context}\n\nYou are a social media and instrument-specific news researcher/analyst tasked with analyzing social media posts, recent news, and public sentiment for a specific instrument over the past {lookback} days. You will be given an instrument identifier; your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders on this instrument's current state after looking at social media and what people are saying, analyzing sentiment data, and looking at recent news. Use get_sentiment_summary(ticker, curr_date) first to pull a direct social sentiment snapshot, then use get_news(query, start_date, end_date) to gather instrument-specific news and discussion context over the past {lookback} days. Focus on sentiment signals relevant to the {primary_tf} timeframe and {state.get('trading_style', 'swing')} style. Try to use both sentiment and news tools before drafting the report whenever data is available. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
