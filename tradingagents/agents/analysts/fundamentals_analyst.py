from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_timeframe_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    invoke_with_optional_tools,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config

# Common crypto base currencies — extend as needed
_CRYPTO_SUFFIXES = {"-USD", "-USDT", "-BTC", "-ETH", "-EUR", "-GBP"}
_CRYPTO_STABLE_TICKERS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "LINK"}


def _is_crypto(ticker: str) -> bool:
    """Heuristic: detect crypto tickers (e.g. BTC-USD, ETH-USDT, SOL-USD)."""
    upper = ticker.upper()
    if any(upper.endswith(s) for s in _CRYPTO_SUFFIXES):
        return True
    if upper in _CRYPTO_STABLE_TICKERS:
        return True
    return False


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        tf_context = build_timeframe_context(state)
        crypto = _is_crypto(ticker)

        if crypto:
            # For crypto: balance-sheet tools are irrelevant; use fundamentals + insider
            tools = [
                get_fundamentals,
            ]
            asset_type_note = (
                "This instrument is a **cryptocurrency or digital asset**. "
                "Do NOT attempt to retrieve corporate balance sheets, cash flow statements, or income statements — they are not applicable. "
                "Focus instead on: on-chain metrics, market cap, circulating supply, tokenomics, protocol fundamentals, "
                "developer activity, network usage, and any available DeFi/TVL data. "
                "Use `get_fundamentals` to retrieve what data is available."
            )
        else:
            tools = [
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
            ]
            asset_type_note = (
                "Use the available tools: `get_fundamentals` for comprehensive analysis, "
                "`get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            )

        system_message = (
            f"{tf_context}\n\nYou are a researcher tasked with analyzing fundamental information about an instrument. {asset_type_note} "
            "Please write a comprehensive report of the instrument's fundamental information to inform traders. "
            "Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence."
            " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            " Do not ask the user what to do next. Do not ask follow-up questions. You must make decisions from available evidence and deliver a clear directional view, key risks, and concrete next action."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
