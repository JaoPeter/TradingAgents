import unittest
from unittest.mock import patch

from tradingagents.agents.utils.news_data_tools import get_sentiment_summary


class NewsDataToolsTests(unittest.TestCase):
    def test_get_sentiment_summary_handles_vendor_runtime_error(self):
        with patch(
            "tradingagents.agents.utils.news_data_tools.route_to_vendor",
            side_effect=RuntimeError("No available vendor for 'get_sentiment_summary'. Last failure: x_sentiment auth failed"),
        ):
            result = get_sentiment_summary.invoke(
                {"ticker": "BTC", "curr_date": "2026-04-22"}
            )

        self.assertIn("Social sentiment currently unavailable", result)
        self.assertIn("Proceeding without X sentiment data", result)


if __name__ == "__main__":
    unittest.main()
