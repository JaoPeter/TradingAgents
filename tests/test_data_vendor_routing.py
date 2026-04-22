import copy
import unittest
from unittest.mock import patch

import tradingagents.default_config as default_config
from tradingagents.dataflows import config as dataflow_config
from tradingagents.dataflows import interface


class DataVendorRoutingTests(unittest.TestCase):
    def test_default_config_prefers_crypto_native_price_feeds(self):
        self.assertEqual(
            default_config.DEFAULT_CONFIG["tool_vendors"]["get_stock_data"],
            "binance,coinmarketcap,yfinance,alpha_vantage",
        )

    def test_default_config_prefers_specialized_news_and_fundamental_feeds(self):
        self.assertEqual(
            default_config.DEFAULT_CONFIG["tool_vendors"]["get_fundamentals"],
            "coinmarketcap,defilama,yfinance,alpha_vantage",
        )
        self.assertEqual(
            default_config.DEFAULT_CONFIG["tool_vendors"]["get_news"],
            "cryptocompare,x_sentiment,crypto_rss,yfinance,alpha_vantage",
        )
        self.assertEqual(
            default_config.DEFAULT_CONFIG["tool_vendors"]["get_sentiment_summary"],
            "x_sentiment",
        )

    def test_route_to_vendor_falls_back_after_vendor_error_string(self):
        with patch.object(dataflow_config, "_config", copy.deepcopy(default_config.DEFAULT_CONFIG)):
            dataflow_config.set_config(
                {
                    "tool_vendors": {
                        "get_stock_data": "binance,coinmarketcap,yfinance",
                    }
                }
            )

            with patch.dict(
                interface.VENDOR_METHODS["get_stock_data"],
                {
                    "binance": lambda *args, **kwargs: "Error fetching Binance klines for BTCUSDT: boom",
                    "coinmarketcap": lambda *args, **kwargs: "Date,Open,High,Low,Close,Volume\n2026-04-21,1,2,0.5,1.5,100\n",
                    "yfinance": lambda *args, **kwargs: self.fail("yfinance should not be used after a successful CoinMarketCap fallback"),
                },
                clear=True,
            ):
                result = interface.route_to_vendor("get_stock_data", "BTC", "2026-04-20", "2026-04-21")

        self.assertIn("2026-04-21", result)

    def test_route_to_vendor_uses_specialized_sentiment_method(self):
        with patch.object(dataflow_config, "_config", copy.deepcopy(default_config.DEFAULT_CONFIG)):
            with patch.dict(
                interface.VENDOR_METHODS["get_sentiment_summary"],
                {
                    "x_sentiment": lambda *args, **kwargs: "# X (Twitter) Sentiment for BTC\nSentiment: bullish",
                },
                clear=True,
            ):
                result = interface.route_to_vendor("get_sentiment_summary", "BTC", "2026-04-21")

        self.assertIn("Sentiment: bullish", result)


if __name__ == "__main__":
    unittest.main()