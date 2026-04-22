import unittest
from unittest.mock import Mock, patch

from tradingagents.dataflows import x_sentiment


def _mock_x_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class XSentimentTests(unittest.TestCase):
    def test_get_sentiment_summary_uses_query_and_formats_ranked_posts(self):
        payload = {
            "data": [
                {
                    "id": "1",
                    "author_id": "42",
                    "text": "$BTC looks bullish after reclaiming the key support zone with strong buying momentum, volume is picking up nicely.",
                    "created_at": "2026-04-22T10:00:00Z",
                    "public_metrics": {
                        "like_count": 12,
                        "retweet_count": 4,
                        "reply_count": 2,
                        "quote_count": 1,
                    },
                },
                {
                    "id": "2",
                    "author_id": "43",
                    "text": "Bitcoin setup looks bearish near major resistance, I am watching closely for a potential breakdown of this level with increasing volume.",
                    "created_at": "2026-04-22T11:00:00Z",
                    "public_metrics": {
                        "like_count": 5,
                        "retweet_count": 1,
                        "reply_count": 0,
                        "quote_count": 0,
                    },
                },
            ],
            "includes": {
                "users": [
                    {
                        "id": "42",
                        "username": "bull_trader",
                        "verified": True,
                        "public_metrics": {"followers_count": 12000},
                    },
                    {
                        "id": "43",
                        "username": "bear_trader",
                        "verified": False,
                        "public_metrics": {"followers_count": 2500},
                    },
                ]
            },
        }

        with patch.dict("os.environ", {"X_API_KEY": "bearer-token"}, clear=False), patch(
            "tradingagents.dataflows.x_sentiment.requests.get",
            return_value=_mock_x_response(payload),
        ), patch(
            "tradingagents.dataflows.x_sentiment._utc_now",
            return_value=x_sentiment.datetime(2026, 4, 22, 12, 0, tzinfo=x_sentiment.timezone.utc),
        ):
            result = x_sentiment.get_sentiment_summary("BTC-USD", "2026-04-22")

        self.assertIn("Search criteria:", result)
        self.assertIn("$BTC", result)
        self.assertIn("Mention Count (considered): 2", result)
        self.assertIn("@bull_trader", result)

    def test_get_news_rejects_windows_older_than_recent_search_retention(self):
        with patch.dict("os.environ", {"X_API_KEY": "bearer-token"}, clear=False), patch(
            "tradingagents.dataflows.x_sentiment._utc_now",
            return_value=x_sentiment.datetime(2026, 4, 22, 12, 0, tzinfo=x_sentiment.timezone.utc),
        ):
            result = x_sentiment.get_news("BTC", "2026-04-01", "2026-04-02")

        self.assertIn("outside X recent-search retention", result)


if __name__ == "__main__":
    unittest.main()