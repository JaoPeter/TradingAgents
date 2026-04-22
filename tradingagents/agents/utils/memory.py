"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.
"""

from typing import List, Tuple, Dict, Any
import re
import json
from pathlib import Path
from datetime import datetime, timezone
import importlib

try:
    _bm25_module = importlib.import_module("rank_bm25")
    BM25Okapi = getattr(_bm25_module, "BM25Okapi", None)
except Exception:  # pragma: no cover - fallback path for environments missing rank_bm25
    BM25Okapi = None


# Conservative blending weights: designed for trend recognition without
# requiring a correctness feedback loop for "right" vs "wrong" entries.
_WEIGHT_LEXICAL = 0.55
_WEIGHT_TREND = 0.30
_WEIGHT_RECENCY = 0.15

_NEWS_QUERY_KEYWORDS = {
    "news", "headline", "macro", "global", "sentiment", "social", "tweet", "x", "twitter",
    "insider", "announcement", "regulation", "policy", "fed", "cpi", "ppi", "jobs", "fomc",
}

_NEWS_WINDOW_DAYS_BY_TF = {
    # 15m: use only very recent context (max daily)
    "15m": 1,
    "1h": 14,
    # 4h: max weekly context
    "4h": 7,
    # 1d: max 30-day context
    "1d": 30,
    "1w": 90,
}


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25."""

    def __init__(self, name: str, config: dict = None):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict
        """
        self.name = name
        self.config = config or {}
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.features: List[Dict[str, Any]] = []
        self.timestamps: List[str] = []
        self.bm25 = None

        self.memory_path = self._resolve_memory_path()
        self._load_from_disk()

    def _resolve_memory_path(self) -> Path:
        """Resolve persistent memory path under data cache directory."""
        cache_dir = self.config.get("data_cache_dir")
        if not cache_dir:
            cache_dir = str(Path.home() / ".tradingagents" / "cache")
        path = Path(cache_dir) / "memory" / f"{self.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract trend/regime features from text (crypto-friendly heuristics)."""
        lowered = text.lower()
        tokens = set(self._tokenize(text))

        bullish_kw = {"bull", "bullish", "uptrend", "breakout", "higher", "higher_high", "higher_low", "momentum_up"}
        bearish_kw = {"bear", "bearish", "downtrend", "breakdown", "lower", "lower_high", "lower_low", "momentum_down"}
        sideway_kw = {"range", "sideways", "consolidation", "chop", "mean_reversion"}
        high_vol_kw = {"volatile", "volatility", "spike", "whipsaw", "high_vol"}
        low_vol_kw = {"compression", "calm", "low_vol", "squeeze"}
        risk_on_kw = {"risk_on", "altseason", "beta", "speculative"}
        risk_off_kw = {"risk_off", "flight", "defensive", "deleveraging"}

        regime = "unknown"
        if tokens & bullish_kw:
            regime = "bullish"
        elif tokens & bearish_kw:
            regime = "bearish"
        elif tokens & sideway_kw:
            regime = "sideways"

        volatility = "unknown"
        if tokens & high_vol_kw:
            volatility = "high"
        elif tokens & low_vol_kw:
            volatility = "low"

        risk_regime = "unknown"
        if tokens & risk_on_kw:
            risk_regime = "risk_on"
        elif tokens & risk_off_kw:
            risk_regime = "risk_off"

        # Track timeframe mentions for pattern continuity
        timeframe_hits = [tf for tf in ["15m", "1h", "4h", "1d", "1w"] if tf in lowered]

        # Crypto context hints
        crypto_hits = [
            c for c in ["btc", "eth", "sol", "xrp", "ada", "bnb", "altcoin", "bitcoin", "ethereum", "defi", "onchain", "tvl", "funding", "open_interest"]
            if c in lowered
        ]

        # Build compact trend tags used for symbolic overlap matching
        trend_tags = set()
        trend_tags.add(f"regime:{regime}")
        trend_tags.add(f"vol:{volatility}")
        trend_tags.add(f"risk:{risk_regime}")
        for tf in timeframe_hits:
            trend_tags.add(f"tf:{tf}")
        for ch in crypto_hits:
            trend_tags.add(f"cx:{ch}")

        return {
            "regime": regime,
            "volatility": volatility,
            "risk_regime": risk_regime,
            "timeframes": timeframe_hits,
            "crypto_context": crypto_hits,
            "trend_tags": sorted(trend_tags),
        }

    def _rebuild_index(self):
        """Rebuild the BM25 index after adding documents."""
        if self.documents and BM25Okapi is not None:
            # Include symbolic feature tags in the index to improve trend/regime matching.
            tokenized_docs = []
            for i, doc in enumerate(self.documents):
                tags = self.features[i].get("trend_tags", []) if i < len(self.features) else []
                merged = f"{doc} {' '.join(tags)}"
                tokenized_docs.append(self._tokenize(merged))
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def _fallback_lexical_score(self, query_tokens: List[str], doc: str, tags: List[str]) -> float:
        """Compute simple token-overlap lexical score when BM25 is unavailable."""
        doc_tokens = set(self._tokenize(f"{doc} {' '.join(tags)}"))
        if not doc_tokens or not query_tokens:
            return 0.0
        query_set = set(query_tokens)
        return len(query_set & doc_tokens) / max(len(query_set), 1)

    def _load_from_disk(self):
        """Load persisted memory entries (if present)."""
        if not self.memory_path.exists():
            return

        docs: List[str] = []
        recs: List[str] = []
        feats: List[Dict[str, Any]] = []
        times: List[str] = []

        try:
            for line in self.memory_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                situation = item.get("situation", "")
                recommendation = item.get("recommendation", "")
                if not situation or not recommendation:
                    continue
                docs.append(situation)
                recs.append(recommendation)
                feats.append(item.get("features") or self._extract_features(situation))
                times.append(item.get("timestamp") or datetime.now(timezone.utc).isoformat())
        except Exception:
            # Corrupted files should not crash runtime.
            docs, recs, feats, times = [], [], [], []

        self.documents = docs
        self.recommendations = recs
        self.features = feats
        self.timestamps = times
        self._rebuild_index()

    def _append_to_disk(self, situation: str, recommendation: str, features: Dict[str, Any], timestamp: str):
        """Append one memory record to persistent storage."""
        payload = {
            "situation": situation,
            "recommendation": recommendation,
            "features": features,
            "timestamp": timestamp,
        }
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _feature_overlap_score(self, query_features: Dict[str, Any], doc_features: Dict[str, Any]) -> float:
        """Compute overlap score for trend/regime tags using Jaccard similarity."""
        q_tags = set(query_features.get("trend_tags", []))
        d_tags = set(doc_features.get("trend_tags", []))
        if not q_tags or not d_tags:
            return 0.0
        union = q_tags | d_tags
        inter = q_tags & d_tags
        if not union:
            return 0.0
        return len(inter) / len(union)

    def _recency_score(self, idx: int) -> float:
        """Return recency score in [0, 1] where newer entries are slightly favored."""
        n = len(self.timestamps)
        if n <= 1:
            return 1.0
        # Index-based fallback for deterministic behavior even if timestamps are irregular.
        return idx / (n - 1)

    def _resolve_query_timeframe(self, query_features: Dict[str, Any], primary_tf: str = None) -> str:
        """Resolve query timeframe from explicit parameter or extracted features."""
        if primary_tf:
            return str(primary_tf).strip().lower()
        tf_list = query_features.get("timeframes", [])
        if tf_list:
            return str(tf_list[0]).strip().lower()
        return ""

    def _candidate_indices_for_timeframe(self, query_tf: str) -> List[int]:
        """Return candidate indices filtered by timeframe (with safe fallback)."""
        all_indices = list(range(len(self.documents)))
        if not query_tf:
            return all_indices

        # Prefer exact timeframe match if available
        tf_matches = []
        for idx, feats in enumerate(self.features):
            tf_values = [str(x).strip().lower() for x in feats.get("timeframes", [])]
            if query_tf in tf_values:
                tf_matches.append(idx)

        # If there are exact matches, enforce strict timeframe retrieval.
        if tf_matches:
            return tf_matches

        # No exact timeframe history available yet -> fallback to full history.
        return all_indices

    def _is_news_or_sentiment_query(self, current_situation: str, query_features: Dict[str, Any]) -> bool:
        """Detect news/sentiment-heavy queries where strict timeframe matching is too rigid."""
        text = current_situation.lower()
        tokens = set(self._tokenize(text))
        if tokens & _NEWS_QUERY_KEYWORDS:
            return True

        # Feature-level hints from extracted crypto/news context.
        context_hits = set(query_features.get("crypto_context", []))
        if {"funding", "open_interest"} & context_hits:
            return True

        return False

    def _filter_indices_by_news_window(self, indices: List[int], query_tf: str) -> List[int]:
        """Filter candidates by timeframe-specific recency windows for news/sentiment contexts."""
        if not indices:
            return indices

        days = _NEWS_WINDOW_DAYS_BY_TF.get(query_tf, 30)
        now = datetime.now(timezone.utc)
        kept: List[int] = []

        for idx in indices:
            ts = self.timestamps[idx] if idx < len(self.timestamps) else ""
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            age_days = (now - dt).days
            if age_days <= days:
                kept.append(idx)

        # If no entries survive the window, fallback to original indices to avoid empty retrieval.
        return kept if kept else indices

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)
            features = self._extract_features(situation)
            timestamp = datetime.now(timezone.utc).isoformat()
            self.features.append(features)
            self.timestamps.append(timestamp)
            self._append_to_disk(situation, recommendation, features, timestamp)

        # Rebuild BM25 index with new documents
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1, primary_tf: str = None) -> List[dict]:
        """Find matching recommendations using BM25 similarity.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return
            primary_tf: Preferred primary timeframe filter (e.g., 15m, 1h, 4h, 1d)

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        if not self.documents:
            return []

        query_features = self._extract_features(current_situation)
        query_tf = self._resolve_query_timeframe(query_features, primary_tf)
        if query_tf:
            query_features["timeframes"] = [query_tf]
            trend_tags = set(query_features.get("trend_tags", []))
            trend_tags.add(f"tf:{query_tf}")
            query_features["trend_tags"] = sorted(trend_tags)

        is_news_query = self._is_news_or_sentiment_query(current_situation, query_features)
        if is_news_query:
            # For news/sentiment, recency window beats strict tf matching.
            candidate_indices = self._filter_indices_by_news_window(
                list(range(len(self.documents))),
                query_tf,
            )
        else:
            candidate_indices = self._candidate_indices_for_timeframe(query_tf)
        if not candidate_indices:
            return []

        query_text = f"{current_situation} {' '.join(query_features.get('trend_tags', []))}"

        # Tokenize query
        query_tokens = self._tokenize(query_text)

        # Get lexical scores (BM25 when available; overlap fallback otherwise)
        if self.bm25 is not None:
            all_scores = self.bm25.get_scores(query_tokens)
            scores = [all_scores[i] for i in candidate_indices]
            max_score = max(scores) if scores else 0.0
            max_lex = float(max_score) if max_score > 0 else 1.0
        else:
            scores = []
            for i in candidate_indices:
                doc = self.documents[i]
                tags = self.features[i].get("trend_tags", []) if i < len(self.features) else []
                scores.append(self._fallback_lexical_score(query_tokens, doc, tags))
            max_lex = max(scores) if scores else 1.0

        blended_scores: List[Tuple[int, float]] = []
        for local_idx, idx in enumerate(candidate_indices):
            lexical = _safe_float(scores[local_idx], 0.0) / max_lex if max_lex > 0 else 0.0
            trend_overlap = self._feature_overlap_score(query_features, self.features[idx] if idx < len(self.features) else {})
            recency = self._recency_score(idx)

            # Deliberately conservative blending: does not depend on PnL correctness labels.
            final_score = (
                _WEIGHT_LEXICAL * lexical
                + _WEIGHT_TREND * trend_overlap
                + _WEIGHT_RECENCY * recency
            )
            blended_scores.append((idx, final_score))

        # Get top-n indices sorted by blended score (descending)
        top_indices = [i for i, _ in sorted(blended_scores, key=lambda x: x[1], reverse=True)[:n_matches]]

        # Build results
        results = []
        max_final = max((s for _, s in blended_scores), default=1.0)
        if max_final <= 0:
            max_final = 1.0

        for idx in top_indices:
            normalized_score = next((s for i, s in blended_scores if i == idx), 0.0) / max_final
            results.append({
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": normalized_score,
                "matched_features": self.features[idx] if idx < len(self.features) else {},
            })

        return results

    def clear(self):
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.features = []
        self.timestamps = []
        self.bm25 = None
        if self.memory_path.exists():
            self.memory_path.unlink()


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test_memory")

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
