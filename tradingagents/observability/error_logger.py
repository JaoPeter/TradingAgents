from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_api_error(error_log_path: str | Path, source: str, error: Exception | str, context: dict[str, Any] | None = None) -> None:
    """Append API/LLM/runtime errors as JSONL for post-mortem debugging."""
    path = Path(error_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "error": str(error),
        "context": context or {},
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
