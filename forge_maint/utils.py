"""Utility functions for signal loading, statistics and JSON IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import csv
import json
import math
import statistics

from .schemas import SignalSummary


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda x: getattr(x, "__dict__", str(x)))


def load_signal_csv(path: str | Path) -> List[float]:
    """Load one-column or multi-column CSV. Uses first numeric field in each row."""
    path = Path(path)
    values: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            for item in row:
                try:
                    values.append(float(item))
                    break
                except ValueError:
                    continue
    return values


def summarize_signal(values: Iterable[float]) -> SignalSummary:
    arr = list(values)
    if not arr:
        return SignalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    finite = [x for x in arr if math.isfinite(x)]
    missing_ratio = 1.0 - len(finite) / max(len(arr), 1)
    if not finite:
        return SignalSummary(len(arr), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    mean = statistics.fmean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    rms = math.sqrt(statistics.fmean([x * x for x in finite]))
    peak = max(abs(x) for x in finite)
    crest_factor = peak / rms if rms > 1e-12 else 0.0
    # Simple kurtosis-like descriptor without external dependencies.
    if std > 1e-12:
        kurtosis_like = statistics.fmean([((x - mean) / std) ** 4 for x in finite])
    else:
        kurtosis_like = 0.0
    return SignalSummary(
        n=len(arr),
        mean=round(mean, 6),
        std=round(std, 6),
        rms=round(rms, 6),
        peak=round(peak, 6),
        crest_factor=round(crest_factor, 6),
        kurtosis_like=round(kurtosis_like, 6),
        missing_ratio=round(missing_ratio, 6),
    )


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def keyword_terms(text: str) -> set[str]:
    separators = [",", ";", "，", "；", "。", ".", "、", ":", "：", "(", ")", "[", "]", "\n", "\t"]
    normalized = text.lower()
    for sep in separators:
        normalized = normalized.replace(sep, " ")
    return {t.strip() for t in normalized.split() if len(t.strip()) >= 2}
