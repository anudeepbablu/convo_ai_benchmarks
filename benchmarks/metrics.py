"""Shared metrics utilities: percentiles, RTF, WER, concurrency aggregation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from jiwer import wer as compute_wer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


@dataclass
class LatencyStats:
    """Latency statistics for a set of request durations (in seconds)."""
    count: int
    success_count: int
    error_count: int
    error_rate: float
    mean_sec: float
    median_sec: float
    p90_sec: float
    p95_sec: float
    p99_sec: float
    min_sec: float
    max_sec: float
    std_sec: float
    throughput_rps: float
    total_duration_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConcurrencyResult:
    """Results for a single concurrency level."""
    concurrency: int
    latency: LatencyStats
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "concurrency": self.concurrency,
            **self.latency.to_dict(),
            **self.extra,
        }
        return d


def percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of data (0-100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def compute_latency_stats(
    latencies_sec: list[float],
    errors: int,
    total_duration_sec: float,
) -> LatencyStats:
    """Compute full latency statistics from a list of per-request latencies."""
    success = len(latencies_sec)
    total = success + errors
    valid = [l for l in latencies_sec if l > 0]
    if not valid:
        return LatencyStats(
            count=total, success_count=success, error_count=errors,
            error_rate=errors / max(total, 1),
            mean_sec=0, median_sec=0, p90_sec=0, p95_sec=0, p99_sec=0,
            min_sec=0, max_sec=0, std_sec=0,
            throughput_rps=success / max(total_duration_sec, 1e-9),
            total_duration_sec=total_duration_sec,
        )
    return LatencyStats(
        count=total,
        success_count=success,
        error_count=errors,
        error_rate=errors / max(total, 1),
        mean_sec=round(statistics.mean(valid), 4),
        median_sec=round(statistics.median(valid), 4),
        p90_sec=round(percentile(valid, 90), 4),
        p95_sec=round(percentile(valid, 95), 4),
        p99_sec=round(percentile(valid, 99), 4),
        min_sec=round(min(valid), 4),
        max_sec=round(max(valid), 4),
        std_sec=round(statistics.stdev(valid) if len(valid) > 1 else 0.0, 4),
        throughput_rps=round(success / max(total_duration_sec, 1e-9), 3),
        total_duration_sec=round(total_duration_sec, 3),
    )


def compute_rtf(audio_duration_sec: float, processing_time_sec: float) -> float:
    """Real-Time Factor: processing_time / audio_duration. <1 means faster than real-time."""
    if audio_duration_sec <= 0:
        return 0.0
    return round(processing_time_sec / audio_duration_sec, 4)


def compute_wer_score(references: list[str], hypotheses: list[str]) -> Optional[float]:
    """Compute Word Error Rate using jiwer. Returns None if jiwer not available."""
    if not _JIWER_AVAILABLE:
        return None
    if not references or not hypotheses:
        return None
    try:
        score = compute_wer(references, hypotheses)
        return round(float(score), 4)
    except Exception:
        return None


def aggregate_concurrency_results(results: list[ConcurrencyResult]) -> list[dict]:
    """Convert a list of ConcurrencyResult to a list of dicts for export."""
    return [r.to_dict() for r in results]
