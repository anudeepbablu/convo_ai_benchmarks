"""TTS benchmark against Nvidia Riva NIM using the official nvidia-riva-client library.

Supports two modes matching NVIDIA's recommended methodology:
  - streaming: synthesize_online with chunked audio responses (matches riva_tts_perf_client --online=true)
  - offline:   synthesize with full audio response in one shot

References:
  https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-performance.html
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from benchmarks.metrics import (
    ConcurrencyResult,
    compute_latency_stats,
    percentile,
)
from benchmarks.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)

try:
    import riva.client as riva_client
    from riva.client import (
        Auth,
        AudioEncoding,
        SpeechSynthesisService,
    )
    _RIVA_AVAILABLE = True
except ImportError:
    _RIVA_AVAILABLE = False
    logger.warning("nvidia-riva-client not installed; TTS benchmark unavailable.")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"
PROMPTS_PATH = Path(__file__).parent.parent / "data" / "prompts" / "tts_prompts.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompts(tier: str = "all") -> list[str]:
    """Load TTS prompts. tier: 'short', 'medium', 'long', or 'all'."""
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    if tier == "all":
        prompts = []
        for t in ("short", "medium", "long"):
            prompts.extend(data.get(t, []))
        return prompts
    return data.get(tier, [])


def _audio_duration_from_bytes(audio_bytes: int, sample_rate_hz: int, bytes_per_sample: int = 2) -> float:
    """Compute audio duration from raw PCM byte count."""
    if sample_rate_hz <= 0:
        return 0.0
    return (audio_bytes / bytes_per_sample) / sample_rate_hz


# ── Streaming mode (matches riva_tts_perf_client --online=true) ──────────────

@dataclass
class StreamingTtsResult:
    prompt: str
    prompt_chars: int
    total_latency_sec: float
    time_to_first_token_sec: float
    time_to_first_audio_sec: float
    generation_after_first_sec: float
    mean_inter_chunk_ms: float
    num_chunks: int
    output_audio_duration_sec: float
    rtfx: float
    success: bool
    error: str = ""


def _synthesize_streaming(
    prompt: str,
    auth: Auth,
    voice_name: str,
    sample_rate_hz: int,
    language_code: str,
) -> StreamingTtsResult:
    """Streaming synthesis matching riva_tts_perf_client --online=true behavior."""
    if not _RIVA_AVAILABLE:
        return StreamingTtsResult(
            prompt=prompt, prompt_chars=len(prompt),
            total_latency_sec=0, time_to_first_token_sec=0,
            time_to_first_audio_sec=0,
            generation_after_first_sec=0, mean_inter_chunk_ms=0,
            num_chunks=0, output_audio_duration_sec=0, rtfx=0,
            success=False, error="nvidia-riva-client not installed",
        )
    try:
        tts = SpeechSynthesisService(auth)

        t_start = time.perf_counter()
        first_response_time: Optional[float] = None
        first_audio_time: Optional[float] = None
        chunk_times: list[float] = []
        audio_bytes_total = 0

        responses = tts.synthesize_online(
            text=prompt,
            voice_name=voice_name,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
        )

        for response in responses:
            now = time.perf_counter()
            if first_response_time is None:
                first_response_time = now - t_start
            if hasattr(response, "audio") and response.audio:
                chunk_times.append(now)
                audio_bytes_total += len(response.audio)
                if first_audio_time is None:
                    first_audio_time = now - t_start

        t_end = time.perf_counter()
        total_latency = t_end - t_start
        ttft = first_response_time if first_response_time is not None else total_latency
        ttfa = first_audio_time if first_audio_time is not None else total_latency
        generation_after_first = total_latency - ttfa

        # Inter-chunk latency
        inter_chunk_deltas = []
        for i in range(1, len(chunk_times)):
            inter_chunk_deltas.append((chunk_times[i] - chunk_times[i - 1]) * 1000)
        mean_inter_chunk = round(sum(inter_chunk_deltas) / len(inter_chunk_deltas), 2) if inter_chunk_deltas else 0.0

        output_duration = _audio_duration_from_bytes(audio_bytes_total, sample_rate_hz)

        # RTFX = audio_duration / compute_time (NVIDIA convention, higher = better)
        rtfx = round(output_duration / total_latency, 2) if total_latency > 0 else 0.0

        return StreamingTtsResult(
            prompt=prompt,
            prompt_chars=len(prompt),
            total_latency_sec=total_latency,
            time_to_first_token_sec=ttft,
            time_to_first_audio_sec=ttfa,
            generation_after_first_sec=generation_after_first,
            mean_inter_chunk_ms=mean_inter_chunk,
            num_chunks=len(chunk_times),
            output_audio_duration_sec=output_duration,
            rtfx=rtfx,
            success=True,
        )
    except Exception as e:
        logger.debug(f"TTS streaming error: {e}")
        return StreamingTtsResult(
            prompt=prompt, prompt_chars=len(prompt),
            total_latency_sec=0, time_to_first_token_sec=0,
            time_to_first_audio_sec=0,
            generation_after_first_sec=0, mean_inter_chunk_ms=0,
            num_chunks=0, output_audio_duration_sec=0, rtfx=0,
            success=False, error=str(e),
        )


# ── Offline mode (Synthesize unary RPC) ─────────────────────────────────────

@dataclass
class OfflineTtsResult:
    prompt: str
    prompt_chars: int
    latency_sec: float
    output_audio_duration_sec: float
    rtfx: float
    success: bool
    error: str = ""


def _synthesize_offline(
    prompt: str,
    auth: Auth,
    voice_name: str,
    sample_rate_hz: int,
    language_code: str,
) -> OfflineTtsResult:
    """Non-streaming synthesis — full audio returned in one response."""
    if not _RIVA_AVAILABLE:
        return OfflineTtsResult(
            prompt=prompt, prompt_chars=len(prompt),
            latency_sec=0, output_audio_duration_sec=0, rtfx=0,
            success=False, error="nvidia-riva-client not installed",
        )
    try:
        tts = SpeechSynthesisService(auth)

        t0 = time.perf_counter()
        response = tts.synthesize(
            text=prompt,
            voice_name=voice_name,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
        )
        latency = time.perf_counter() - t0

        audio_bytes = len(response.audio) if hasattr(response, "audio") and response.audio else 0
        output_duration = _audio_duration_from_bytes(audio_bytes, sample_rate_hz)
        rtfx = round(output_duration / latency, 2) if latency > 0 else 0.0

        return OfflineTtsResult(
            prompt=prompt,
            prompt_chars=len(prompt),
            latency_sec=latency,
            output_audio_duration_sec=output_duration,
            rtfx=rtfx,
            success=True,
        )
    except Exception as e:
        logger.debug(f"TTS offline error: {e}")
        return OfflineTtsResult(
            prompt=prompt, prompt_chars=len(prompt),
            latency_sec=0, output_audio_duration_sec=0, rtfx=0,
            success=False, error=str(e),
        )


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_tts_benchmark(
    concurrency_levels: list[int],
    requests_per_level: int = 50,
    prompt_tier: str = "all",
    host: Optional[str] = None,
    port: Optional[int] = None,
    use_ssl: Optional[bool] = None,
    voice_name: str = "English-US.Female-1",
    sample_rate_hz: int = 22050,
    language_code: str = "en-US",
    mode: str = "streaming",
    gpu_index: int = 0,
    gpu_monitor_interval: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run TTS benchmark across multiple concurrency levels.

    Args:
        mode: "streaming" (chunked, matches riva_tts_perf_client --online=true) or "offline" (unary).

    Returns dict with:
      - results: list of per-concurrency result dicts
      - gpu_summary: GpuSummary dict or None
      - config_used: dict
    """
    cfg = load_config()
    tts_cfg = cfg.get("tts", {})
    host = host or tts_cfg.get("host", "localhost")
    port = port or tts_cfg.get("port", 50052)
    use_ssl = use_ssl if use_ssl is not None else tts_cfg.get("use_ssl", False)

    prompts = load_prompts(prompt_tier)
    if not prompts:
        raise ValueError(f"No prompts found for tier='{prompt_tier}'.")

    # Reuse a single Auth across all requests
    auth = Auth(uri=f"{host}:{port}", use_ssl=use_ssl)

    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _log(f"TTS benchmark mode={mode} voice={voice_name} sample_rate={sample_rate_hz}")

    monitor = GpuMonitor(gpu_index=gpu_index, interval=gpu_monitor_interval)
    monitor.start()

    concurrency_results: list[ConcurrencyResult] = []

    for concurrency in concurrency_levels:
        _log(f"TTS {mode}: concurrency={concurrency}, requests={requests_per_level}")
        level_prompts = random.choices(prompts, k=requests_per_level)

        level_gpu_t0 = time.time()
        t_start = time.perf_counter()

        if mode == "streaming":
            results_list: list[StreamingTtsResult] = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _synthesize_streaming,
                        p, auth, voice_name, sample_rate_hz, language_code,
                    ): p
                    for p in level_prompts
                }
                for future in as_completed(futures):
                    results_list.append(future.result())

            total_duration = time.perf_counter() - t_start
            successes = [r for r in results_list if r.success]
            errors = len(results_list) - len(successes)

            latencies = [r.total_latency_sec for r in successes]
            latency_stats = compute_latency_stats(latencies, errors, total_duration)

            # RTFX aggregate: total audio generated / total wall time
            total_audio_sec = sum(r.output_audio_duration_sec for r in successes)
            rtfx = round(total_audio_sec / total_duration, 2) if total_duration > 0 else 0.0

            # Time to first token (first gRPC response)
            ttfts = [r.time_to_first_token_sec for r in successes]
            mean_ttft = round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0
            p99_ttft = round(percentile(ttfts, 99), 4) if ttfts else 0.0

            # Time to first audio (first response with audio bytes)
            ttfas = [r.time_to_first_audio_sec for r in successes]
            mean_ttfa = round(sum(ttfas) / len(ttfas), 4) if ttfas else 0.0
            p99_ttfa = round(percentile(ttfas, 99), 4) if ttfas else 0.0

            # Inter-chunk latency
            inter_chunks = [r.mean_inter_chunk_ms for r in successes if r.mean_inter_chunk_ms > 0]
            mean_inter_chunk = round(sum(inter_chunks) / len(inter_chunks), 2) if inter_chunks else 0.0

            # Chars throughput
            total_chars = sum(r.prompt_chars for r in successes)
            chars_per_sec = round(total_chars / max(total_duration, 1e-9), 1)

            extra = {
                "rtfx": rtfx,
                "mean_time_to_first_token_sec": mean_ttft,
                "p99_time_to_first_token_sec": p99_ttft,
                "mean_time_to_first_audio_sec": mean_ttfa,
                "p99_time_to_first_audio_sec": p99_ttfa,
                "mean_inter_chunk_ms": mean_inter_chunk,
                "chars_per_sec": chars_per_sec,
                "total_audio_sec": round(total_audio_sec, 2),
            }
            extra.update(monitor.snapshot(since=level_gpu_t0))

            concurrency_results.append(ConcurrencyResult(
                concurrency=concurrency,
                latency=latency_stats,
                extra=extra,
            ))
            _log(f"  -> RTFX={rtfx}  TTFT={mean_ttft:.3f}s  TTFA={mean_ttfa:.3f}s  "
                 f"inter_chunk={mean_inter_chunk:.1f}ms  latency={latency_stats.mean_sec:.3f}s")

        else:  # offline
            results_list_off: list[OfflineTtsResult] = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _synthesize_offline,
                        p, auth, voice_name, sample_rate_hz, language_code,
                    ): p
                    for p in level_prompts
                }
                for future in as_completed(futures):
                    results_list_off.append(future.result())

            total_duration = time.perf_counter() - t_start
            successes_off = [r for r in results_list_off if r.success]
            errors = len(results_list_off) - len(successes_off)

            latencies = [r.latency_sec for r in successes_off]
            latency_stats = compute_latency_stats(latencies, errors, total_duration)

            total_audio_sec = sum(r.output_audio_duration_sec for r in successes_off)
            rtfx = round(total_audio_sec / total_duration, 2) if total_duration > 0 else 0.0

            total_chars = sum(r.prompt_chars for r in successes_off)
            chars_per_sec = round(total_chars / max(total_duration, 1e-9), 1)

            extra = {
                "rtfx": rtfx,
                "chars_per_sec": chars_per_sec,
                "total_audio_sec": round(total_audio_sec, 2),
            }
            extra.update(monitor.snapshot(since=level_gpu_t0))

            concurrency_results.append(ConcurrencyResult(
                concurrency=concurrency,
                latency=latency_stats,
                extra=extra,
            ))
            _log(f"  -> RTFX={rtfx}  latency={latency_stats.mean_sec:.3f}s  chars/s={chars_per_sec}")

    gpu_summary = monitor.stop()

    return {
        "results": [r.to_dict() for r in concurrency_results],
        "gpu_summary": gpu_summary.to_dict() if gpu_summary else None,
        "config_used": {
            "host": host, "port": port, "use_ssl": use_ssl,
            "mode": mode, "voice_name": voice_name,
            "sample_rate_hz": sample_rate_hz,
        },
    }
