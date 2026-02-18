"""TTS benchmark against Nvidia Riva Magpie TTS NIM via gRPC."""

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
    compute_rtf,
)
from benchmarks.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)

try:
    import riva.client as riva_client
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


@dataclass
class TtsRequestResult:
    prompt: str
    prompt_chars: int
    latency_sec: float
    time_to_first_byte_sec: float
    output_audio_duration_sec: float
    rtf: float
    success: bool
    error: str = ""


def _synthesize_one(
    prompt: str,
    host: str,
    port: int,
    use_ssl: bool,
    voice_name: str,
    sample_rate_hz: int,
    language_code: str,
) -> TtsRequestResult:
    """Synthesize speech for one prompt using Riva TTS gRPC streaming."""
    if not _RIVA_AVAILABLE:
        return TtsRequestResult(
            prompt=prompt, prompt_chars=len(prompt),
            latency_sec=0, time_to_first_byte_sec=0,
            output_audio_duration_sec=0, rtf=0,
            success=False, error="nvidia-riva-client not installed",
        )
    try:
        auth = riva_client.Auth(uri=f"{host}:{port}", use_ssl=use_ssl)
        tts_service = riva_client.SpeechSynthesisService(auth)

        t_start = time.perf_counter()
        first_byte_time: Optional[float] = None
        audio_bytes_total = bytearray()

        responses = tts_service.synthesize_online(
            text=prompt,
            voice_name=voice_name,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
        )

        for response in responses:
            if first_byte_time is None:
                first_byte_time = time.perf_counter() - t_start
            if hasattr(response, 'audio') and response.audio:
                audio_bytes_total.extend(response.audio)

        t_end = time.perf_counter()
        latency = t_end - t_start
        ttfb = first_byte_time if first_byte_time is not None else latency

        # Riva TTS returns 16-bit PCM (2 bytes per sample)
        bytes_per_sample = 2
        num_samples = len(audio_bytes_total) / bytes_per_sample
        output_duration = num_samples / sample_rate_hz if sample_rate_hz > 0 else 0.0

        rtf = compute_rtf(output_duration, latency)

        return TtsRequestResult(
            prompt=prompt,
            prompt_chars=len(prompt),
            latency_sec=latency,
            time_to_first_byte_sec=ttfb,
            output_audio_duration_sec=output_duration,
            rtf=rtf,
            success=True,
        )
    except Exception as e:
        logger.debug(f"TTS request error: {e}")
        return TtsRequestResult(
            prompt=prompt, prompt_chars=len(prompt),
            latency_sec=0, time_to_first_byte_sec=0,
            output_audio_duration_sec=0, rtf=0,
            success=False, error=str(e),
        )


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
    gpu_index: int = 0,
    gpu_monitor_interval: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run TTS benchmark across multiple concurrency levels.

    Returns dict with:
      - results: list of ConcurrencyResult dicts
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

    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    monitor = GpuMonitor(gpu_index=gpu_index, interval=gpu_monitor_interval)
    monitor.start()

    concurrency_results: list[ConcurrencyResult] = []

    for concurrency in concurrency_levels:
        _log(f"TTS benchmark: concurrency={concurrency}, requests={requests_per_level}")

        level_prompts = random.choices(prompts, k=requests_per_level)

        t_start = time.perf_counter()
        request_results: list[TtsRequestResult] = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _synthesize_one,
                    p, host, port, use_ssl, voice_name, sample_rate_hz, language_code,
                ): p
                for p in level_prompts
            }
            for future in as_completed(futures):
                request_results.append(future.result())

        total_duration = time.perf_counter() - t_start

        successes = [r for r in request_results if r.success]
        errors = len(request_results) - len(successes)
        latencies = [r.latency_sec for r in successes]

        latency_stats = compute_latency_stats(latencies, errors, total_duration)

        mean_rtf = (
            round(sum(r.rtf for r in successes) / len(successes), 4)
            if successes else 0.0
        )
        mean_ttfb = (
            round(sum(r.time_to_first_byte_sec for r in successes) / len(successes), 4)
            if successes else 0.0
        )
        total_chars = sum(r.prompt_chars for r in successes)
        chars_per_sec = round(total_chars / max(total_duration, 1e-9), 1)

        concurrency_results.append(ConcurrencyResult(
            concurrency=concurrency,
            latency=latency_stats,
            extra={
                "mean_rtf": mean_rtf,
                "mean_time_to_first_byte_sec": mean_ttfb,
                "chars_per_sec": chars_per_sec,
                "prompt_tier": prompt_tier,
            },
        ))
        _log(
            f"  -> mean_latency={latency_stats.mean_sec:.3f}s "
            f"TTFB={mean_ttfb:.3f}s RTF={mean_rtf:.3f}"
        )

    gpu_summary = monitor.stop()

    return {
        "results": [r.to_dict() for r in concurrency_results],
        "gpu_summary": gpu_summary.to_dict() if gpu_summary else None,
        "config_used": {
            "host": host, "port": port, "use_ssl": use_ssl,
            "voice_name": voice_name, "sample_rate_hz": sample_rate_hz,
        },
    }
