"""LLM benchmark against Nvidia NIM (meta/llama-3.1-8b-instruct) via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
import json
import random
import time
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
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("openai package not installed; LLM benchmark unavailable.")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"
PROMPTS_PATH = Path(__file__).parent.parent / "data" / "prompts" / "llm_prompts.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompts(tier: str = "all") -> list[str]:
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    if tier == "all":
        prompts = []
        for t in ("short", "medium", "long"):
            prompts.extend(data.get(t, []))
        return prompts
    return data.get(tier, [])


@dataclass
class LlmRequestResult:
    prompt: str
    ttft_sec: float
    total_latency_sec: float
    generation_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float
    success: bool
    error: str = ""


async def _complete_one(
    client: "AsyncOpenAI",
    prompt: str,
    model: str,
    max_tokens: int,
) -> LlmRequestResult:
    """Stream one chat completion and capture TTFT + token metrics."""
    if not _OPENAI_AVAILABLE:
        return LlmRequestResult(
            prompt=prompt, ttft_sec=0, total_latency_sec=0,
            generation_time_sec=0, prompt_tokens=0, completion_tokens=0,
            tokens_per_sec=0, success=False, error="openai not installed",
        )
    t_start = time.perf_counter()
    first_token_time: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if (
                first_token_time is None
                and chunk.choices
                and chunk.choices[0].delta.content
            ):
                first_token_time = time.perf_counter() - t_start

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    completion_tokens += 1

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens

        t_end = time.perf_counter()
        total_latency = t_end - t_start
        ttft = first_token_time if first_token_time is not None else total_latency
        generation_time = max(total_latency - ttft, 1e-9)
        tokens_per_sec = completion_tokens / generation_time if completion_tokens > 0 else 0.0

        return LlmRequestResult(
            prompt=prompt,
            ttft_sec=round(ttft, 4),
            total_latency_sec=round(total_latency, 4),
            generation_time_sec=round(generation_time, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_sec=round(tokens_per_sec, 2),
            success=True,
        )
    except Exception as e:
        t_end = time.perf_counter()
        logger.debug(f"LLM request error: {e}")
        return LlmRequestResult(
            prompt=prompt, ttft_sec=0,
            total_latency_sec=round(t_end - t_start, 4),
            generation_time_sec=0, prompt_tokens=0, completion_tokens=0,
            tokens_per_sec=0, success=False, error=str(e),
        )


async def _run_level(
    prompts: list[str],
    concurrency: int,
    requests_per_level: int,
    client: "AsyncOpenAI",
    model: str,
    max_tokens: int,
) -> tuple[list[LlmRequestResult], float]:
    """Run one concurrency level using a semaphore to cap parallelism."""
    level_prompts = random.choices(prompts, k=requests_per_level)
    semaphore = asyncio.Semaphore(concurrency)

    async def _limited(prompt: str) -> LlmRequestResult:
        async with semaphore:
            return await _complete_one(client, prompt, model, max_tokens)

    t_start = time.perf_counter()
    results = await asyncio.gather(*[_limited(p) for p in level_prompts])
    total_duration = time.perf_counter() - t_start
    return list(results), total_duration


async def _benchmark_async(
    prompts: list[str],
    concurrency_levels: list[int],
    requests_per_level: int,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    progress_callback: Optional[Callable[[str], None]],
) -> list[ConcurrencyResult]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    concurrency_results: list[ConcurrencyResult] = []

    for concurrency in concurrency_levels:
        _log(f"LLM benchmark: concurrency={concurrency}, requests={requests_per_level}")

        request_results, total_duration = await _run_level(
            prompts, concurrency, requests_per_level, client, model, max_tokens
        )

        successes = [r for r in request_results if r.success]
        errors = len(request_results) - len(successes)
        latencies = [r.total_latency_sec for r in successes]
        ttfts = [r.ttft_sec for r in successes]

        latency_stats = compute_latency_stats(latencies, errors, total_duration)

        mean_ttft = round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0
        p90_ttft = round(percentile(ttfts, 90), 4) if ttfts else 0.0
        tps_vals = [r.tokens_per_sec for r in successes if r.tokens_per_sec > 0]
        mean_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0

        concurrency_results.append(ConcurrencyResult(
            concurrency=concurrency,
            latency=latency_stats,
            extra={
                "mean_ttft_sec": mean_ttft,
                "p90_ttft_sec": p90_ttft,
                "mean_tokens_per_sec": mean_tps,
                "total_prompt_tokens": sum(r.prompt_tokens for r in successes),
                "total_completion_tokens": sum(r.completion_tokens for r in successes),
            },
        ))
        _log(
            f"  -> mean_latency={latency_stats.mean_sec:.3f}s "
            f"TTFT={mean_ttft:.3f}s tokens/s={mean_tps:.1f}"
        )

    await client.close()
    return concurrency_results


def run_llm_benchmark(
    concurrency_levels: list[int],
    requests_per_level: int = 50,
    prompt_tier: str = "all",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 256,
    gpu_index: int = 0,
    gpu_monitor_interval: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run LLM benchmark across multiple concurrency levels.

    Returns dict with:
      - results: list of ConcurrencyResult dicts
      - gpu_summary: GpuSummary dict or None
      - config_used: dict
    """
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    base_url = base_url or llm_cfg.get("base_url", "http://localhost:8000/v1")
    model = model or llm_cfg.get("model", "meta/llama-3.1-8b-instruct")
    api_key = api_key or llm_cfg.get("api_key", "not-required")

    prompts = load_prompts(prompt_tier)
    if not prompts:
        raise ValueError(f"No prompts found for tier='{prompt_tier}'.")

    monitor = GpuMonitor(gpu_index=gpu_index, interval=gpu_monitor_interval)
    monitor.start()

    try:
        concurrency_results = asyncio.run(
            _benchmark_async(
                prompts=prompts,
                concurrency_levels=concurrency_levels,
                requests_per_level=requests_per_level,
                base_url=base_url,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                progress_callback=progress_callback,
            )
        )
    finally:
        gpu_summary = monitor.stop()

    return {
        "results": [r.to_dict() for r in concurrency_results],
        "gpu_summary": gpu_summary.to_dict() if gpu_summary else None,
        "config_used": {
            "base_url": base_url,
            "model": model,
            "max_tokens": max_tokens,
        },
    }
