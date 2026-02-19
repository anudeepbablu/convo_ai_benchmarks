#!/usr/bin/env python3
"""CLI runner for ASR and LLM benchmarks — no Streamlit UI required.

Usage:
    python run_bench.py asr
    python run_bench.py llm --max-tokens 128 --requests 30
    python run_bench.py asr --host 10.0.0.5 --port 50051 --concurrency 1,2,4,8,16

For TTS benchmarks, use run_tts_bench.py instead.
All flags are optional — defaults come from config/endpoints.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "endpoints.yaml"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_bench")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def parse_concurrency(raw: str | None, cfg: dict) -> list[int]:
    """Parse '1,5,10,20' or fall back to config min/max/step range."""
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    bench = cfg.get("benchmark", {})
    lo = bench.get("default_concurrency_min", 1)
    hi = bench.get("default_concurrency_max", 50)
    step = bench.get("default_concurrency_step", 5)
    levels = list(range(lo, hi + 1, step))
    return levels or [lo]


def save_result(pipeline: str, data: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{pipeline}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def print_summary(result: dict, pipeline: str) -> None:
    """Print a human-readable summary table to stdout."""
    results = result.get("results", [])
    if not results:
        print("No results collected.")
        return

    # Header
    cols = ["Conc", "Mean(ms)", "P90(ms)", "P95(ms)", "P99(ms)", "Tput(r/s)", "Errs"]
    extra_cols: list[str] = []

    if pipeline == "asr":
        mode = result.get("config_used", {}).get("mode", "offline")
        if mode == "streaming":
            extra_cols = ["RTFX", "Audio(s)", "Stream(s)", "Tail(ms)", "WER"]
        else:
            extra_cols = ["RTFX", "WER"]
    elif pipeline == "llm":
        extra_cols = ["TTFT(ms)", "Tok/s"]

    header = cols + extra_cols
    widths = [max(len(h), 9) for h in header]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(v.rjust(w) for v, w in zip(row, widths))

    print()
    print(fmt_row(header))
    print("  ".join("─" * w for w in widths))

    for r in results:
        row = [
            str(r.get("concurrency", "")),
            f"{r.get('mean_sec', 0) * 1000:.1f}",
            f"{r.get('p90_sec', 0) * 1000:.1f}",
            f"{r.get('p95_sec', 0) * 1000:.1f}",
            f"{r.get('p99_sec', 0) * 1000:.1f}",
            f"{r.get('throughput_rps', 0):.1f}",
            str(r.get("error_count", 0)),
        ]

        if pipeline == "asr":
            row.append(f"{r.get('rtfx', 0):.1f}")
            if mode == "streaming":
                row.append(f"{r.get('mean_audio_duration_sec', 0):.1f}")
                row.append(f"{r.get('mean_audio_streaming_sec', 0):.1f}")
                row.append(f"{r.get('mean_server_compute_after_audio_sec', 0) * 1000:.1f}")
            wer = r.get("wer")
            row.append(f"{wer * 100:.2f}%" if wer is not None else "n/a")
        elif pipeline == "llm":
            row.append(f"{r.get('mean_ttft_sec', 0) * 1000:.1f}")
            row.append(f"{r.get('mean_tokens_per_sec', 0):.1f}")

        print(fmt_row(row))

    # GPU summary
    gpu = result.get("gpu_summary")
    if gpu:
        print()
        print(f"GPU: {gpu.get('gpu_name', 'unknown')}  |  "
              f"Peak util: {gpu.get('peak_utilization_gpu_pct', 0)}%  |  "
              f"Peak VRAM: {gpu.get('peak_memory_used_mb', 0):.0f} MB  |  "
              f"Peak temp: {gpu.get('peak_temperature_c', 0)}°C  |  "
              f"Peak power: {gpu.get('peak_power_draw_w', 0):.0f} W")


def generate_report(result: dict, pipeline: str, wall_time: float) -> Path:
    """Write a human-readable markdown report alongside the JSON results."""
    results = result.get("results", [])
    ts = result.get("timestamp", datetime.now().isoformat())
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_used = result.get("config_used", {})

    PIPELINE_LABELS = {
        "asr": "ASR — Parakeet 1.1B CTC (gRPC)",
        "llm": "LLM — Llama 3.1-8B Instruct (HTTP)",
    }

    lines: list[str] = []
    w = lines.append

    w(f"# NIM Benchmark Report — {pipeline.upper()}")
    w("")
    w(f"**Pipeline:** {PIPELINE_LABELS.get(pipeline, pipeline)}")
    w(f"**Timestamp:** {ts}")
    w(f"**Total wall time:** {wall_time:.1f}s")
    w("")

    # Config
    w("## Configuration")
    w("")
    for k, v in config_used.items():
        w(f"- **{k}:** `{v}`")
    if results:
        concurrencies = [r["concurrency"] for r in results]
        w(f"- **concurrency levels:** {concurrencies}")
        w(f"- **requests per level:** {results[0].get('count', '?')}")
    w("")

    # Results table
    w("## Results")
    w("")

    headers = ["Concurrency", "Mean (ms)", "P90 (ms)", "P95 (ms)", "P99 (ms)", "Throughput (r/s)", "Errors"]

    asr_mode = config_used.get("mode", "offline")
    if pipeline == "asr":
        if asr_mode == "streaming":
            headers += ["RTFX", "Audio (s)", "Streaming (s)", "Server Tail (ms)", "WER"]
        else:
            headers += ["RTFX", "WER"]
    elif pipeline == "llm":
        headers += ["TTFT (ms)", "Tok/s"]

    w("| " + " | ".join(headers) + " |")
    w("| " + " | ".join("---:" for _ in headers) + " |")

    for r in results:
        cells = [
            str(r.get("concurrency", "")),
            f"{r.get('mean_sec', 0) * 1000:.1f}",
            f"{r.get('p90_sec', 0) * 1000:.1f}",
            f"{r.get('p95_sec', 0) * 1000:.1f}",
            f"{r.get('p99_sec', 0) * 1000:.1f}",
            f"{r.get('throughput_rps', 0):.1f}",
            str(r.get("error_count", 0)),
        ]

        if pipeline == "asr":
            cells.append(f"{r.get('rtfx', 0):.1f}")
            if asr_mode == "streaming":
                cells.append(f"{r.get('mean_audio_duration_sec', 0):.1f}")
                cells.append(f"{r.get('mean_audio_streaming_sec', 0):.1f}")
                cells.append(f"{r.get('mean_server_compute_after_audio_sec', 0) * 1000:.1f}")
            wer = r.get("wer")
            cells.append(f"{wer * 100:.2f}%" if wer is not None else "n/a")
        elif pipeline == "llm":
            cells.append(f"{r.get('mean_ttft_sec', 0) * 1000:.1f}")
            cells.append(f"{r.get('mean_tokens_per_sec', 0):.1f}")

        w("| " + " | ".join(cells) + " |")

    w("")

    # Best / worst concurrency highlights
    if results:
        best_tput = max(results, key=lambda r: r.get("throughput_rps", 0))
        best_lat = min(results, key=lambda r: r.get("mean_sec", float("inf")))
        w("## Highlights")
        w("")
        w(f"- **Best throughput:** {best_tput.get('throughput_rps', 0):.1f} req/s at concurrency {best_tput['concurrency']}")
        w(f"- **Lowest mean latency:** {best_lat.get('mean_sec', 0) * 1000:.1f} ms at concurrency {best_lat['concurrency']}")

        if pipeline == "asr":
            best_rtfx = max(results, key=lambda r: r.get("rtfx", 0))
            w(f"- **Best RTFX:** {best_rtfx.get('rtfx', 0):.1f}x real-time at concurrency {best_rtfx['concurrency']}")
            wer_vals = [(r["concurrency"], r.get("wer")) for r in results if r.get("wer") is not None]
            if wer_vals:
                w(f"- **WER (last level):** {wer_vals[-1][1] * 100:.2f}%")
        elif pipeline == "llm":
            best_tps = max(results, key=lambda r: r.get("mean_tokens_per_sec", 0))
            w(f"- **Best tokens/sec:** {best_tps.get('mean_tokens_per_sec', 0):.1f} at concurrency {best_tps['concurrency']}")

        total_errors = sum(r.get("error_count", 0) for r in results)
        total_reqs = sum(r.get("count", 0) for r in results)
        w(f"- **Total errors:** {total_errors}/{total_reqs}")
        w("")

    # GPU section
    gpu = result.get("gpu_summary")
    if gpu:
        w("## GPU Utilization")
        w("")
        w(f"**GPU:** {gpu.get('gpu_name', 'unknown')}")
        w(f"**Total VRAM:** {gpu.get('memory_total_mb', 0) / 1024:.1f} GB")
        w(f"**Monitoring duration:** {gpu.get('duration_sec', 0):.1f}s ({gpu.get('sample_count', 0)} samples)")
        w("")
        w("| Metric | Peak | Mean |")
        w("| :--- | ---: | ---: |")
        w(f"| GPU Utilization (%) | {gpu.get('peak_utilization_gpu_pct', 0)} | {gpu.get('mean_utilization_gpu_pct', 0)} |")
        w(f"| Memory Used (MB) | {gpu.get('peak_memory_used_mb', 0):.0f} | {gpu.get('mean_memory_used_mb', 0):.0f} |")
        w(f"| Temperature (C) | {gpu.get('peak_temperature_c', 0)} | {gpu.get('mean_temperature_c', 0)} |")
        w(f"| Power Draw (W) | {gpu.get('peak_power_draw_w', 0):.0f} | {gpu.get('mean_power_draw_w', 0):.0f} |")
        w("")

    path = RESULTS_DIR / f"{pipeline}_{ts_file}.md"
    path.write_text("\n".join(lines))
    return path


# ── Subcommand handlers ─────────────────────────────────────────────────────

def run_asr(args: argparse.Namespace, cfg: dict) -> None:
    from benchmarks.asr_benchmark import run_asr_benchmark
    from data_prep.download_librispeech import load_transcripts, AUDIO_DIR

    asr_cfg = cfg.get("asr", {})
    host = args.host or asr_cfg.get("host", "localhost")
    port = args.port or asr_cfg.get("port", 50051)
    use_ssl = args.ssl if args.ssl else asr_cfg.get("use_ssl", False)
    levels = parse_concurrency(args.concurrency, cfg)

    transcripts = load_transcripts()
    audio_dir = str(AUDIO_DIR)
    if not AUDIO_DIR.exists() or not list(AUDIO_DIR.glob("*.wav")):
        logger.error("No audio files found. Run: python data_prep/download_librispeech.py")
        sys.exit(1)

    logger.info(f"ASR benchmark — host={host}:{port} ssl={use_ssl} mode={args.mode}")
    logger.info(f"  concurrency={levels}  requests/level={args.requests}")
    if args.mode == "streaming":
        logger.info(f"  chunk={args.chunk_duration_ms}ms  simulate_realtime={args.simulate_realtime}")

    t0 = time.time()
    result = run_asr_benchmark(
        audio_dir=audio_dir,
        transcripts=transcripts,
        concurrency_levels=levels,
        requests_per_level=args.requests,
        host=host,
        port=port,
        use_ssl=use_ssl,
        mode=args.mode,
        chunk_duration_ms=args.chunk_duration_ms,
        simulate_realtime=args.simulate_realtime,
        gpu_index=args.gpu,
        gpu_monitor_interval=args.gpu_interval,
        progress_callback=lambda msg: logger.info(msg),
    )
    wall_time = time.time() - t0
    result["pipeline"] = "asr"
    result["timestamp"] = datetime.now().isoformat()

    json_path = save_result("asr", result)
    md_path = generate_report(result, "asr", wall_time)
    print_summary(result, "asr")
    logger.info(f"JSON  -> {json_path}")
    logger.info(f"Report -> {md_path}")


def run_llm(args: argparse.Namespace, cfg: dict) -> None:
    from benchmarks.llm_benchmark import run_llm_benchmark

    llm_cfg = cfg.get("llm", {})
    base_url = args.base_url or llm_cfg.get("base_url", "http://localhost:8000/v1")
    model = args.model or llm_cfg.get("model", "meta/llama-3.1-8b-instruct")
    api_key = args.api_key or llm_cfg.get("api_key", "not-required")
    levels = parse_concurrency(args.concurrency, cfg)

    logger.info(f"LLM benchmark — url={base_url} model={model}")
    logger.info(f"  concurrency={levels}  requests/level={args.requests}  tier={args.prompt_tier}")

    t0 = time.time()
    result = run_llm_benchmark(
        concurrency_levels=levels,
        requests_per_level=args.requests,
        prompt_tier=args.prompt_tier,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        gpu_index=args.gpu,
        gpu_monitor_interval=args.gpu_interval,
        progress_callback=lambda msg: logger.info(msg),
    )
    wall_time = time.time() - t0
    result["pipeline"] = "llm"
    result["timestamp"] = datetime.now().isoformat()

    json_path = save_result("llm", result)
    md_path = generate_report(result, "llm", wall_time)
    print_summary(result, "llm")
    logger.info(f"JSON  -> {json_path}")
    logger.info(f"Report -> {md_path}")


# ── CLI definition ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASR/LLM benchmarks from the command line. For TTS, use run_tts_bench.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="pipeline", required=True)

    # Shared arguments added to every subcommand
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--concurrency", type=str, default=None,
                       help="Comma-separated concurrency levels, e.g. '1,5,10,20'. "
                            "Default: range from endpoints.yaml")
        p.add_argument("-n", "--requests", type=int, default=50,
                       help="Requests per concurrency level (default: 50)")
        p.add_argument("--gpu", type=int, default=0,
                       help="GPU index to monitor (default: 0)")
        p.add_argument("--gpu-interval", type=float, default=1.0,
                       help="GPU sampling interval in seconds (default: 1.0)")

    # ── ASR ──
    asr = sub.add_parser("asr", help="Benchmark ASR (Parakeet 1.1B CTC)")
    add_common(asr)
    asr.add_argument("--host", type=str, default=None, help="gRPC host")
    asr.add_argument("--port", type=int, default=None, help="gRPC port")
    asr.add_argument("--ssl", action="store_true", help="Enable SSL")
    asr.add_argument("--mode", type=str, default="streaming",
                     choices=["streaming", "offline"],
                     help="Benchmark mode: 'streaming' (chunked, matches riva_streaming_asr_client) "
                          "or 'offline' (batch, matches riva_asr_client). Default: streaming")
    asr.add_argument("--chunk-duration-ms", type=float, default=800,
                     help="Audio chunk duration in ms for streaming mode (default: 800)")
    asr.add_argument("--simulate-realtime", action="store_true", default=True,
                     help="Pace audio chunks at real-time speed in streaming mode (default: True)")
    asr.add_argument("--no-simulate-realtime", dest="simulate_realtime", action="store_false",
                     help="Send audio chunks as fast as possible (no pacing)")

    # ── LLM ──
    llm = sub.add_parser("llm", help="Benchmark LLM (Llama 3.1-8B)")
    add_common(llm)
    llm.add_argument("--base-url", type=str, default=None,
                     help="OpenAI-compatible base URL")
    llm.add_argument("--model", type=str, default=None, help="Model name")
    llm.add_argument("--api-key", type=str, default=None, help="API key")
    llm.add_argument("--max-tokens", type=int, default=256,
                     help="Max completion tokens (default: 256)")
    llm.add_argument("--prompt-tier", type=str, default="all",
                     choices=["all", "short", "medium", "long"],
                     help="Prompt complexity tier (default: all)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()

    if args.pipeline == "asr":
        run_asr(args, cfg)
    elif args.pipeline == "llm":
        run_llm(args, cfg)


if __name__ == "__main__":
    main()
