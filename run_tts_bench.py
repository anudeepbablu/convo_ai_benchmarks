#!/usr/bin/env python3
"""Standalone CLI runner for TTS benchmarks against Nvidia Riva NIM.

Usage:
    python run_tts_bench.py
    python run_tts_bench.py --mode streaming --concurrency 1,4,8,16,32
    python run_tts_bench.py --mode offline --prompt-tier short -n 100
    python run_tts_bench.py --host 10.0.0.5 --port 50052 --voice "English-US.Male-1"

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
logger = logging.getLogger("run_tts_bench")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def parse_concurrency(raw: str | None, cfg: dict) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    bench = cfg.get("benchmark", {})
    lo = bench.get("default_concurrency_min", 1)
    hi = bench.get("default_concurrency_max", 50)
    step = bench.get("default_concurrency_step", 5)
    levels = list(range(lo, hi + 1, step))
    return levels or [lo]


def save_json(data: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"tts_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ── Console summary ─────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    results = result.get("results", [])
    if not results:
        print("No results collected.")
        return

    mode = result.get("config_used", {}).get("mode", "streaming")

    # Build columns
    cols = ["Conc", "Mean(ms)", "P90(ms)", "P95(ms)", "P99(ms)", "Tput(r/s)", "Errs"]
    if mode == "streaming":
        cols += ["RTFX", "TTFT(ms)", "TTFA(ms)", "P99 TTFA", "IChunk(ms)", "Chars/s", "Audio(s)"]
    else:
        cols += ["RTFX", "Chars/s", "Audio(s)"]

    widths = [max(len(h), 9) for h in cols]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(v.rjust(w) for v, w in zip(row, widths))

    print()
    print(fmt_row(cols))
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

        row.append(f"{r.get('rtfx', 0):.1f}")
        if mode == "streaming":
            row.append(f"{r.get('mean_time_to_first_token_sec', 0) * 1000:.1f}")
            row.append(f"{r.get('mean_time_to_first_audio_sec', 0) * 1000:.1f}")
            row.append(f"{r.get('p99_time_to_first_audio_sec', 0) * 1000:.1f}")
            row.append(f"{r.get('mean_inter_chunk_ms', 0):.1f}")
        row.append(f"{r.get('chars_per_sec', 0):.0f}")
        row.append(f"{r.get('total_audio_sec', 0):.1f}")

        print(fmt_row(row))

    # Per-level GPU stats
    has_gpu = any(r.get("gpu_util_peak_pct") is not None for r in results)
    if has_gpu:
        gpu_cols = ["Conc", "GPU Util%", "GPU Mean%", "VRAM Peak(MB)", "VRAM Mean(MB)", "Power Peak(W)", "Temp(C)"]
        gpu_widths = [max(len(h), 9) for h in gpu_cols]

        def fmt_gpu(row: list[str]) -> str:
            return "  ".join(v.rjust(w) for v, w in zip(row, gpu_widths))

        print()
        print(fmt_gpu(gpu_cols))
        print("  ".join("─" * w for w in gpu_widths))
        for r in results:
            print(fmt_gpu([
                str(r.get("concurrency", "")),
                f"{r.get('gpu_util_peak_pct', 0):.0f}",
                f"{r.get('gpu_util_mean_pct', 0):.0f}",
                f"{r.get('gpu_mem_peak_mb', 0):.0f}",
                f"{r.get('gpu_mem_mean_mb', 0):.0f}",
                f"{r.get('gpu_power_peak_w', 0):.0f}",
                f"{r.get('gpu_temp_peak_c', 0):.0f}",
            ]))

    # Overall GPU summary
    gpu = result.get("gpu_summary")
    if gpu:
        print()
        print(f"GPU: {gpu.get('gpu_name', 'unknown')}  |  "
              f"Peak util: {gpu.get('peak_utilization_gpu_pct', 0)}%  |  "
              f"Peak VRAM: {gpu.get('peak_memory_used_mb', 0):.0f} MB  |  "
              f"Peak temp: {gpu.get('peak_temperature_c', 0)}°C  |  "
              f"Peak power: {gpu.get('peak_power_draw_w', 0):.0f} W")


# ── Markdown report ─────────────────────────────────────────────────────────

def generate_report(result: dict, wall_time: float) -> Path:
    results = result.get("results", [])
    config_used = result.get("config_used", {})
    mode = config_used.get("mode", "streaming")
    ts = result.get("timestamp", datetime.now().isoformat())
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines: list[str] = []
    w = lines.append

    w("# TTS Benchmark Report")
    w("")
    w(f"**Pipeline:** TTS — Riva NIM (gRPC)")
    w(f"**Mode:** {mode} {'(synthesize_online — matches riva_tts_perf_client --online=true)' if mode == 'streaming' else '(synthesize — unary RPC)'}")
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
    if mode == "streaming":
        headers += ["RTFX", "TTFT (ms)", "TTFA (ms)", "P99 TTFA (ms)", "Inter-Chunk (ms)", "Chars/s", "Audio (s)"]
    else:
        headers += ["RTFX", "Chars/s", "Audio (s)"]

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

        cells.append(f"{r.get('rtfx', 0):.1f}")
        if mode == "streaming":
            cells.append(f"{r.get('mean_time_to_first_token_sec', 0) * 1000:.1f}")
            cells.append(f"{r.get('mean_time_to_first_audio_sec', 0) * 1000:.1f}")
            cells.append(f"{r.get('p99_time_to_first_audio_sec', 0) * 1000:.1f}")
            cells.append(f"{r.get('mean_inter_chunk_ms', 0):.1f}")
        cells.append(f"{r.get('chars_per_sec', 0):.0f}")
        cells.append(f"{r.get('total_audio_sec', 0):.1f}")

        w("| " + " | ".join(cells) + " |")

    w("")

    # Highlights
    if results:
        w("## Highlights")
        w("")

        best_tput = max(results, key=lambda r: r.get("throughput_rps", 0))
        best_lat = min(results, key=lambda r: r.get("mean_sec", float("inf")))
        best_rtfx = max(results, key=lambda r: r.get("rtfx", 0))

        w(f"- **Best throughput:** {best_tput.get('throughput_rps', 0):.1f} req/s at concurrency {best_tput['concurrency']}")
        w(f"- **Lowest mean latency:** {best_lat.get('mean_sec', 0) * 1000:.1f} ms at concurrency {best_lat['concurrency']}")
        w(f"- **Best RTFX:** {best_rtfx.get('rtfx', 0):.1f}x real-time at concurrency {best_rtfx['concurrency']}")

        total_audio = sum(r.get("total_audio_sec", 0) for r in results)
        w(f"- **Total audio generated:** {total_audio:.1f}s across all levels")

        if mode == "streaming":
            c1 = results[0]
            cmax = results[-1]
            w(f"- **TTFT at c=1:** {c1.get('mean_time_to_first_token_sec', 0) * 1000:.1f}ms  "
              f"(p99: {c1.get('p99_time_to_first_token_sec', 0) * 1000:.1f}ms)")
            w(f"- **TTFA at c=1:** {c1.get('mean_time_to_first_audio_sec', 0) * 1000:.1f}ms  "
              f"(p99: {c1.get('p99_time_to_first_audio_sec', 0) * 1000:.1f}ms)")
            w(f"- **TTFT at c={cmax['concurrency']}:** {cmax.get('mean_time_to_first_token_sec', 0) * 1000:.1f}ms  "
              f"(p99: {cmax.get('p99_time_to_first_token_sec', 0) * 1000:.1f}ms)")
            w(f"- **TTFA at c={cmax['concurrency']}:** {cmax.get('mean_time_to_first_audio_sec', 0) * 1000:.1f}ms  "
              f"(p99: {cmax.get('p99_time_to_first_audio_sec', 0) * 1000:.1f}ms)")

        total_errors = sum(r.get("error_count", 0) for r in results)
        total_reqs = sum(r.get("count", 0) for r in results)
        w(f"- **Total errors:** {total_errors}/{total_reqs}")
        w("")

    # Per-level GPU table
    has_gpu = any(r.get("gpu_util_peak_pct") is not None for r in results)
    gpu = result.get("gpu_summary")

    if has_gpu or gpu:
        w("## GPU Utilization")
        w("")

    if gpu:
        w(f"**GPU:** {gpu.get('gpu_name', 'unknown')}")
        w(f"**Total VRAM:** {gpu.get('memory_total_mb', 0) / 1024:.1f} GB")
        w(f"**Monitoring duration:** {gpu.get('duration_sec', 0):.1f}s ({gpu.get('sample_count', 0)} samples)")
        w("")

    if has_gpu:
        w("### Per-Concurrency GPU Stats")
        w("")
        w("| Concurrency | GPU Util Peak (%) | GPU Util Mean (%) | VRAM Peak (MB) | VRAM Mean (MB) | Power Peak (W) | Temp Peak (C) |")
        w("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results:
            w(f"| {r.get('concurrency', '')} "
              f"| {r.get('gpu_util_peak_pct', 0):.0f} "
              f"| {r.get('gpu_util_mean_pct', 0):.0f} "
              f"| {r.get('gpu_mem_peak_mb', 0):.0f} "
              f"| {r.get('gpu_mem_mean_mb', 0):.0f} "
              f"| {r.get('gpu_power_peak_w', 0):.0f} "
              f"| {r.get('gpu_temp_peak_c', 0):.0f} |")
        w("")

    if gpu:
        w("### Overall Summary")
        w("")
        w("| Metric | Peak | Mean |")
        w("| :--- | ---: | ---: |")
        w(f"| GPU Utilization (%) | {gpu.get('peak_utilization_gpu_pct', 0)} | {gpu.get('mean_utilization_gpu_pct', 0)} |")
        w(f"| Memory Used (MB) | {gpu.get('peak_memory_used_mb', 0):.0f} | {gpu.get('mean_memory_used_mb', 0):.0f} |")
        w(f"| Temperature (C) | {gpu.get('peak_temperature_c', 0)} | {gpu.get('mean_temperature_c', 0)} |")
        w(f"| Power Draw (W) | {gpu.get('peak_power_draw_w', 0):.0f} | {gpu.get('mean_power_draw_w', 0):.0f} |")
        w("")

    path = RESULTS_DIR / f"tts_{ts_file}.md"
    path.write_text("\n".join(lines))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TTS benchmark against Nvidia Riva NIM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--concurrency", type=str, default=None,
                        help="Comma-separated concurrency levels, e.g. '1,4,8,16,32'. "
                             "Default: range from endpoints.yaml")
    parser.add_argument("-n", "--requests", type=int, default=50,
                        help="Requests per concurrency level (default: 50)")
    parser.add_argument("--mode", type=str, default="streaming",
                        choices=["streaming", "offline"],
                        help="'streaming' (synthesize_online, matches riva_tts_perf_client --online=true) "
                             "or 'offline' (synthesize, unary RPC). Default: streaming")
    parser.add_argument("--host", type=str, default=None, help="gRPC host")
    parser.add_argument("--port", type=int, default=None, help="gRPC port")
    parser.add_argument("--ssl", action="store_true", help="Enable SSL")
    parser.add_argument("--prompt-tier", type=str, default="all",
                        choices=["all", "short", "medium", "long"],
                        help="Prompt complexity tier (default: all)")
    parser.add_argument("--voice", type=str, default="English-US.Female-1",
                        help="Voice name (default: English-US.Female-1)")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Output sample rate in Hz (default: 22050)")
    parser.add_argument("--language-code", type=str, default="en-US",
                        help="Language code (default: en-US)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to monitor (default: 0)")
    parser.add_argument("--gpu-interval", type=float, default=1.0,
                        help="GPU sampling interval in seconds (default: 1.0)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()

    tts_cfg = cfg.get("tts", {})
    host = args.host or tts_cfg.get("host", "localhost")
    port = args.port or tts_cfg.get("port", 50052)
    use_ssl = args.ssl if args.ssl else tts_cfg.get("use_ssl", False)
    levels = parse_concurrency(args.concurrency, cfg)

    logger.info(f"TTS benchmark — host={host}:{port} ssl={use_ssl} mode={args.mode}")
    logger.info(f"  voice={args.voice}  sample_rate={args.sample_rate}  language={args.language_code}")
    logger.info(f"  concurrency={levels}  requests/level={args.requests}  tier={args.prompt_tier}")

    from benchmarks.tts_benchmark import run_tts_benchmark

    t0 = time.time()
    result = run_tts_benchmark(
        concurrency_levels=levels,
        requests_per_level=args.requests,
        prompt_tier=args.prompt_tier,
        host=host,
        port=port,
        use_ssl=use_ssl,
        voice_name=args.voice,
        sample_rate_hz=args.sample_rate,
        language_code=args.language_code,
        mode=args.mode,
        gpu_index=args.gpu,
        gpu_monitor_interval=args.gpu_interval,
        progress_callback=lambda msg: logger.info(msg),
    )
    wall_time = time.time() - t0
    result["pipeline"] = "tts"
    result["timestamp"] = datetime.now().isoformat()

    json_path = save_json(result)
    md_path = generate_report(result, wall_time)
    print_summary(result)
    print()
    logger.info(f"JSON   -> {json_path}")
    logger.info(f"Report -> {md_path}")


if __name__ == "__main__":
    main()
