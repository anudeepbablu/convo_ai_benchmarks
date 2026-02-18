"""Streamlit UI for NIM Benchmark Suite — ASR, TTS, and LLM performance benchmarking."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "endpoints.yaml"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIM Benchmark Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_endpoints_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_result(pipeline: str, data: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{pipeline}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_history() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    history = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            data["_filename"] = f.name
            history.append(data)
        except Exception:
            pass
    return history


def make_latency_chart(results: list[dict], title: str) -> go.Figure:
    concurrencies = [r["concurrency"] for r in results]
    fig = go.Figure()
    for metric, name, color in [
        ("mean_sec", "Mean", "#4C9BE8"),
        ("p90_sec", "P90", "#F4C430"),
        ("p95_sec", "P95", "#FF8C00"),
        ("p99_sec", "P99", "#E84C4C"),
    ]:
        vals = [r.get(metric, 0) * 1000 for r in results]
        fig.add_trace(go.Scatter(
            x=concurrencies, y=vals, mode="lines+markers",
            name=name, line=dict(color=color, width=2),
        ))
    fig.update_layout(
        title=title, xaxis_title="Concurrency", yaxis_title="Latency (ms)",
        template="plotly_dark", height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_throughput_chart(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["concurrency"] for r in results],
        y=[r.get("throughput_rps", 0) for r in results],
        mode="lines+markers",
        name="Throughput",
        line=dict(color="#50C878", width=2),
        fill="tozeroy",
        fillcolor="rgba(80,200,120,0.1)",
    ))
    fig.update_layout(
        title="Throughput vs Concurrency",
        xaxis_title="Concurrency", yaxis_title="Requests/sec",
        template="plotly_dark", height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_gpu_timeline(samples: list[dict], metric: str, label: str, color: str) -> go.Figure:
    if not samples:
        return go.Figure()
    t0 = samples[0]["timestamp"]
    times = [s["timestamp"] - t0 for s in samples]
    vals = [s.get(metric, 0) for s in samples]
    # Parse hex color to rgba for fill
    try:
        r_val = int(color[1:3], 16)
        g_val = int(color[3:5], 16)
        b_val = int(color[5:7], 16)
        fill_color = f"rgba({r_val},{g_val},{b_val},0.15)"
    except Exception:
        fill_color = "rgba(128,128,128,0.15)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=vals, mode="lines",
        name=label, line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fill_color,
    ))
    fig.update_layout(
        title=f"{label} Over Time",
        xaxis_title="Time (s)", yaxis_title=label,
        template="plotly_dark", height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Session state init ────────────────────────────────────────────────────────
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "progress_log" not in st.session_state:
    st.session_state.progress_log = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "progress_queue" not in st.session_state:
    st.session_state.progress_queue = queue.Queue()
if "run_error" not in st.session_state:
    st.session_state.run_error = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NIM Benchmark Suite")
    st.markdown("---")

    pipeline = st.radio(
        "Select Pipeline",
        ["ASR (Parakeet 1.1B)", "TTS (Magpie)", "LLM (Llama 3.1-8B)"],
        key="pipeline_radio",
    )
    pipeline_key = pipeline.split(" ")[0].lower()

    cfg = load_endpoints_config()

    with st.expander("Endpoint Configuration", expanded=True):
        if pipeline_key == "asr":
            asr_host = st.text_input(
                "ASR Host",
                value=cfg.get("asr", {}).get("host", "localhost"),
                key="asr_host",
            )
            asr_port = st.number_input(
                "ASR gRPC Port",
                value=int(cfg.get("asr", {}).get("port", 50051)),
                min_value=1, max_value=65535, key="asr_port",
            )
            asr_ssl = st.checkbox(
                "Use SSL",
                value=cfg.get("asr", {}).get("use_ssl", False),
                key="asr_ssl",
            )
        elif pipeline_key == "tts":
            tts_host = st.text_input(
                "TTS Host",
                value=cfg.get("tts", {}).get("host", "localhost"),
                key="tts_host",
            )
            tts_port = st.number_input(
                "TTS gRPC Port",
                value=int(cfg.get("tts", {}).get("port", 50052)),
                min_value=1, max_value=65535, key="tts_port",
            )
            tts_ssl = st.checkbox(
                "Use SSL",
                value=cfg.get("tts", {}).get("use_ssl", False),
                key="tts_ssl",
            )
        else:
            llm_base_url = st.text_input(
                "LLM Base URL",
                value=cfg.get("llm", {}).get("base_url", "http://localhost:8000/v1"),
                key="llm_base_url",
            )
            llm_model = st.text_input(
                "Model Name",
                value=cfg.get("llm", {}).get("model", "meta/llama-3.1-8b-instruct"),
                key="llm_model",
            )

    with st.expander("Concurrency Settings", expanded=True):
        min_conc = st.number_input("Min Concurrency", min_value=1, value=1, key="min_conc")
        max_conc = st.number_input("Max Concurrency", min_value=1, value=20, key="max_conc")
        conc_step = st.number_input("Step", min_value=1, value=5, key="conc_step")
        reqs_per_level = st.number_input(
            "Requests per Level", min_value=1, value=20, key="reqs_per_level"
        )

    with st.expander("Pipeline Settings"):
        if pipeline_key == "tts":
            prompt_tier_tts = st.selectbox(
                "Prompt Tier", ["all", "short", "medium", "long"], key="tts_prompt_tier"
            )
            voice_name = st.text_input(
                "Voice Name", value="English-US.Female-1", key="tts_voice"
            )
            sample_rate = st.selectbox(
                "Sample Rate (Hz)", [22050, 16000, 44100], key="tts_sr"
            )
        elif pipeline_key == "llm":
            prompt_tier_llm = st.selectbox(
                "Prompt Tier", ["all", "short", "medium", "long"], key="llm_prompt_tier"
            )
            max_tokens = st.number_input(
                "Max Tokens", min_value=32, value=256, key="llm_max_tokens"
            )

    with st.expander("GPU Settings"):
        gpu_index = st.number_input("GPU Index", min_value=0, value=0, key="gpu_idx")
        gpu_interval = st.number_input(
            "Monitor Interval (s)", min_value=0.1, value=1.0, step=0.1, key="gpu_interval"
        )

    st.markdown("---")
    run_button = st.button(
        "▶ Run Benchmark",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )


# ── Concurrency levels ────────────────────────────────────────────────────────
_min = int(st.session_state.get("min_conc", 1))
_max = int(st.session_state.get("max_conc", 20))
_step = int(st.session_state.get("conc_step", 5))
concurrency_levels = list(range(_min, _max + 1, _step))
if not concurrency_levels:
    concurrency_levels = [_min]


# ── Benchmark thread ──────────────────────────────────────────────────────────
def run_benchmark_thread(pipeline_k: str, params: dict, progress_q: queue.Queue):
    def cb(msg: str):
        progress_q.put(("progress", msg))

    try:
        if pipeline_k == "asr":
            from benchmarks.asr_benchmark import run_asr_benchmark
            from data_prep.download_librispeech import load_transcripts, AUDIO_DIR
            transcripts = load_transcripts()
            audio_dir = str(AUDIO_DIR)
            if not Path(audio_dir).exists() or not list(Path(audio_dir).glob("*.wav")):
                progress_q.put((
                    "progress",
                    "WARNING: No audio files found. Run: python data_prep/download_librispeech.py",
                ))
            result = run_asr_benchmark(
                audio_dir=audio_dir,
                transcripts=transcripts,
                concurrency_levels=params["concurrency_levels"],
                requests_per_level=params["requests_per_level"],
                host=params["host"],
                port=params["port"],
                use_ssl=params["use_ssl"],
                gpu_index=params["gpu_index"],
                gpu_monitor_interval=params["gpu_interval"],
                progress_callback=cb,
            )

        elif pipeline_k == "tts":
            from benchmarks.tts_benchmark import run_tts_benchmark
            result = run_tts_benchmark(
                concurrency_levels=params["concurrency_levels"],
                requests_per_level=params["requests_per_level"],
                prompt_tier=params.get("prompt_tier", "all"),
                host=params["host"],
                port=params["port"],
                use_ssl=params["use_ssl"],
                voice_name=params.get("voice_name", "English-US.Female-1"),
                sample_rate_hz=params.get("sample_rate_hz", 22050),
                gpu_index=params["gpu_index"],
                gpu_monitor_interval=params["gpu_interval"],
                progress_callback=cb,
            )

        elif pipeline_k == "llm":
            from benchmarks.llm_benchmark import run_llm_benchmark
            result = run_llm_benchmark(
                concurrency_levels=params["concurrency_levels"],
                requests_per_level=params["requests_per_level"],
                prompt_tier=params.get("prompt_tier", "all"),
                base_url=params["base_url"],
                model=params["model"],
                api_key=params.get("api_key", "not-required"),
                max_tokens=params.get("max_tokens", 256),
                gpu_index=params["gpu_index"],
                gpu_monitor_interval=params["gpu_interval"],
                progress_callback=cb,
            )
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_k}")

        result["pipeline"] = pipeline_k
        result["timestamp"] = datetime.now().isoformat()
        progress_q.put(("done", result))

    except Exception as e:
        logger.exception(f"Benchmark error: {e}")
        progress_q.put(("error", str(e)))


# ── Handle run button ─────────────────────────────────────────────────────────
if run_button and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.progress_log = []
    st.session_state.current_results = None
    st.session_state.run_error = None
    st.session_state.progress_queue = queue.Queue()

    params: dict = {
        "concurrency_levels": concurrency_levels,
        "requests_per_level": int(st.session_state.get("reqs_per_level", 20)),
        "gpu_index": int(st.session_state.get("gpu_idx", 0)),
        "gpu_interval": float(st.session_state.get("gpu_interval", 1.0)),
    }

    if pipeline_key == "asr":
        params.update({
            "host": st.session_state.get("asr_host", "localhost"),
            "port": int(st.session_state.get("asr_port", 50051)),
            "use_ssl": bool(st.session_state.get("asr_ssl", False)),
        })
    elif pipeline_key == "tts":
        params.update({
            "host": st.session_state.get("tts_host", "localhost"),
            "port": int(st.session_state.get("tts_port", 50052)),
            "use_ssl": bool(st.session_state.get("tts_ssl", False)),
            "prompt_tier": st.session_state.get("tts_prompt_tier", "all"),
            "voice_name": st.session_state.get("tts_voice", "English-US.Female-1"),
            "sample_rate_hz": int(st.session_state.get("tts_sr", 22050)),
        })
    else:
        params.update({
            "base_url": st.session_state.get("llm_base_url", "http://localhost:8000/v1"),
            "model": st.session_state.get("llm_model", "meta/llama-3.1-8b-instruct"),
            "api_key": "not-required",
            "prompt_tier": st.session_state.get("llm_prompt_tier", "all"),
            "max_tokens": int(st.session_state.get("llm_max_tokens", 256)),
        })

    t = threading.Thread(
        target=run_benchmark_thread,
        args=(pipeline_key, params, st.session_state.progress_queue),
        daemon=True,
    )
    t.start()


# ── Drain progress queue ──────────────────────────────────────────────────────
if st.session_state.is_running:
    q = st.session_state.progress_queue
    while not q.empty():
        try:
            kind, payload = q.get_nowait()
            if kind == "progress":
                st.session_state.progress_log.append(payload)
            elif kind == "done":
                st.session_state.current_results = payload
                save_result(pipeline_key, payload)
                st.session_state.is_running = False
                st.session_state.progress_log.append("Benchmark complete.")
            elif kind == "error":
                st.session_state.run_error = payload
                st.session_state.is_running = False
                st.session_state.progress_log.append(f"Error: {payload}")
        except queue.Empty:
            break
    if st.session_state.is_running:
        time.sleep(0.5)
        st.rerun()


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_gpu, tab_history = st.tabs(
    ["▶ Run", "📊 Results", "GPU Monitor", "📁 History"]
)


# ── Run tab ───────────────────────────────────────────────────────────────────
with tab_run:
    st.subheader(f"Pipeline: {pipeline}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Concurrency Levels", len(concurrency_levels))
    col2.metric("Range", f"{concurrency_levels[0]}–{concurrency_levels[-1]}")
    col3.metric("Requests/Level", st.session_state.get("reqs_per_level", 20))

    st.markdown("**Concurrency sweep:** " + ", ".join(str(c) for c in concurrency_levels))

    if st.session_state.is_running:
        st.info("Benchmark in progress...")

    if st.session_state.progress_log:
        log_text = "\n".join(st.session_state.progress_log[-60:])
        st.text_area("Progress Log", value=log_text, height=250, key="progress_display")

    if st.session_state.run_error:
        st.error(f"Benchmark failed: {st.session_state.run_error}")

    if st.session_state.current_results and not st.session_state.is_running:
        res = st.session_state.current_results
        results_list = res.get("results", [])
        if results_list:
            st.success("Benchmark completed successfully!")
            c1, c2, c3, c4 = st.columns(4)
            last = results_list[-1]
            c1.metric("Peak Concurrency", last.get("concurrency", "—"))
            c2.metric("Mean Latency", f"{last.get('mean_sec', 0) * 1000:.1f} ms")
            c3.metric("Throughput", f"{last.get('throughput_rps', 0):.1f} req/s")
            c4.metric("Error Rate", f"{last.get('error_rate', 0) * 100:.1f}%")


# ── Results tab ───────────────────────────────────────────────────────────────
with tab_results:
    res = st.session_state.current_results
    if res is None:
        st.info("No results yet. Run a benchmark from the sidebar.")
    else:
        results_list = res.get("results", [])
        if not results_list:
            st.warning("Benchmark completed but no results recorded.")
        else:
            pl = res.get("pipeline", pipeline_key)

            st.plotly_chart(
                make_latency_chart(results_list, "Latency vs Concurrency"),
                use_container_width=True,
            )

            col_a, col_b = st.columns(2)
            col_a.plotly_chart(make_throughput_chart(results_list), use_container_width=True)

            if pl == "asr":
                rtf_fig = go.Figure()
                rtf_fig.add_trace(go.Scatter(
                    x=[r["concurrency"] for r in results_list],
                    y=[r.get("mean_rtf", 0) for r in results_list],
                    mode="lines+markers", name="RTF",
                    line=dict(color="#BA55D3", width=2),
                ))
                rtf_fig.update_layout(
                    title="Real-Time Factor vs Concurrency",
                    xaxis_title="Concurrency", yaxis_title="RTF",
                    template="plotly_dark", height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                col_b.plotly_chart(rtf_fig, use_container_width=True)

                wer_vals = [r.get("wer") for r in results_list if r.get("wer") is not None]
                if wer_vals:
                    st.metric("Word Error Rate (WER)", f"{wer_vals[-1] * 100:.2f}%")

            elif pl == "tts":
                ttfb_fig = go.Figure()
                ttfb_fig.add_trace(go.Scatter(
                    x=[r["concurrency"] for r in results_list],
                    y=[r.get("mean_time_to_first_byte_sec", 0) * 1000 for r in results_list],
                    mode="lines+markers", name="TTFB",
                    line=dict(color="#20B2AA", width=2),
                ))
                ttfb_fig.update_layout(
                    title="Time-to-First-Byte vs Concurrency",
                    xaxis_title="Concurrency", yaxis_title="TTFB (ms)",
                    template="plotly_dark", height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                col_b.plotly_chart(ttfb_fig, use_container_width=True)

            elif pl == "llm":
                ttft_fig = go.Figure()
                ttft_fig.add_trace(go.Scatter(
                    x=[r["concurrency"] for r in results_list],
                    y=[r.get("mean_ttft_sec", 0) * 1000 for r in results_list],
                    mode="lines+markers", name="Mean TTFT",
                    line=dict(color="#FF6347", width=2),
                ))
                ttft_fig.add_trace(go.Scatter(
                    x=[r["concurrency"] for r in results_list],
                    y=[r.get("p90_ttft_sec", 0) * 1000 for r in results_list],
                    mode="lines+markers", name="P90 TTFT",
                    line=dict(color="#FFA07A", width=2, dash="dash"),
                ))
                ttft_fig.update_layout(
                    title="Time-to-First-Token vs Concurrency",
                    xaxis_title="Concurrency", yaxis_title="TTFT (ms)",
                    template="plotly_dark", height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                col_b.plotly_chart(ttft_fig, use_container_width=True)

                tps_fig = go.Figure()
                tps_fig.add_trace(go.Bar(
                    x=[r["concurrency"] for r in results_list],
                    y=[r.get("mean_tokens_per_sec", 0) for r in results_list],
                    name="Tokens/sec",
                    marker_color="#6495ED",
                ))
                tps_fig.update_layout(
                    title="Tokens/sec vs Concurrency",
                    xaxis_title="Concurrency", yaxis_title="Tokens/sec",
                    template="plotly_dark", height=320,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(tps_fig, use_container_width=True)

            st.subheader("Raw Results")
            df = pd.DataFrame(results_list)
            for col in list(df.columns):
                if col.endswith("_sec") and col not in ("total_duration_sec",):
                    df[col] = (df[col] * 1000).round(2)
                    df.rename(columns={col: col.replace("_sec", "_ms")}, inplace=True)
            st.dataframe(df, use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{pl}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            col_dl2.download_button(
                "Download JSON",
                data=json.dumps(res, indent=2, default=str),
                file_name=f"{pl}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


# ── GPU Monitor tab ───────────────────────────────────────────────────────────
with tab_gpu:
    res = st.session_state.current_results
    gpu = res.get("gpu_summary") if res else None

    if gpu is None:
        st.info("GPU monitoring data will appear here after the first completed benchmark run.")
    else:
        st.subheader(f"GPU: {gpu.get('gpu_name', 'Unknown')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total VRAM", f"{gpu.get('memory_total_mb', 0) / 1024:.1f} GB")
        c2.metric("Samples", gpu.get("sample_count", 0))
        c3.metric("Duration", f"{gpu.get('duration_sec', 0):.1f}s")
        c4.metric("Power Limit", f"{gpu.get('power_limit_w', 0):.0f}W")

        st.markdown("---")
        st.subheader("Peak & Mean Statistics")
        stats_data = {
            "Metric": [
                "GPU Utilization (%)", "Memory Used (MB)",
                "Temperature (C)", "Power Draw (W)",
            ],
            "Peak": [
                gpu.get("peak_utilization_gpu_pct", 0),
                round(gpu.get("peak_memory_used_mb", 0), 1),
                gpu.get("peak_temperature_c", 0),
                gpu.get("peak_power_draw_w", 0),
            ],
            "Mean": [
                gpu.get("mean_utilization_gpu_pct", 0),
                round(gpu.get("mean_memory_used_mb", 0), 1),
                gpu.get("mean_temperature_c", 0),
                gpu.get("mean_power_draw_w", 0),
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        samples = gpu.get("samples", [])
        if samples:
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            col_g1.plotly_chart(
                make_gpu_timeline(samples, "utilization_gpu_pct", "GPU Utilization (%)", "#4C9BE8"),
                use_container_width=True,
            )
            col_g2.plotly_chart(
                make_gpu_timeline(samples, "memory_used_mb", "VRAM Used (MB)", "#50C878"),
                use_container_width=True,
            )
            col_g3, col_g4 = st.columns(2)
            col_g3.plotly_chart(
                make_gpu_timeline(samples, "temperature_c", "Temperature (C)", "#FF8C00"),
                use_container_width=True,
            )
            col_g4.plotly_chart(
                make_gpu_timeline(samples, "power_draw_w", "Power Draw (W)", "#E84C4C"),
                use_container_width=True,
            )


# ── History tab ───────────────────────────────────────────────────────────────
with tab_history:
    history = load_history()
    if not history:
        st.info("No saved benchmark runs yet. Results are saved automatically after each run.")
    else:
        st.subheader(f"{len(history)} Saved Run(s)")
        for entry in history[:20]:
            pl = entry.get("pipeline", "unknown").upper()
            ts = entry.get("timestamp", entry.get("_filename", ""))
            results_list = entry.get("results", [])
            if results_list:
                last = results_list[-1]
                mean_ms = last.get("mean_sec", 0) * 1000
                tps = last.get("throughput_rps", 0)
                label = (
                    f"**{pl}** — {str(ts)[:19]}  |  "
                    f"Mean: {mean_ms:.1f}ms  |  "
                    f"Throughput: {tps:.1f} req/s  |  "
                    f"Max concurrency: {last.get('concurrency', '?')}"
                )
            else:
                label = f"**{pl}** — {str(ts)[:19]} (no results)"

            with st.expander(label):
                cfg_used = entry.get("config_used", {})
                st.json(cfg_used)
                if results_list:
                    df_h = pd.DataFrame(results_list)
                    for col in list(df_h.columns):
                        if col.endswith("_sec") and col not in ("total_duration_sec",):
                            df_h[col] = (df_h[col] * 1000).round(2)
                            df_h.rename(
                                columns={col: col.replace("_sec", "_ms")}, inplace=True
                            )
                    st.dataframe(df_h, use_container_width=True)
                    st.plotly_chart(
                        make_latency_chart(results_list, f"{pl} Latency"),
                        use_container_width=True,
                    )
