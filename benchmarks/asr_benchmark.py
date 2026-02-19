"""ASR benchmark against Nvidia Riva NIM using the official nvidia-riva-client library.

Supports two modes matching NVIDIA's recommended methodology:
  - streaming: AudioChunkFileIterator + streaming_response_generator (matches riva_streaming_asr_client)
  - offline:   offline_recognize with full audio (matches riva_asr_client)

References:
  https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-performance.html
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf
import yaml

from benchmarks.metrics import (
    ConcurrencyResult,
    compute_latency_stats,
    compute_wer_score,
    percentile,
)
from benchmarks.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)

try:
    import riva.client as riva_client
    from riva.client import (
        ASRService,
        AudioChunkFileIterator,
        AudioEncoding,
        Auth,
        RecognitionConfig,
        StreamingRecognitionConfig,
    )
    _RIVA_AVAILABLE = True
except ImportError:
    _RIVA_AVAILABLE = False
    logger.warning("nvidia-riva-client not installed; ASR benchmark unavailable.")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_audio_duration(path: str) -> float:
    try:
        with sf.SoundFile(path) as f:
            return len(f) / f.samplerate
    except Exception:
        return 0.0


def load_audio_samples(audio_dir: Path, transcripts: dict[str, str], n: int = 1000) -> list[dict]:
    """Load up to n audio samples with their transcripts."""
    wav_files = list(audio_dir.glob("**/*.wav")) + list(audio_dir.glob("**/*.flac"))
    if not wav_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    if len(wav_files) > n:
        wav_files = random.sample(wav_files, n)
    samples = []
    for f in wav_files:
        key = f.stem
        ref = transcripts.get(key, "")
        samples.append({"path": str(f), "reference": ref, "duration": get_audio_duration(str(f))})
    return samples


# ── Streaming mode (matches riva_streaming_asr_client) ───────────────────────

@dataclass
class StreamingRequestResult:
    audio_file: str
    audio_duration_sec: float
    total_latency_sec: float
    audio_streaming_sec: float        # time spent sending chunks (includes real-time pacing sleeps)
    server_compute_after_audio_sec: float  # time from last chunk sent to final result
    time_to_first_response_sec: float
    final_latency_sec: float
    transcript: str
    reference: str
    success: bool
    error: str = ""


def _streaming_recognize_one(
    audio_path: str,
    reference: str,
    audio_duration: float,
    auth: Auth,
    chunk_duration_ms: float,
    simulate_realtime: bool,
) -> StreamingRequestResult:
    """Stream one audio file in chunks, matching riva_streaming_asr_client behavior."""
    if not _RIVA_AVAILABLE:
        return StreamingRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            total_latency_sec=0, audio_streaming_sec=0,
            server_compute_after_audio_sec=0,
            time_to_first_response_sec=0, final_latency_sec=0,
            transcript="", reference=reference, success=False,
            error="nvidia-riva-client not installed",
        )
    try:
        asr = ASRService(auth)

        # Read sample rate from file
        try:
            with sf.SoundFile(audio_path) as sf_file:
                sample_rate = sf_file.samplerate
        except Exception:
            sample_rate = 16000

        config = RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
            sample_rate_hertz=sample_rate,
        )
        streaming_config = StreamingRecognitionConfig(
            config=config,
            interim_results=False,
        )

        # Chunk size in frames: chunk_duration_ms * sample_rate / 1000
        chunk_n_frames = int(chunk_duration_ms * sample_rate / 1000)

        # Track when the last chunk is sent via a custom callback wrapper
        last_chunk_sent_time = None

        if simulate_realtime:
            def _delay_and_track(audio_chunk: bytes, delay_sec: float):
                nonlocal last_chunk_sent_time
                time.sleep(delay_sec)
                last_chunk_sent_time = time.perf_counter()
            delay_callback = _delay_and_track
        else:
            def _track_only(audio_chunk: bytes, delay_sec: float):
                nonlocal last_chunk_sent_time
                last_chunk_sent_time = time.perf_counter()
            delay_callback = _track_only

        audio_iter = AudioChunkFileIterator(
            input_file=audio_path,
            chunk_n_frames=chunk_n_frames,
            delay_callback=delay_callback,
        )

        t_start = time.perf_counter()
        first_response_time: Optional[float] = None
        last_final_time: Optional[float] = None
        transcript = ""

        responses = asr.streaming_response_generator(audio_iter, streaming_config)
        for response in responses:
            now = time.perf_counter()
            if first_response_time is None and response.results:
                first_response_time = now - t_start

            for result in response.results:
                if result.is_final:
                    last_final_time = now - t_start
                    if result.alternatives:
                        transcript += result.alternatives[0].transcript + " "

        t_end = time.perf_counter()
        total_latency = t_end - t_start
        transcript = transcript.strip()

        # Compute timing breakdown
        audio_streaming = (last_chunk_sent_time - t_start) if last_chunk_sent_time else total_latency
        server_tail = (t_end - last_chunk_sent_time) if last_chunk_sent_time else 0.0

        return StreamingRequestResult(
            audio_file=audio_path,
            audio_duration_sec=audio_duration,
            total_latency_sec=total_latency,
            audio_streaming_sec=audio_streaming,
            server_compute_after_audio_sec=server_tail,
            time_to_first_response_sec=first_response_time if first_response_time is not None else total_latency,
            final_latency_sec=last_final_time if last_final_time is not None else total_latency,
            transcript=transcript,
            reference=reference,
            success=True,
        )
    except Exception as e:
        return StreamingRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            total_latency_sec=0, audio_streaming_sec=0,
            server_compute_after_audio_sec=0,
            time_to_first_response_sec=0, final_latency_sec=0,
            transcript="", reference=reference, success=False, error=str(e),
        )


# ── Offline mode (matches riva_asr_client) ───────────────────────────────────

@dataclass
class OfflineRequestResult:
    audio_file: str
    audio_duration_sec: float
    latency_sec: float
    transcript: str
    reference: str
    success: bool
    error: str = ""


def _offline_recognize_one(
    audio_path: str,
    reference: str,
    audio_duration: float,
    auth: Auth,
) -> OfflineRequestResult:
    """Recognize a full audio file in one shot, matching riva_asr_client behavior."""
    if not _RIVA_AVAILABLE:
        return OfflineRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            latency_sec=0, transcript="", reference=reference,
            success=False, error="nvidia-riva-client not installed",
        )
    try:
        asr = ASRService(auth)

        with open(audio_path, "rb") as fh:
            audio_bytes = fh.read()

        try:
            with sf.SoundFile(audio_path) as sf_file:
                sample_rate = sf_file.samplerate
        except Exception:
            sample_rate = 16000

        config = RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
            sample_rate_hertz=sample_rate,
        )

        t0 = time.perf_counter()
        response = asr.offline_recognize(audio_bytes, config)
        latency = time.perf_counter() - t0

        transcript = ""
        if response.results:
            for result in response.results:
                if result.alternatives:
                    transcript += result.alternatives[0].transcript + " "
        transcript = transcript.strip()

        return OfflineRequestResult(
            audio_file=audio_path,
            audio_duration_sec=audio_duration,
            latency_sec=latency,
            transcript=transcript,
            reference=reference,
            success=True,
        )
    except Exception as e:
        return OfflineRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            latency_sec=0, transcript="", reference=reference,
            success=False, error=str(e),
        )


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_asr_benchmark(
    audio_dir: str,
    transcripts: dict[str, str],
    concurrency_levels: list[int],
    requests_per_level: int = 50,
    host: Optional[str] = None,
    port: Optional[int] = None,
    use_ssl: Optional[bool] = None,
    mode: str = "streaming",
    chunk_duration_ms: float = 800,
    simulate_realtime: bool = True,
    gpu_index: int = 0,
    gpu_monitor_interval: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run ASR benchmark across multiple concurrency levels.

    Args:
        mode: "streaming" (chunked gRPC streaming) or "offline" (full-audio batch).
        chunk_duration_ms: Audio chunk size for streaming mode (default: 800ms).
        simulate_realtime: Whether to pace audio chunks at real-time speed (streaming only).

    Returns dict with:
      - results: list of per-concurrency result dicts
      - gpu_summary: GpuSummary dict or None
      - config_used: dict
    """
    cfg = load_config()
    asr_cfg = cfg.get("asr", {})
    host = host or asr_cfg.get("host", "localhost")
    port = port or asr_cfg.get("port", 50051)
    use_ssl = use_ssl if use_ssl is not None else asr_cfg.get("use_ssl", False)

    audio_path = Path(audio_dir)
    samples = load_audio_samples(audio_path, transcripts, n=1000)
    if not samples:
        raise ValueError("No audio samples available.")

    # Reuse a single Auth across all requests (connection sharing)
    auth = Auth(uri=f"{host}:{port}", use_ssl=use_ssl)

    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _log(f"ASR benchmark mode={mode} chunk={chunk_duration_ms}ms realtime={simulate_realtime}")

    monitor = GpuMonitor(gpu_index=gpu_index, interval=gpu_monitor_interval)
    monitor.start()

    concurrency_results: list[ConcurrencyResult] = []

    for concurrency in concurrency_levels:
        _log(f"ASR {mode}: concurrency={concurrency}, requests={requests_per_level}")
        level_samples = random.choices(samples, k=requests_per_level)

        t_start = time.perf_counter()

        if mode == "streaming":
            results_list: list[StreamingRequestResult] = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _streaming_recognize_one,
                        s["path"], s["reference"], s["duration"],
                        auth, chunk_duration_ms, simulate_realtime,
                    ): s
                    for s in level_samples
                }
                for future in as_completed(futures):
                    results_list.append(future.result())

            total_duration = time.perf_counter() - t_start
            successes = [r for r in results_list if r.success]
            errors = len(results_list) - len(successes)

            latencies = [r.total_latency_sec for r in successes]
            latency_stats = compute_latency_stats(latencies, errors, total_duration)

            # RTFX = total audio duration / total compute time  (NVIDIA convention, higher = better)
            total_audio_sec = sum(r.audio_duration_sec for r in successes)
            rtfx = round(total_audio_sec / total_duration, 2) if total_duration > 0 else 0.0

            # Timing breakdown
            audio_streaming_times = [r.audio_streaming_sec for r in successes]
            server_tail_times = [r.server_compute_after_audio_sec for r in successes]
            mean_audio_streaming = round(sum(audio_streaming_times) / len(audio_streaming_times), 4) if audio_streaming_times else 0.0
            mean_server_tail = round(sum(server_tail_times) / len(server_tail_times), 4) if server_tail_times else 0.0
            p99_server_tail = round(percentile(server_tail_times, 99), 4) if server_tail_times else 0.0

            # Mean audio duration for reference
            mean_audio_dur = round(sum(r.audio_duration_sec for r in successes) / len(successes), 4) if successes else 0.0

            # Time to first response
            ttfrs = [r.time_to_first_response_sec for r in successes]
            mean_ttfr = round(sum(ttfrs) / len(ttfrs), 4) if ttfrs else 0.0

            # WER
            refs = [r.reference for r in successes if r.reference]
            hyps = [r.transcript for r in successes if r.reference]
            wer_score = compute_wer_score(refs, hyps) if refs else None

            concurrency_results.append(ConcurrencyResult(
                concurrency=concurrency,
                latency=latency_stats,
                extra={
                    "rtfx": rtfx,
                    "mean_audio_duration_sec": mean_audio_dur,
                    "mean_audio_streaming_sec": mean_audio_streaming,
                    "mean_server_compute_after_audio_sec": mean_server_tail,
                    "p99_server_compute_after_audio_sec": p99_server_tail,
                    "mean_time_to_first_response_sec": mean_ttfr,
                    "wer": wer_score,
                    "total_audio_sec": round(total_audio_sec, 2),
                },
            ))
            _log(f"  -> RTFX={rtfx}  audio={mean_audio_dur:.1f}s  "
                 f"stream={mean_audio_streaming:.1f}s  server_tail={mean_server_tail:.3f}s  WER={wer_score}")

        else:  # offline
            results_list_off: list[OfflineRequestResult] = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _offline_recognize_one,
                        s["path"], s["reference"], s["duration"], auth,
                    ): s
                    for s in level_samples
                }
                for future in as_completed(futures):
                    results_list_off.append(future.result())

            total_duration = time.perf_counter() - t_start
            successes_off = [r for r in results_list_off if r.success]
            errors = len(results_list_off) - len(successes_off)

            latencies = [r.latency_sec for r in successes_off]
            latency_stats = compute_latency_stats(latencies, errors, total_duration)

            total_audio_sec = sum(r.audio_duration_sec for r in successes_off)
            rtfx = round(total_audio_sec / total_duration, 2) if total_duration > 0 else 0.0

            refs = [r.reference for r in successes_off if r.reference]
            hyps = [r.transcript for r in successes_off if r.reference]
            wer_score = compute_wer_score(refs, hyps) if refs else None

            concurrency_results.append(ConcurrencyResult(
                concurrency=concurrency,
                latency=latency_stats,
                extra={
                    "rtfx": rtfx,
                    "wer": wer_score,
                    "total_audio_sec": round(total_audio_sec, 2),
                },
            ))
            _log(f"  -> RTFX={rtfx}  mean_latency={latency_stats.mean_sec:.3f}s  WER={wer_score}")

    gpu_summary = monitor.stop()

    return {
        "results": [r.to_dict() for r in concurrency_results],
        "gpu_summary": gpu_summary.to_dict() if gpu_summary else None,
        "config_used": {
            "host": host, "port": port, "use_ssl": use_ssl,
            "mode": mode, "chunk_duration_ms": chunk_duration_ms,
            "simulate_realtime": simulate_realtime,
        },
    }
