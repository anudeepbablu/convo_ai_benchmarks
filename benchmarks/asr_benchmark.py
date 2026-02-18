"""ASR benchmark against Nvidia Riva Parakeet 1.1B NIM via gRPC."""

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
    compute_rtf,
    compute_wer_score,
)
from benchmarks.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)

try:
    import riva.client as riva_client
    _RIVA_AVAILABLE = True
except ImportError:
    _RIVA_AVAILABLE = False
    logger.warning("nvidia-riva-client not installed; ASR benchmark unavailable.")


CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class AsrRequestResult:
    audio_file: str
    audio_duration_sec: float
    latency_sec: float
    transcript: str
    reference: str
    success: bool
    error: str = ""


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


def _transcribe_one(
    audio_path: str,
    reference: str,
    audio_duration: float,
    host: str,
    port: int,
    use_ssl: bool,
) -> AsrRequestResult:
    """Transcribe a single audio file using Riva ASR gRPC."""
    if not _RIVA_AVAILABLE:
        return AsrRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            latency_sec=0, transcript="", reference=reference,
            success=False, error="nvidia-riva-client not installed",
        )
    try:
        auth = riva_client.Auth(uri=f"{host}:{port}", use_ssl=use_ssl)
        asr = riva_client.ASRService(auth)

        with open(audio_path, "rb") as fh:
            audio_bytes = fh.read()

        try:
            with sf.SoundFile(audio_path) as sf_file:
                sample_rate = sf_file.samplerate
        except Exception:
            sample_rate = 16000

        config = riva_client.RecognitionConfig(
            encoding=riva_client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=False,
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

        return AsrRequestResult(
            audio_file=audio_path,
            audio_duration_sec=audio_duration,
            latency_sec=latency,
            transcript=transcript,
            reference=reference,
            success=True,
        )
    except Exception as e:
        return AsrRequestResult(
            audio_file=audio_path, audio_duration_sec=audio_duration,
            latency_sec=0, transcript="", reference=reference,
            success=False, error=str(e),
        )


def run_asr_benchmark(
    audio_dir: str,
    transcripts: dict[str, str],
    concurrency_levels: list[int],
    requests_per_level: int = 50,
    host: Optional[str] = None,
    port: Optional[int] = None,
    use_ssl: Optional[bool] = None,
    gpu_index: int = 0,
    gpu_monitor_interval: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run ASR benchmark across multiple concurrency levels.

    Returns a dict with:
      - results: list of ConcurrencyResult dicts
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

    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    monitor = GpuMonitor(gpu_index=gpu_index, interval=gpu_monitor_interval)
    monitor.start()

    concurrency_results: list[ConcurrencyResult] = []

    for concurrency in concurrency_levels:
        _log(f"ASR benchmark: concurrency={concurrency}, requests={requests_per_level}")

        level_samples = random.choices(samples, k=requests_per_level)

        t_start = time.perf_counter()
        request_results: list[AsrRequestResult] = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _transcribe_one,
                    s["path"], s["reference"], s["duration"],
                    host, port, use_ssl,
                ): s
                for s in level_samples
            }
            for future in as_completed(futures):
                request_results.append(future.result())

        total_duration = time.perf_counter() - t_start

        successes = [r for r in request_results if r.success]
        errors = len(request_results) - len(successes)
        latencies = [r.latency_sec for r in successes]

        latency_stats = compute_latency_stats(latencies, errors, total_duration)

        rtf_values = [
            compute_rtf(r.audio_duration_sec, r.latency_sec)
            for r in successes if r.audio_duration_sec > 0
        ]
        mean_rtf = round(sum(rtf_values) / len(rtf_values), 4) if rtf_values else 0.0

        refs = [r.reference for r in successes if r.reference]
        hyps = [r.transcript for r in successes if r.reference]
        wer_score = compute_wer_score(refs, hyps) if refs else None

        total_audio_hrs = sum(r.audio_duration_sec for r in successes) / 3600
        audio_hours_per_hour = round(total_audio_hrs / (total_duration / 3600), 2) if total_duration > 0 else 0.0

        concurrency_results.append(ConcurrencyResult(
            concurrency=concurrency,
            latency=latency_stats,
            extra={
                "mean_rtf": mean_rtf,
                "wer": wer_score,
                "audio_hours_per_hour": audio_hours_per_hour,
            },
        ))
        _log(f"  -> mean_latency={latency_stats.mean_sec:.3f}s RTF={mean_rtf:.3f} WER={wer_score}")

    gpu_summary = monitor.stop()

    return {
        "results": [r.to_dict() for r in concurrency_results],
        "gpu_summary": gpu_summary.to_dict() if gpu_summary else None,
        "config_used": {"host": host, "port": port, "use_ssl": use_ssl},
    }
