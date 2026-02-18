#!/usr/bin/env python3
"""Download LibriSpeech test-clean split and prepare audio files for ASR benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import io

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
TRANSCRIPTS_FILE = Path(__file__).parent.parent / "data" / "transcripts.json"
MAX_SAMPLES = 1000


def download_and_prepare(force: bool = False) -> None:
    existing_wavs = list(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []
    if TRANSCRIPTS_FILE.exists() and existing_wavs and not force:
        logger.info(
            f"Data already exists: {len(existing_wavs)} WAV files, transcripts.json. "
            f"Use --force to re-download."
        )
        return

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading LibriSpeech test-clean from HuggingFace datasets...")
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        logger.error("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )
    # Disable auto-decoding to avoid needing torchcodec; we decode bytes with soundfile
    dataset = dataset.cast_column("audio", Audio(decode=False))

    logger.info(f"Streaming test-clean split. Will save up to {MAX_SAMPLES} WAV files...")

    transcripts: dict[str, str] = {}

    for i, item in enumerate(tqdm(dataset, total=MAX_SAMPLES, desc="Saving audio")):
        if i >= MAX_SAMPLES:
            break

        audio = item["audio"]
        raw_bytes = audio.get("bytes") or open(audio["path"], "rb").read()
        array, sample_rate = sf.read(io.BytesIO(raw_bytes))
        array = np.array(array, dtype=np.float32)
        text = item["text"].strip()

        utterance_id = item.get("id", str(i))
        safe_id = str(utterance_id).replace("/", "_").replace(" ", "_")
        filename = f"{safe_id}.wav"
        wav_path = AUDIO_DIR / filename

        sf.write(str(wav_path), array, sample_rate, subtype="PCM_16")
        transcripts[wav_path.stem] = text

    with open(TRANSCRIPTS_FILE, "w") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(transcripts)} WAV files to {AUDIO_DIR}")
    logger.info(f"Saved transcripts to {TRANSCRIPTS_FILE}")


def load_transcripts() -> dict[str, str]:
    """Load the transcripts mapping. Returns empty dict if not downloaded yet."""
    if not TRANSCRIPTS_FILE.exists():
        return {}
    with open(TRANSCRIPTS_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LibriSpeech test-clean for ASR benchmarking")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()
    download_and_prepare(force=args.force)
