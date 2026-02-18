# convo_ai_benchmarks

Performance benchmarking suite for Nvidia NIM inference services: ASR (Parakeet 1.1B), TTS (Magpie), and LLM (Llama 3.1 8B). Measures latency, throughput, concurrency scaling, and GPU utilization via an interactive Streamlit UI.

---

## Architecture

Each NIM runs in its own Docker container. Benchmarks are run independently — one pipeline at a time — against a locally running container.

```
convo_ai_benchmarks/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env.example
├── config/
│   └── endpoints.yaml              # Service host/port config + benchmark defaults
├── benchmarks/
│   ├── gpu_monitor.py              # pynvml-based GPU metrics (background thread)
│   ├── metrics.py                  # Shared stats: latency percentiles, RTF, WER
│   ├── asr_benchmark.py            # Riva Parakeet ASR via gRPC
│   ├── tts_benchmark.py            # Riva Magpie TTS via gRPC (streaming)
│   └── llm_benchmark.py            # LLaMA 3.1 NIM via OpenAI-compatible API
├── data_prep/
│   └── download_librispeech.py     # Download LibriSpeech test-clean (1000 samples)
├── data/prompts/
│   ├── tts_prompts.json            # Short / medium / long TTS prompts
│   └── llm_prompts.json            # Short / medium / long LLM prompts
└── docker/
    ├── asr/run.sh                  # Pull + start Parakeet NIM (gRPC :50051)
    ├── tts/run.sh                  # Pull + start Magpie TTS NIM (gRPC :50052)
    └── llm/run.sh                  # Pull + start LLaMA 3.1 NIM (HTTP :8000)
```

---

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Python 3.10+
- NGC API key (for pulling NIM containers)
- Single GPU recommended (run one NIM at a time)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/anudeepbablu/convo_ai_benchmarks.git
cd convo_ai_benchmarks
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your NGC_API_KEY
```

### 4. (ASR only) Download LibriSpeech test-clean

```bash
python -m data_prep.download_librispeech
```

This downloads 1000 audio samples from LibriSpeech test-clean and saves them to `data/audio/` along with `data/transcripts.json` for WER computation.

---

## Running a NIM Container

Start the NIM for the pipeline you want to benchmark. **Only one NIM at a time** (single GPU).

```bash
# ASR — Parakeet 1.1B CTC
bash docker/asr/run.sh

# TTS — Magpie TTS
bash docker/tts/run.sh

# LLM — LLaMA 3.1 8B Instruct
bash docker/llm/run.sh
```

Each script pulls the container, starts it with GPU access, and waits for the health check to pass before returning.

---

## Running the Benchmark UI

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### UI Workflow

1. **Select pipeline** — ASR, TTS, or LLM — from the sidebar
2. **Configure endpoint** — host/port (defaults match the docker scripts)
3. **Set concurrency sweep** — min, max, step, and requests per level
4. **Click Run Benchmark**
5. View results in the **Results**, **GPU Monitor**, and **History** tabs

---

## Metrics

| Pipeline | Metrics |
|----------|---------|
| **ASR** | Latency (mean/p90/p95/p99), RTF, WER, audio hours/hour |
| **TTS** | Latency, TTFB (time to first byte), RTF, chars/sec |
| **LLM** | Latency, TTFT (time to first token), tokens/sec, total tokens |
| **All** | Throughput (req/s), error rate, GPU utilization, VRAM, temperature, power |

---

## Configuration

Edit `config/endpoints.yaml` to change default endpoints or benchmark parameters:

```yaml
asr:
  host: localhost
  port: 50051

tts:
  host: localhost
  port: 50052

llm:
  base_url: "http://localhost:8000/v1"
  model: "meta/llama-3.1-8b-instruct"

benchmark:
  default_concurrency_min: 1
  default_concurrency_max: 50
  default_concurrency_step: 5
  default_requests_per_level: 50
```

---

## NIM Images Used

| Service | Image |
|---------|-------|
| ASR | `nvcr.io/nim/nvidia/parakeet-ctc-1.1b-asr:latest` |
| TTS | `nvcr.io/nim/nvidia/magpie-tts:latest` |
| LLM | `nvcr.io/nim/meta/llama-3.1-8b-instruct:latest` |

---

## Results

Benchmark results are auto-saved to `results/<pipeline>_<timestamp>.json` after each run. The History tab in the UI shows the last 20 runs with side-by-side comparison. CSV and JSON downloads are available from the Results tab.
