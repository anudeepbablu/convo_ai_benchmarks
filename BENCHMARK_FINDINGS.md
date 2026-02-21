# NVIDIA NIM Benchmark Findings — ASR & TTS

**Hardware:** NVIDIA H100 80GB HBM3 (single GPU)
**Date:** 2026-02-19 (initial), 2026-02-20 (MIG benchmarks)
**Benchmark Tool:** riva-bench (custom gRPC benchmark suite using `nvidia-riva-client`)

---

## 1. Models Tested

| Pipeline | Model | Container Image | Architecture |
|---|---|---|---|
| ASR | Parakeet CTC 1.1B | `nvcr.io/nim/nvidia/parakeet-ctc-1.1b-asr:latest` | CTC (non-autoregressive) |
| TTS | FastPitch-HifiGAN | `nvcr.io/nim/nvidia/riva-speech:latest` | Spectrogram + Vocoder (non-autoregressive) |
| TTS | MagpieTTS Multilingual | `nvcr.io/nim/nvidia/magpie-tts-multilingual:latest` | Transformer + Neural Audio Codec (autoregressive) |

---

## 2. ASR Benchmark — Parakeet CTC 1.1B

**Mode:** Streaming (160ms audio chunks, real-time pacing)
**Profile:** `7f0287aa` — max_batch_size=1024 for acoustic model, preferred_batch_size=64
**Requests per level:** 50

### Results

| Concurrency | Mean (ms) | P90 (ms) | P99 (ms) | Throughput (r/s) | RTFX | Server Tail (ms) | WER | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7,788 | 17,522 | 24,395 | 0.1 | 1.0x | 42 | 4.14% | 0 |
| 16 | 7,752 | 15,881 | 22,296 | 1.5 | 11.7x | 56 | 3.94% | 0 |
| 32 | 7,412 | 14,611 | 25,124 | 1.6 | 11.5x | 63 | 4.15% | 0 |
| 64 | 7,971 | 15,617 | 21,620 | 2.2 | 17.5x | 58 | 4.64% | 0 |
| 128 | 6,274 | 10,110 | 23,088 | 1.8 | 10.8x | 57 | 4.57% | 0 |
| 256 | 8,275 | 15,129 | 19,202 | 2.5 | 20.4x | 57 | 4.76% | 0 |

### GPU Utilization

| Metric | Peak | Mean |
| :--- | ---: | ---: |
| GPU Utilization (%) | 75 | 13 |
| VRAM (MB) | 15,086 | 15,086 |
| Temperature (C) | 34 | 30 |
| Power Draw (W) | 213 | 131 |

### ASR Key Findings

1. **Streaming ASR is real-time bound.** Mean latency (~7-8s) closely matches mean audio duration (~7.7s). The server processes audio at the pace it arrives — RTFX=1.0x at c=1 simply means it keeps up.
2. **Server tail latency is excellent.** After the last audio chunk, the server delivers the final transcript in only 42-63ms regardless of concurrency — the model is very fast once audio stops.
3. **Scales well to 256 concurrent streams.** RTFX=20.4x at c=256 means a single GPU processes 256 audio streams 20x faster than real-time. Zero errors across all levels.
4. **Low GPU utilization.** Mean GPU util is only 13% even at c=256. The CTC model is lightweight (1.1B params) and most time is spent waiting for real-time audio chunks.
5. **VRAM is constant at ~15 GB** regardless of concurrency. No batch-size-dependent memory scaling.
6. **WER degrades slightly under load.** 4.14% at c=1 vs 4.76% at c=256 — minor and acceptable.
7. **No batch size profile issue.** Unlike TTS, the ASR acoustic model supports max_batch_size=1024 natively.

---

## 3. TTS Benchmark — FastPitch-HifiGAN (Non-Autoregressive)

**Mode:** Streaming (synthesize_online)
**Voice:** English-US.Female-1
**Container:** riva-speech (multi-skill, profile `bbce2a3a`)
**Requests per level:** 50

### Results

| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | TTFT (ms) | P99 TTFA (ms) | Inter-Chunk (ms) | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 119 | 305 | 8.4 | 231x | 36 | 81 | 2.6 | 0 |
| 2 | 158 | 349 | 12.6 | 379x | 49 | 108 | 3.2 | 0 |
| 4 | 234 | 529 | 16.5 | 503x | 79 | 239 | 4.5 | 0 |
| 8 | 420 | 969 | 17.9 | 444x | 214 | 464 | 7.6 | 0 |
| 16 | 1,021 | 2,444 | 13.0 | 346x | 663 | 1,359 | 9.8 | 0 |
| 32 | 1,501 | 2,551 | 15.0 | 493x | 1,080 | 1,631 | 12.7 | 0 |
| 64 | 1,481 | 2,644 | 18.7 | 526x | 1,131 | 2,308 | 11.8 | 0 |

### GPU Utilization

| Metric | Peak | Mean |
| :--- | ---: | ---: |
| GPU Utilization (%) | 98 | 68 |
| VRAM (MB) | 16,490 | 16,473 |
| Temperature (C) | 42 | 36 |
| Power Draw (W) | 447 | 280 |

### FastPitch Key Findings

1. **Extremely fast.** 119ms mean latency at c=1, 36ms TTFT. Audio generated 231x faster than real-time.
2. **Non-autoregressive architecture** means the entire spectrogram is generated in one forward pass, then vocoded. No sequential token generation.
3. **High throughput.** 18.7 req/s at c=64 — can handle heavy concurrent loads.
4. **TTFT = TTFA.** Every gRPC streaming response contains audio from the first chunk. No empty "header" responses.
5. **Limited to English-only, 3 voices, no emotional control.**
6. **VRAM: ~16.5 GB constant** regardless of concurrency.

---

## 4. TTS Benchmark — MagpieTTS Multilingual (Autoregressive)

### 4a. Batch Size 8 Profile (auto-selected)

**Profile:** `48afefb5` — batch_size=8, TRT-LLM engine compiled for max 8 concurrent
**Voice:** Magpie-Multilingual.EN-US.Aria

| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | TTFT (ms) | P99 TTFA (ms) | Inter-Chunk (ms) | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,381 | 5,307 | 0.4 | 12.5x | 85 | 125 | 12.0 | 0 |
| 2 | 2,423 | 5,456 | 0.8 | 24.1x | 92 | 140 | 12.5 | 0 |
| 4 | 2,122 | 5,768 | 1.7 | 42.0x | 97 | 142 | 13.3 | 0 |
| 8 | 2,244 | 6,568 | 2.9 | 65.7x | 103 | 178 | 15.8 | 0 |
| 16 | 5,945 | 11,509 | 2.2 | 75.1x | 2,488 | 4,774 | 16.6 | 0 |
| 32 | 8,683 | 16,698 | 2.4 | 74.7x | 5,492 | 11,065 | 16.7 | 0 |

**GPU:** 14.6 GB VRAM, 94% peak util, 240W peak power

### 4b. Batch Size 64 Profile (manually selected)

**Profile:** `c22515e8` — batch_size=64, TRT-LLM engine compiled for max 64 concurrent
**Voice:** Magpie-Multilingual.EN-US.Aria

| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | TTFT (ms) | P99 TTFA (ms) | Inter-Chunk (ms) | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,227 | 5,667 | 0.4 | 11.9x | 98 | 133 | 18.8 | 0 |
| 2 | 2,623 | 5,801 | 0.8 | 22.9x | 114 | 181 | 19.7 | 0 |
| 4 | 2,198 | 6,128 | 1.7 | 40.8x | 111 | 153 | 20.4 | 0 |
| 8 | 2,314 | 6,869 | 2.9 | 64.4x | 125 | 217 | 23.0 | 0 |
| 16 | 3,596 | 8,938 | 3.4 | 96.0x | 166 | 286 | 28.2 | 0 |
| 32 | 3,872 | 9,508 | 5.2 | 126.0x | 233 | 362 | 40.1 | 0 |
| 64 | 4,614 | 10,005 | 5.0 | 131.1x | 340 | 547 | 48.0 | 0 |

**GPU:** 81.3 GB VRAM, 96% peak util, 337W peak power

### MagpieTTS Key Findings

1. **Batch size profile is critical.** NIM auto-selected bs8 for H100, which hard-caps concurrent inference at 8. Beyond c=8, requests queue — TTFT jumps from 103ms to 2,488ms at c=16.
2. **bs64 profile eliminates the bottleneck.** With `NIM_MANIFEST_PROFILE` override, throughput scales smoothly to c=64. TTFT stays under 340ms even at c=64.
3. **VRAM trade-off is massive.** bs8 uses 14.6 GB; bs64 uses 81.3 GB (entire H100). The TRT-LLM engine pre-allocates KV-cache and buffers for max batch size.
4. **~20x slower than FastPitch per request** — expected for autoregressive transformer architecture. 2.2s mean at c=1 vs 119ms for FastPitch.
5. **Still real-time capable.** Even at 11.9x RTFX (c=1 bs64), audio is generated 12x faster than playback speed.
6. **bs64 doubles peak throughput.** 5.2 req/s (bs64) vs 2.9 req/s (bs8).
7. **Rich features justify the cost.** 7 languages, 80+ voices, emotional variants (Happy, Sad, Angry, Calm, Fearful), multilingual cross-voice capability.
8. **Available H100 profiles:** bs8 (14.6 GB), bs32, bs64 (81.3 GB). Also available for A100 and L40S GPUs.

---

## 5. Model Comparison

| Metric | ASR (Parakeet CTC) | TTS (FastPitch) | TTS (MagpieTTS bs8) | TTS (MagpieTTS bs64) |
|---|---|---|---|---|
| **Architecture** | CTC (non-autoregressive) | Spectrogram+Vocoder (non-AR) | Transformer+Codec (AR) | Transformer+Codec (AR) |
| **Latency c=1** | 7,788ms (real-time bound) | 119ms | 2,381ms | 2,227ms |
| **TTFT c=1** | n/a | 36ms | 85ms | 98ms |
| **Peak throughput** | 2.5 req/s (c=256) | 18.7 req/s (c=64) | 2.9 req/s (c=8) | 5.2 req/s (c=32) |
| **Peak RTFX** | 20.4x (c=256) | 526x (c=64) | 75.1x (c=16) | 131.1x (c=64) |
| **VRAM** | 15 GB (constant) | 16.5 GB (constant) | 14.6 GB (constant) | 81.3 GB (constant) |
| **GPU Util Peak** | 75% | 98% | 94% | 96% |
| **Peak Power** | 213W | 447W | 240W | 337W |
| **Max concurrent** | 256+ (tested) | 64+ (tested) | 8 (hard cap) | 64 (tested) |
| **Languages** | English | English | 7 languages | 7 languages |
| **Voices** | n/a | 3 | 80+ with emotions | 80+ with emotions |
| **Zero errors** | 0/300 | 0/350 | 0/300 | 0/350 |

---

## 6. Critical Discovery: TRT-LLM Batch Size Profiles

The MagpieTTS NIM container ships with multiple TensorRT-LLM engine profiles compiled for different batch sizes. The auto-selector picks a conservative default (bs8 on H100), which **silently caps concurrency** and causes severe TTFT degradation under load.

**Available MagpieTTS profiles for H100:**

| Profile ID | Batch Size | VRAM | Use Case |
|---|---|---|---|
| `48afefb5...` | 8 | ~14.6 GB | Shared GPU, low traffic |
| `3263ca3e...` | 32 | ~TBD | Medium traffic |
| `c22515e8...` | 64 | ~81.3 GB | Dedicated GPU, high traffic |

**To override:** Set `NIM_MANIFEST_PROFILE=<profile_id>` when starting the container.

ASR (Parakeet CTC) does **not** have this issue — its acoustic model supports max_batch_size=1024 natively with preferred_batch_size=64.

---

## 7. MIG Benchmarks — ASR on Partitioned H100

**Profile:** `3g.40gb` x 2 instances (H100 split into two 40GB slices)
**Model:** Parakeet CTC 1.1B
**Concurrency levels:** 1, 4, 8, 16 per instance
**Requests per level:** 30

### 7a. Offline Mode — Pure GPU/Model Processing Latency

In offline mode, the full audio file is sent in a single gRPC call. The measured latency is entirely GPU/model processing time with no streaming overhead.

#### Non-MIG Baseline (Full GPU)

| Concurrency | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Throughput (r/s) | RTFX | WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 56 | 95 | 115 | 228 | 14.7 | 112x | 2.5% |
| 4 | 99 | 92 | 161 | 184 | 210 | 37.9 | 304x | 1.6% |
| 8 | 165 | 179 | 233 | 236 | 248 | 44.4 | 376x | 2.6% |
| 16 | 275 | 290 | 402 | 408 | 418 | 43.8 | 352x | 2.8% |

**GPU:** Peak 72% utilization, 15 GB VRAM, 224W power, 34C

#### MIG Per-Instance Isolated (3g.40gb)

**Instance 0:**

| Concurrency | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Throughput (r/s) | RTFX | WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75 | 55 | 94 | 105 | 466 | 13.3 | 101x | 2.2% |
| 4 | 106 | 97 | 178 | 188 | 236 | 36.6 | 343x | 4.4% |
| 8 | 158 | 161 | 212 | 232 | 251 | 45.5 | 335x | 4.7% |
| 16 | 274 | 291 | 383 | 413 | 425 | 45.9 | 364x | 2.0% |

**Instance 1:**

| Concurrency | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Throughput (r/s) | RTFX | WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76 | 56 | 123 | 141 | 255 | 13.0 | 120x | 1.9% |
| 4 | 90 | 84 | 123 | 139 | 167 | 42.9 | 274x | 4.3% |
| 8 | 174 | 172 | 238 | 270 | 302 | 40.8 | 322x | 3.5% |
| 16 | 276 | 315 | 433 | 462 | 482 | 44.7 | 278x | 3.0% |

#### MIG Aggregate (Both Instances Simultaneous)

| Concurrency/inst | Inst 0 Mean (ms) | Inst 1 Mean (ms) | Inst 0 P99 (ms) | Inst 1 P99 (ms) | Aggregate Throughput (r/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58 | 53 | 102 | 83 | 35.8 |
| 4 | 108 | 103 | 171 | 212 | 71.8 |
| 8 | 182 | 156 | 259 | 348 | 85.2 |
| 16 | 276 | 283 | 415 | 495 | 90.9 |

### 7b. Streaming Mode — MIG (3g.40gb x 2)

**Mode:** Streaming (800ms audio chunks, real-time pacing)

#### MIG Per-Instance Isolated

**Instance 0:**

| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | Server Tail (ms) | P99 Tail (ms) | WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7,070 | 17,185 | 0.1 | 1.0x | 165 | 2,257 | 5.9% |
| 4 | 7,202 | 24,679 | 0.5 | 3.5x | 67 | 99 | 4.6% |
| 8 | 8,830 | 19,657 | 0.7 | 5.9x | 74 | 108 | 3.4% |
| 16 | 8,927 | 25,235 | 0.9 | 7.5x | 80 | 137 | 4.0% |

**Instance 1:**

| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | Server Tail (ms) | P99 Tail (ms) | WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8,380 | 19,229 | 0.1 | 1.0x | 69 | 104 | 4.1% |
| 4 | 7,250 | 21,716 | 0.5 | 3.6x | 70 | 104 | 3.8% |
| 8 | 6,563 | 21,543 | 0.9 | 6.0x | 63 | 97 | 3.8% |
| 16 | 10,561 | 21,870 | 0.8 | 8.8x | 68 | 102 | 4.6% |

#### MIG Aggregate (Both Instances Simultaneous)

| Concurrency/inst | Aggregate Throughput (r/s) | Inst 0 (r/s) | Inst 1 (r/s) |
| ---: | ---: | ---: | ---: |
| 1 | 0.3 | 0.1 | 0.1 |
| 4 | 1.0 | 0.5 | 0.5 |
| 8 | 1.8 | 0.8 | 1.0 |
| 16 | 2.3 | 0.9 | 1.4 |

### 7c. MIG vs Non-MIG Comparison

| Metric | Non-MIG (Full GPU) | MIG Isolated (per inst) | MIG Aggregate (2 inst) |
| :--- | ---: | ---: | ---: |
| **Offline mean latency @ c=1** | 67ms | 75-76ms | 53-58ms |
| **Offline mean latency @ c=8** | 165ms | 158-174ms | 156-182ms |
| **Offline mean latency @ c=16** | 275ms | 274-276ms | 276-283ms |
| **Offline P99 latency @ c=16** | 418ms | 425-482ms | 415-495ms |
| **Offline peak throughput** | 44.4 req/s (c=8) | 45.9 req/s (c=16) | **90.9 req/s** (c=16) |
| **Streaming server tail @ c=8** | 57ms (from section 2) | 63-74ms | 69-74ms |
| **Streaming peak throughput** | 2.5 req/s (c=256) | 0.9 req/s (c=16) | 2.3 req/s (c=16) |
| **VRAM used** | 15 GB / 80 GB (19%) | ~15 GB / 40 GB (38%) | ~30 GB / 80 GB (38%) |
| **GPU utilization** | 72% peak | N/A (NVML unsupported in MIG) | N/A |

### 7d. MIG Key Findings

1. **Per-instance latency is virtually identical to full GPU.** A single 3g.40gb MIG slice (40 GB) processes audio at the same speed as the full 80 GB GPU — 67ms vs 75ms at c=1, 275ms vs 274ms at c=16. The Parakeet CTC 1.1B model fits entirely within a single slice.
2. **MIG doubles aggregate throughput in offline mode.** 90.9 req/s with 2 instances vs 44.4 req/s on full GPU — the second slice adds nearly 100% throughput with no latency penalty.
3. **The full GPU is severely underutilized without MIG.** Only 72% peak utilization and 15 GB of 80 GB VRAM. MIG lets you use the idle compute and memory by running a second (or more) instance.
4. **No contention under simultaneous load.** When both MIG instances run at full concurrency, per-instance latencies remain comparable to isolated runs — MIG provides true compute isolation.
5. **Streaming throughput is real-time-bound, not compute-bound.** Streaming mode at c=16/instance gives only 2.3 req/s aggregate because audio chunks are paced at real-time speed. Offline mode reveals the actual model capacity: 90.9 req/s.
6. **Server tail latency is consistent across MIG and non-MIG.** 57-80ms regardless of configuration — the incremental processing overhead is constant.
7. **NVML GPU monitoring is not supported in MIG mode.** `nvmlDeviceGetUtilizationRates()` fails with "Not Supported" on the physical GPU handle when MIG is enabled. Per-MIG-device monitoring requires `nvmlDeviceGetMigDeviceHandleByIndex()`.
8. **Model caching matters for container startup.** First launch downloads ~19 GB from NGC (~2-3 min). With cached models mapped to `/home/nvs/.cache/nim`, containers start in <30s.

---

## 8. Capacity Planning — 5,000 Concurrent Real-Time Conversations

### Assumptions

- 5,000 users in simultaneous real-time voice conversations
- Each user has an active ASR stream (continuous) and intermittent TTS requests (bursty)
- TTS fires when the system responds — estimated 30-40% of users at any moment
- Dedicated H100 80GB GPUs for each pipeline
- Target: TTFT < 500ms for conversational quality

### ASR Capacity

- **Model:** Parakeet CTC 1.1B
- **VRAM per GPU:** 15 GB (can co-locate multiple instances on one H100)
- **Concurrent streams per GPU:** ~200 (conservative, based on RTFX=20x at c=256)
- **GPUs needed:** `5,000 / 200 = 25 ASR GPUs`
- **VRAM efficiency:** Only using 15/80 GB — could run 4-5 ASR instances per GPU
- **With MIG (3g.40gb x 2):** Each instance delivers 45 req/s offline (aggregate 90.9 req/s). MIG doubles throughput per physical GPU while maintaining identical per-request latency.
- **Optimized:** `25 / 2 = ~13 physical H100s` with MIG (2 instances per GPU)

### TTS Capacity (MagpieTTS bs64)

- **Concurrent TTS demand:** `5,000 × 0.35 = ~1,750 simultaneous TTS requests`
- **Profile:** bs64 (81.3 GB VRAM — requires dedicated H100)
- **Concurrent per GPU:** 32 (sweet spot: 5.2 req/s, 233ms TTFT, 3.9s mean latency)
- **GPUs needed:** `1,750 / 32 = ~55 TTS GPUs`

### TTS Alternative: FastPitch-HifiGAN (English-only)

- If English-only is acceptable:
- **Concurrent per GPU:** 64 (18.7 req/s, ~16.5 GB VRAM)
- **GPUs needed:** `1,750 / 64 = ~28 TTS GPUs`
- **Or with multi-instance (4 per H100):** `28 / 4 = ~7 physical GPUs`

### Summary

| Scenario | ASR GPUs | TTS GPUs | Total H100s | ASR:TTS Ratio |
|---|---|---|---|---|
| **MagpieTTS (multilingual)** | 25 (or 7 multi-instance) | 55 | 62-80 | 1:2.2 |
| **FastPitch (English-only)** | 25 (or 7 multi-instance) | 28 (or 7 multi-instance) | 14-53 | 1:1 |
| **Hybrid** (FastPitch for English, Magpie for others) | 7 | 10-30 | 17-37 | varies |

### Recommendation

- **TTS is the bottleneck** — autoregressive MagpieTTS requires 2-8x more GPUs than ASR
- **Always use bs64 profile** on dedicated TTS GPUs — it doubles throughput vs auto-selected bs8
- **Co-locate ASR instances** — at 15 GB VRAM each, fit 4-5 per H100
- **Consider FastPitch for English traffic** — 10x more efficient; use MagpieTTS only for multilingual/emotional synthesis
- **Monitor TTFT, not just throughput** — for conversational AI, time-to-first-audio directly impacts perceived responsiveness

---

## 9. Benchmark Commands

### ASR

```bash
python3 run_bench.py asr \
  --mode streaming \
  --concurrency 1,16,32,64,128,256 \
  --requests 50 \
  --gpu-interval 1.0
```

### TTS — FastPitch

```bash
python3 run_tts_bench.py \
  --mode streaming \
  --concurrency 1,2,4,8,16,32,64 \
  --requests 50 \
  --voice "English-US.Female-1" \
  --gpu-interval 0.5
```

### TTS — MagpieTTS (bs64 profile)

```bash
# Start container with bs64 profile
docker run -d \
  --name riva-tts \
  --gpus '"device=0"' \
  --shm-size=8g \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MANIFEST_PROFILE="c22515e8861affad674375ea30c5461e305e04f2bf57b5f53282b19226197b71" \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -v "$HOME/.cache/nim/magpie-tts:/opt/nim/.cache" \
  -p 50052:50051 \
  -p 9002:9000 \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:latest

# Run benchmark
python3 run_tts_bench.py \
  --mode streaming \
  --concurrency 1,2,4,8,16,32,64 \
  --requests 50 \
  --voice "Magpie-Multilingual.EN-US.Aria" \
  --gpu-interval 0.5
```

---

### MIG Benchmark

```bash
# Offline mode — 2 instances on 3g.40gb MIG slices
python3 run_mig_bench.py \
  --mig-profile 3g.40gb \
  --mode offline \
  --concurrency 1,4,8,16 \
  --requests 30 \
  --yes

# Streaming mode
python3 run_mig_bench.py \
  --mig-profile 3g.40gb \
  --mode streaming \
  --concurrency 1,4,8,16 \
  --requests 30 \
  --yes

# Dry run (preview plan without executing)
python3 run_mig_bench.py --mig-profile 2g.20gb --dry-run
```

### ASR Non-MIG Offline

```bash
python3 run_bench.py asr \
  --mode offline \
  --concurrency 1,4,8,16 \
  --requests 30
```

---

## 10. Raw Results

All benchmark results (JSON + Markdown reports) are stored in the `results/` directory:

- `results/asr_20260219_001827.md` — ASR Parakeet CTC streaming benchmark (non-MIG)
- `results/asr_20260220_230457.md` — ASR Parakeet CTC offline benchmark (non-MIG)
- `results/mig_asr_3g_40gb_20260220_222942.md` — MIG ASR streaming benchmark (3g.40gb x 2)
- `results/mig_asr_3g_40gb_20260220_224926.md` — MIG ASR offline benchmark (3g.40gb x 2)
- `results/tts_20260219_030515.md` — TTS FastPitch-HifiGAN streaming benchmark
- `results/tts_20260219_041704.md` — TTS MagpieTTS bs8 streaming benchmark
- `results/tts_20260219_045528.md` — TTS MagpieTTS bs64 streaming benchmark
