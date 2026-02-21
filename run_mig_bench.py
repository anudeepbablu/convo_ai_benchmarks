#!/usr/bin/env python3
"""MIG-partitioned ASR benchmark — run multiple ASR NIM instances on GPU slices.

Partitions an H100/A100 via MIG, launches one ASR container per slice,
benchmarks each in isolation, then benchmarks all simultaneously to
measure aggregate throughput.

Usage:
    python run_mig_bench.py --mig-profile 3g.40gb              # 2 instances
    python run_mig_bench.py --mig-profile 2g.20gb              # 3 instances
    python run_mig_bench.py --mig-profile 3g.40gb --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ASR_IMAGE = "nvcr.io/nim/nvidia/parakeet-ctc-1.1b-asr:latest"
CONTAINER_PREFIX = "mig-asr"
BASE_GRPC_PORT = 50061
ASR_MIN_VRAM_GB = 15

# MIG profile -> (num_instances, profile_id for nvidia-smi)
# profile_id values from `nvidia-smi mig -lgip`
MIG_PROFILES = {
    "1g.10gb":  {"instances": 7, "vram_gb": 9.8},
    "1g.20gb":  {"instances": 4, "vram_gb": 19.6},
    "2g.20gb":  {"instances": 3, "vram_gb": 19.6},
    "3g.40gb":  {"instances": 2, "vram_gb": 39.5},
    "4g.40gb":  {"instances": 1, "vram_gb": 39.5},
    "7g.80gb":  {"instances": 1, "vram_gb": 79.0},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mig_bench")


# ── Shell helpers ────────────────────────────────────────────────────────────

def run_cmd(cmd: str, check: bool = True, sudo: bool = False,
            capture: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    if sudo:
        cmd = f"sudo {cmd}"
    logger.debug(f"$ {cmd}")
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture,
        text=True, timeout=timeout,
    )


def run_cmd_output(cmd: str, sudo: bool = False, timeout: int = 60) -> str:
    r = run_cmd(cmd, sudo=sudo, timeout=timeout)
    return r.stdout.strip()


# ── Pre-flight checks ───────────────────────────────────────────────────────

def check_gpu_mig_capable(gpu_index: int) -> str:
    """Return GPU name; abort if not MIG-capable."""
    try:
        name = run_cmd_output(
            f"nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i {gpu_index}"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("nvidia-smi not found or GPU not accessible")
        sys.exit(1)

    mig_keywords = ["H100", "H200", "A100", "A30"]
    if not any(kw in name for kw in mig_keywords):
        logger.error(f"GPU '{name}' may not support MIG. Expected one of: {mig_keywords}")
        sys.exit(1)
    logger.info(f"GPU {gpu_index}: {name}")
    return name


def check_sudo():
    """Verify sudo nvidia-smi works."""
    try:
        run_cmd("nvidia-smi -L", sudo=True)
    except subprocess.CalledProcessError:
        logger.error("sudo nvidia-smi failed — MIG commands require root privileges")
        sys.exit(1)
    logger.info("sudo nvidia-smi: OK")


def check_audio_data() -> tuple[str, dict]:
    """Return (audio_dir, transcripts) or abort."""
    sys.path.insert(0, str(ROOT))
    from data_prep.download_librispeech import load_transcripts, AUDIO_DIR

    if not AUDIO_DIR.exists() or not list(AUDIO_DIR.glob("*.wav")):
        logger.error(f"No audio files in {AUDIO_DIR}. Run: python data_prep/download_librispeech.py")
        sys.exit(1)
    transcripts = load_transcripts()
    n_wav = len(list(AUDIO_DIR.glob("*.wav")))
    logger.info(f"Audio data: {n_wav} WAV files, {len(transcripts)} transcripts")
    return str(AUDIO_DIR), transcripts


def stop_existing_containers(yes: bool = False) -> None:
    """Stop any running NIM / benchmark containers that use the GPU."""
    out = run_cmd_output("docker ps --format '{{.Names}}'", timeout=30)
    nim_prefixes = (CONTAINER_PREFIX, "riva-asr", "riva-tts", "riva-llm")
    targets = [n for n in out.splitlines() if n.startswith(nim_prefixes)]
    if not targets:
        return
    logger.warning(f"Found running GPU containers: {targets}")
    if not yes:
        resp = input(f"Stop these containers? [y/N] ").strip().lower()
        if resp != "y":
            logger.error("Cannot proceed with existing containers. Aborting.")
            sys.exit(1)
    for name in targets:
        run_cmd(f"docker stop {name}", check=False, timeout=60)
        run_cmd(f"docker rm {name}", check=False, timeout=30)
    logger.info("Existing containers stopped")
    time.sleep(3)  # give GPU a moment to release resources


def check_gpu_processes(gpu_index: int, yes: bool = False) -> None:
    """Warn if processes are using the GPU. Try to kill remaining docker containers."""
    try:
        out = run_cmd_output(
            f"nvidia-smi -i {gpu_index} --query-compute-apps=pid,name --format=csv,noheader"
        )
    except subprocess.CalledProcessError:
        return
    if not out.strip():
        return
    logger.warning(f"GPU {gpu_index} has running processes:\n{out}")
    # Try to find and stop docker containers using the GPU
    try:
        gpu_containers = run_cmd_output(
            "docker ps -q --filter 'status=running'", timeout=15
        )
        for cid in gpu_containers.splitlines():
            if cid.strip():
                run_cmd(f"docker stop {cid.strip()}", check=False, timeout=60)
                run_cmd(f"docker rm {cid.strip()}", check=False, timeout=30)
        logger.info("Stopped remaining docker containers using GPU")
        time.sleep(3)
    except Exception:
        pass
    # Re-check
    try:
        out2 = run_cmd_output(
            f"nvidia-smi -i {gpu_index} --query-compute-apps=pid,name --format=csv,noheader"
        )
    except subprocess.CalledProcessError:
        return
    if out2.strip():
        logger.warning(f"GPU still has processes after cleanup:\n{out2}")
        if not yes:
            resp = input("Continue anyway? [y/N] ").strip().lower()
            if resp != "y":
                sys.exit(1)


# ── MIG management ───────────────────────────────────────────────────────────

def get_current_mig_mode(gpu_index: int) -> bool:
    """Return True if MIG is currently enabled."""
    try:
        out = run_cmd_output(
            f"nvidia-smi -i {gpu_index} --query-gpu=mig.mode.current --format=csv,noheader"
        )
        return out.strip().lower() == "enabled"
    except subprocess.CalledProcessError:
        return False


def destroy_existing_mig(gpu_index: int) -> None:
    """Destroy any existing MIG instances."""
    run_cmd(f"nvidia-smi mig -i {gpu_index} -dci", sudo=True, check=False)
    run_cmd(f"nvidia-smi mig -i {gpu_index} -dgi", sudo=True, check=False)


def enable_mig(gpu_index: int) -> None:
    """Enable MIG mode, resetting GPU if needed."""
    if get_current_mig_mode(gpu_index):
        logger.info("MIG mode already enabled")
        destroy_existing_mig(gpu_index)
        return

    logger.info("Enabling MIG mode...")
    run_cmd(f"nvidia-smi -i {gpu_index} -mig 1", sudo=True)

    # Check if MIG activated immediately (common with nvidia-persistenced)
    if get_current_mig_mode(gpu_index):
        logger.info("MIG mode enabled (no GPU reset needed)")
        destroy_existing_mig(gpu_index)
        return

    # MIG is pending — need GPU reset to activate
    logger.info("MIG mode pending, resetting GPU...")
    for attempt in range(5):
        try:
            run_cmd(f"nvidia-smi -i {gpu_index} -r", sudo=True)
            break
        except subprocess.CalledProcessError:
            if attempt < 4:
                logger.warning(f"GPU reset failed (attempt {attempt+1}/5), retrying in 5s...")
                time.sleep(5)
            else:
                logger.error("GPU reset failed. Try: sudo systemctl stop nvidia-persistenced, "
                             "then sudo nvidia-smi -i 0 -r, then sudo systemctl start nvidia-persistenced")
                raise
    time.sleep(3)
    if not get_current_mig_mode(gpu_index):
        logger.error("MIG mode failed to activate after GPU reset")
        sys.exit(1)
    logger.info("MIG mode enabled")


def create_mig_instances(gpu_index: int, profile: str, count: int) -> list[str]:
    """Create GPU instances + compute instances. Return list of MIG UUIDs."""
    logger.info(f"Creating {count} x {profile} GPU instances...")

    # Create GPU instances and capture IDs from output
    gi_ids: list[str] = []
    for _ in range(count):
        r = run_cmd(f"nvidia-smi mig -i {gpu_index} -cgi {profile}", sudo=True)
        # Output: "Successfully created GPU instance ID  2 on GPU  0 ..."
        m = re.search(r"GPU instance ID\s+(\d+)", r.stdout, re.IGNORECASE)
        if m:
            gi_ids.append(m.group(1))

    # Fallback: parse from -lgi if stdout capture missed IDs
    if len(gi_ids) < count:
        out = run_cmd_output(f"nvidia-smi mig -i {gpu_index} -lgi", sudo=True)
        logger.debug(f"-lgi output:\n{out}")
        # Table format:  "  0  MIG 3g.40gb    9    1    0:4"
        # Instance ID is the 4th column in the data rows
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("+") or line.startswith("|") and "Name" in line:
                continue
            # Match lines like: |   0  MIG 3g.40gb    9    1    0:4  |
            m = re.search(r"MIG\s+\S+\s+(\d+)\s+(\d+)", line)
            if m:
                gi_id = m.group(2)
                if gi_id not in gi_ids:
                    gi_ids.append(gi_id)
    logger.info(f"GPU instance IDs: {gi_ids}")

    # Create compute instances for each GPU instance
    for gi_id in gi_ids:
        run_cmd(f"nvidia-smi mig -i {gpu_index} -gi {gi_id} -cci", sudo=True)

    # Parse MIG UUIDs from nvidia-smi -L
    time.sleep(2)
    uuids = parse_mig_uuids(gpu_index)
    if len(uuids) < count:
        logger.error(f"Expected {count} MIG UUIDs, got {len(uuids)}: {uuids}")
        sys.exit(1)
    uuids = uuids[:count]
    for i, uuid in enumerate(uuids):
        logger.info(f"  Slice {i}: {uuid}")

    # Regenerate CDI spec so Docker can resolve MIG device references
    logger.info("Regenerating CDI spec for MIG devices...")
    run_cmd("nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml",
            sudo=True, timeout=30)
    return uuids


def parse_mig_uuids(gpu_index: int) -> list[str]:
    """Parse MIG device UUIDs from nvidia-smi -L."""
    out = run_cmd_output("nvidia-smi -L", sudo=True)
    uuids = []
    # MIG device lines look like:
    #   MIG 3g.40gb  Device  0: (UUID: MIG-abc123...)
    for line in out.splitlines():
        m = re.search(r"(MIG-[a-f0-9-]+)", line, re.IGNORECASE)
        if m:
            uuids.append(m.group(1))
    return uuids


def disable_mig(gpu_index: int) -> None:
    """Destroy instances and disable MIG mode."""
    logger.info("Destroying MIG instances...")
    destroy_existing_mig(gpu_index)
    logger.info("Disabling MIG mode...")
    run_cmd(f"nvidia-smi -i {gpu_index} -mig 0", sudo=True, check=False)
    if get_current_mig_mode(gpu_index):
        # MIG still active — try GPU reset
        r = run_cmd(f"nvidia-smi -i {gpu_index} -r", sudo=True, check=False)
        if r.returncode != 0:
            logger.warning("GPU reset failed during MIG disable (nvidia-persistenced may be holding the device). "
                           "MIG mode may still be pending-disabled until next reboot or manual reset.")
    time.sleep(3)
    logger.info("MIG mode disabled (or pending disable)")


# ── Container management ────────────────────────────────────────────────────

def docker_login() -> None:
    ngc_key = os.environ.get("NGC_API_KEY", "")
    if not ngc_key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("NGC_API_KEY="):
                    ngc_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["NGC_API_KEY"] = ngc_key
                    break
    if not ngc_key:
        logger.error("NGC_API_KEY not set in environment or .env file")
        sys.exit(1)
    run_cmd(
        f'echo "{ngc_key}" | docker login nvcr.io -u \'$oauthtoken\' --password-stdin',
        timeout=60,
    )
    logger.info("Docker login to nvcr.io: OK")


def launch_container(instance_idx: int, mig_uuid: str, grpc_port: int) -> str:
    """Launch an ASR NIM container on a MIG slice. Return container name."""
    name = f"{CONTAINER_PREFIX}-{instance_idx}"
    ngc_key = os.environ.get("NGC_API_KEY", "")
    cache_dir = Path.home() / ".cache" / "nim" / f"asr-mig-{instance_idx}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stop if exists
    run_cmd(f"docker stop {name}", check=False, timeout=30)
    run_cmd(f"docker rm {name}", check=False, timeout=30)

    # NIM health API port — offset to avoid collision with Triton HTTP on 8000
    nim_http_port = 8100 + instance_idx

    cmd = (
        f"docker run -d --name {name} "
        f"--runtime=nvidia "
        f"-e NVIDIA_VISIBLE_DEVICES={mig_uuid} "
        f"-e NGC_API_KEY={ngc_key} "
        f"-e NIM_HTTP_API_PORT={nim_http_port} "
        f"--shm-size=8g "
        f"-v {cache_dir}:/home/nvs/.cache/nim "
        f"-p {grpc_port}:50051 "
        f"{ASR_IMAGE}"
    )
    run_cmd(cmd, timeout=120)
    logger.info(f"Launched {name} -> gRPC port {grpc_port}, health port {nim_http_port} on {mig_uuid}")
    return name


def health_check(container_name: str, timeout_sec: int = 600) -> bool:
    """Poll Triton HTTP health endpoint until ready or timeout."""
    deadline = time.time() + timeout_sec
    interval = 5
    while time.time() < deadline:
        try:
            # Use Triton's HTTP health endpoint (port 8000)
            # Note: port 8001 is gRPC (not HTTP) and fails with curl
            r = run_cmd(
                f"docker exec {container_name} curl -sf http://localhost:8000/v2/health/ready",
                check=True, timeout=15,
            )
            logger.info(f"{container_name}: READY")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Check container is still running
            try:
                state = run_cmd_output(
                    f"docker inspect {container_name} --format '{{{{.State.Running}}}}'",
                    timeout=10,
                )
                if state.strip().lower() != "true":
                    logger.error(f"{container_name}: container exited prematurely")
                    return False
            except Exception:
                pass
            elapsed = timeout_sec - (deadline - time.time())
            logger.info(f"  {container_name}: waiting... ({elapsed:.0f}s)")
            time.sleep(interval)
    logger.error(f"{container_name}: health check timed out after {timeout_sec}s")
    return False


def health_check_all(container_names: list[str], timeout_sec: int = 600) -> bool:
    """Health check all containers in parallel."""
    logger.info(f"Health-checking {len(container_names)} containers (timeout {timeout_sec}s)...")
    with ThreadPoolExecutor(max_workers=len(container_names)) as executor:
        futures = {
            executor.submit(health_check, name, timeout_sec): name
            for name in container_names
        }
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    all_ready = all(results.values())
    if not all_ready:
        failed = [n for n, ok in results.items() if not ok]
        logger.error(f"Containers not ready: {failed}")
    return all_ready


def stop_containers(container_names: list[str]) -> None:
    """Stop and remove all benchmark containers."""
    for name in container_names:
        run_cmd(f"docker stop {name}", check=False, timeout=30)
        run_cmd(f"docker rm {name}", check=False, timeout=30)
    logger.info(f"Stopped {len(container_names)} containers")


# ── Benchmark orchestration ──────────────────────────────────────────────────

def parse_concurrency(raw: Optional[str]) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [1, 4, 8, 16]


def run_single_instance_benchmark(
    instance_idx: int,
    port: int,
    audio_dir: str,
    transcripts: dict,
    concurrency_levels: list[int],
    requests_per_level: int,
    mode: str,
    chunk_duration_ms: float,
) -> dict:
    """Run benchmark on a single ASR instance."""
    from benchmarks.asr_benchmark import run_asr_benchmark

    logger.info(f"Benchmarking instance {instance_idx} (port {port})...")
    result = run_asr_benchmark(
        audio_dir=audio_dir,
        transcripts=transcripts,
        concurrency_levels=concurrency_levels,
        requests_per_level=requests_per_level,
        host="localhost",
        port=port,
        use_ssl=False,
        mode=mode,
        chunk_duration_ms=chunk_duration_ms,
        simulate_realtime=(mode == "streaming"),
        gpu_index=0,
        gpu_monitor_interval=1.0,
        progress_callback=lambda msg: logger.info(f"  [inst-{instance_idx}] {msg}"),
    )
    result["instance_idx"] = instance_idx
    result["port"] = port
    return result


def run_aggregate_benchmark(
    ports: list[int],
    audio_dir: str,
    transcripts: dict,
    concurrency_levels: list[int],
    requests_per_level: int,
    mode: str,
    chunk_duration_ms: float,
) -> dict:
    """Run benchmarks on all instances simultaneously and combine results."""
    from benchmarks.asr_benchmark import run_asr_benchmark

    n = len(ports)
    logger.info(f"Running aggregate benchmark across {n} instances simultaneously...")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {}
        for i, port in enumerate(ports):
            f = executor.submit(
                run_asr_benchmark,
                audio_dir=audio_dir,
                transcripts=transcripts,
                concurrency_levels=concurrency_levels,
                requests_per_level=requests_per_level,
                host="localhost",
                port=port,
                use_ssl=False,
                mode=mode,
                chunk_duration_ms=chunk_duration_ms,
                simulate_realtime=(mode == "streaming"),
                gpu_index=0,
                gpu_monitor_interval=1.0,
                progress_callback=lambda msg, idx=i: logger.info(f"  [agg-{idx}] {msg}"),
            )
            futures[f] = i

        instance_results = {}
        for future in as_completed(futures):
            idx = futures[future]
            instance_results[idx] = future.result()

    wall_time = time.perf_counter() - t0

    # Combine: sum throughputs across instances for each concurrency level
    combined_results = []
    first_result = instance_results[0]
    for level_idx, level_data in enumerate(first_result.get("results", [])):
        concurrency = level_data["concurrency"]
        agg = {
            "concurrency": concurrency,
            "aggregate_throughput_rps": 0.0,
            "per_instance": [],
        }
        for inst_idx in sorted(instance_results.keys()):
            inst_levels = instance_results[inst_idx].get("results", [])
            if level_idx < len(inst_levels):
                inst_level = inst_levels[level_idx]
                agg["aggregate_throughput_rps"] += inst_level.get("throughput_rps", 0)
                agg["per_instance"].append({
                    "instance": inst_idx,
                    "port": ports[inst_idx],
                    "throughput_rps": inst_level.get("throughput_rps", 0),
                    "mean_sec": inst_level.get("mean_sec", 0),
                    "p99_sec": inst_level.get("p99_sec", 0),
                    "rtfx": inst_level.get("rtfx", 0),
                    "error_count": inst_level.get("error_count", 0),
                })
        agg["aggregate_throughput_rps"] = round(agg["aggregate_throughput_rps"], 3)
        combined_results.append(agg)

    return {
        "aggregate_results": combined_results,
        "instance_results": {k: v for k, v in instance_results.items()},
        "wall_time_sec": round(wall_time, 2),
        "num_instances": n,
    }


# ── Report generation ────────────────────────────────────────────────────────

def generate_mig_report(
    profile: str,
    num_instances: int,
    gpu_name: str,
    mig_uuids: list[str],
    isolated_results: dict[int, dict],
    aggregate_result: dict,
    mode: str,
    concurrency_levels: list[int],
    requests_per_level: int,
    wall_time: float,
) -> tuple[Path, Path]:
    """Generate JSON + Markdown reports. Return (json_path, md_path)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_profile = profile.replace(".", "_")

    full_data = {
        "timestamp": datetime.now().isoformat(),
        "mig_profile": profile,
        "num_instances": num_instances,
        "gpu_name": gpu_name,
        "mig_uuids": mig_uuids,
        "mode": mode,
        "concurrency_levels": concurrency_levels,
        "requests_per_level": requests_per_level,
        "wall_time_sec": round(wall_time, 2),
        "isolated": {str(k): v for k, v in isolated_results.items()},
        "aggregate": aggregate_result,
    }

    json_path = RESULTS_DIR / f"mig_asr_{safe_profile}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=2, default=str)

    # Markdown report
    lines: list[str] = []
    w = lines.append

    w(f"# MIG Benchmark Report — ASR ({profile})")
    w("")
    w(f"**GPU:** {gpu_name}")
    w(f"**MIG profile:** {profile} x {num_instances}")
    w(f"**Mode:** {mode}")
    w(f"**Timestamp:** {datetime.now().isoformat()}")
    w(f"**Total wall time:** {wall_time:.1f}s")
    w("")

    # Phase 1: Isolated results
    w("## Phase 1: Per-Instance Isolated Performance")
    w("")
    for inst_idx in sorted(isolated_results.keys()):
        res = isolated_results[inst_idx]
        results = res.get("results", [])
        if not results:
            continue
        w(f"### Instance {inst_idx} (port {res.get('config_used', {}).get('port', '?')})")
        w("")
        if mode == "streaming":
            w("| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | Errors |")
            w("| ---: | ---: | ---: | ---: | ---: | ---: |")
        else:
            w("| Concurrency | Mean (ms) | P99 (ms) | Throughput (r/s) | RTFX | Errors |")
            w("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results:
            w(f"| {r.get('concurrency', '')} "
              f"| {r.get('mean_sec', 0) * 1000:.1f} "
              f"| {r.get('p99_sec', 0) * 1000:.1f} "
              f"| {r.get('throughput_rps', 0):.1f} "
              f"| {r.get('rtfx', 0):.1f} "
              f"| {r.get('error_count', 0)} |")
        w("")

    # Phase 2: Aggregate results
    agg_results = aggregate_result.get("aggregate_results", [])
    if agg_results:
        w("## Phase 2: Aggregate Performance (All Instances Simultaneous)")
        w("")
        w("| Concurrency/inst | Aggregate Throughput (r/s) | " +
          " | ".join(f"Inst {i} (r/s)" for i in range(num_instances)) + " |")
        w("| ---: | ---: | " + " | ".join("---:" for _ in range(num_instances)) + " |")
        for agg in agg_results:
            cells = [str(agg["concurrency"]), f"{agg['aggregate_throughput_rps']:.1f}"]
            for pi in agg.get("per_instance", []):
                cells.append(f"{pi['throughput_rps']:.1f}")
            # Pad if missing instances
            while len(cells) < 2 + num_instances:
                cells.append("—")
            w("| " + " | ".join(cells) + " |")
        w("")

    # Highlights
    w("## Highlights")
    w("")
    if agg_results:
        best_agg = max(agg_results, key=lambda r: r["aggregate_throughput_rps"])
        w(f"- **Best aggregate throughput:** {best_agg['aggregate_throughput_rps']:.1f} req/s "
          f"at concurrency {best_agg['concurrency']}/instance")

    for inst_idx in sorted(isolated_results.keys()):
        results = isolated_results[inst_idx].get("results", [])
        if results:
            best = max(results, key=lambda r: r.get("throughput_rps", 0))
            w(f"- **Instance {inst_idx} best throughput:** {best.get('throughput_rps', 0):.1f} req/s "
              f"at concurrency {best['concurrency']}")
    w("")

    md_path = RESULTS_DIR / f"mig_asr_{safe_profile}_{ts}.md"
    md_path.write_text("\n".join(lines))

    return json_path, md_path


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MIG-partitioned ASR benchmark on H100/A100 GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mig-profile", required=True,
                   help="MIG profile name (e.g. '2g.20gb', '3g.40gb')")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="Physical GPU index (default: 0)")
    p.add_argument("-c", "--concurrency", type=str, default=None,
                   help="Concurrency levels per instance (e.g. '1,4,8,16')")
    p.add_argument("-n", "--requests", type=int, default=30,
                   help="Requests per concurrency level per instance (default: 30)")
    p.add_argument("--mode", choices=["streaming", "offline"], default="streaming",
                   help="ASR mode (default: streaming)")
    p.add_argument("--chunk-duration-ms", type=float, default=800,
                   help="Audio chunk duration for streaming mode (default: 800)")
    p.add_argument("--skip-cleanup", action="store_true",
                   help="Leave MIG state after benchmarks")
    p.add_argument("--yes", action="store_true",
                   help="Skip confirmation prompts")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without executing")
    p.add_argument("--container-timeout", type=int, default=600,
                   help="Health check timeout seconds (default: 600)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile = args.mig_profile
    gpu_index = args.gpu_index
    concurrency_levels = parse_concurrency(args.concurrency)
    container_names: list[str] = []
    mig_uuids: list[str] = []

    # Resolve profile
    prof_info = MIG_PROFILES.get(profile)
    if prof_info is None:
        logger.error(f"Unknown MIG profile '{profile}'. Known: {list(MIG_PROFILES.keys())}")
        sys.exit(1)

    num_instances = prof_info["instances"]
    vram_gb = prof_info["vram_gb"]

    if vram_gb < ASR_MIN_VRAM_GB:
        logger.error(f"Profile {profile} provides {vram_gb} GB VRAM, ASR needs >= {ASR_MIN_VRAM_GB} GB")
        sys.exit(1)

    ports = [BASE_GRPC_PORT + i for i in range(num_instances)]

    # Dry-run
    if args.dry_run:
        print(f"\n=== MIG Benchmark Plan ===")
        print(f"  GPU index:     {gpu_index}")
        print(f"  MIG profile:   {profile}")
        print(f"  Instances:     {num_instances}")
        print(f"  VRAM/slice:    {vram_gb} GB")
        print(f"  gRPC ports:    {ports}")
        print(f"  Concurrency:   {concurrency_levels}")
        print(f"  Requests/lvl:  {args.requests}")
        print(f"  Mode:          {args.mode}")
        print(f"  Chunk ms:      {args.chunk_duration_ms}")
        print(f"  Container img: {ASR_IMAGE}")
        print(f"\nSteps:")
        print(f"  1. Pre-flight checks (GPU, sudo, audio data)")
        print(f"  2. Enable MIG mode on GPU {gpu_index}")
        print(f"  3. Create {num_instances} x {profile} GPU instances + compute instances")
        print(f"  4. Launch {num_instances} ASR containers")
        print(f"  5. Health check all containers")
        print(f"  6. Phase 1: Benchmark each instance in isolation")
        print(f"  7. Phase 2: Benchmark all instances simultaneously")
        print(f"  8. Cleanup: stop containers, destroy MIG instances, disable MIG")
        print()
        return

    # Confirmation
    if not args.yes:
        print(f"\nThis will partition GPU {gpu_index} into {num_instances} x {profile} MIG slices")
        print(f"and run ASR benchmarks. Existing MIG state will be destroyed.\n")
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # ── Pre-flight ───────────────────────────────────────────────────────
    gpu_name = check_gpu_mig_capable(gpu_index)
    check_sudo()
    stop_existing_containers(yes=args.yes)
    check_gpu_processes(gpu_index, yes=args.yes)
    audio_dir, transcripts = check_audio_data()

    # Set up signal handler for cleanup
    cleanup_done = False

    def cleanup(signum=None, frame=None):
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        logger.info("Cleaning up...")
        if container_names:
            stop_containers(container_names)
        if not args.skip_cleanup:
            try:
                disable_mig(gpu_index)
            except Exception as e:
                logger.warning(f"MIG cleanup error: {e}")

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    t_total_start = time.time()

    try:
        # ── MIG setup ────────────────────────────────────────────────────
        enable_mig(gpu_index)
        mig_uuids = create_mig_instances(gpu_index, profile, num_instances)

        # ── Container deployment ─────────────────────────────────────────
        docker_login()
        for i in range(num_instances):
            name = launch_container(i, mig_uuids[i], ports[i])
            container_names.append(name)

        if not health_check_all(container_names, timeout_sec=args.container_timeout):
            logger.error("Not all containers became healthy. Aborting.")
            sys.exit(1)

        # ── Phase 1: Isolated benchmarks ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("PHASE 1: Per-instance isolated benchmarks")
        logger.info("=" * 60)

        isolated_results: dict[int, dict] = {}
        for i in range(num_instances):
            result = run_single_instance_benchmark(
                instance_idx=i,
                port=ports[i],
                audio_dir=audio_dir,
                transcripts=transcripts,
                concurrency_levels=concurrency_levels,
                requests_per_level=args.requests,
                mode=args.mode,
                chunk_duration_ms=args.chunk_duration_ms,
            )
            isolated_results[i] = result
            logger.info(f"Instance {i} isolated benchmark complete")

        # ── Phase 2: Aggregate benchmark ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("PHASE 2: Aggregate benchmark (all instances simultaneous)")
        logger.info("=" * 60)

        aggregate_result = run_aggregate_benchmark(
            ports=ports,
            audio_dir=audio_dir,
            transcripts=transcripts,
            concurrency_levels=concurrency_levels,
            requests_per_level=args.requests,
            mode=args.mode,
            chunk_duration_ms=args.chunk_duration_ms,
        )
        logger.info("Aggregate benchmark complete")

        # ── Results ──────────────────────────────────────────────────────
        wall_time = time.time() - t_total_start

        json_path, md_path = generate_mig_report(
            profile=profile,
            num_instances=num_instances,
            gpu_name=gpu_name,
            mig_uuids=mig_uuids,
            isolated_results=isolated_results,
            aggregate_result=aggregate_result,
            mode=args.mode,
            concurrency_levels=concurrency_levels,
            requests_per_level=args.requests,
            wall_time=wall_time,
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"MIG BENCHMARK COMPLETE — {profile} x {num_instances}")
        print("=" * 60)

        agg_results = aggregate_result.get("aggregate_results", [])
        if agg_results:
            print(f"\nAggregate throughput (all {num_instances} instances):")
            for agg in agg_results:
                print(f"  concurrency={agg['concurrency']}/inst -> "
                      f"{agg['aggregate_throughput_rps']:.1f} req/s total")

        print(f"\nJSON  -> {json_path}")
        print(f"Report -> {md_path}")
        print(f"Wall time: {wall_time:.1f}s\n")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
