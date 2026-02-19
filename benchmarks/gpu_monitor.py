"""GPU monitoring via pynvml for background metrics collection during benchmarks."""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
    logger.warning("pynvml not installed; GPU monitoring disabled.")


@dataclass
class GpuSample:
    timestamp: float
    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    utilization_gpu_pct: float
    utilization_memory_pct: float
    memory_used_mb: float
    memory_free_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    sm_clock_mhz: int
    mem_clock_mhz: int


@dataclass
class GpuSummary:
    gpu_name: str
    gpu_uuid: str
    memory_total_mb: float
    sample_count: int
    duration_sec: float
    peak_utilization_gpu_pct: float
    mean_utilization_gpu_pct: float
    peak_memory_used_mb: float
    mean_memory_used_mb: float
    peak_temperature_c: float
    mean_temperature_c: float
    peak_power_draw_w: float
    mean_power_draw_w: float
    power_limit_w: float
    samples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class GpuMonitor:
    """Background GPU metrics sampler using pynvml.

    Usage:
        monitor = GpuMonitor(gpu_index=0, interval=1.0)
        monitor.start()
        # ... run benchmark ...
        summary = monitor.stop()
    """

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self._samples: list[GpuSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._available = False
        self._start_time: float = 0.0

    def _init_nvml(self) -> bool:
        if not _NVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_index >= device_count:
                logger.warning(f"GPU index {self.gpu_index} not available (count={device_count})")
                return False
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            return True
        except pynvml.NVMLError as e:
            logger.warning(f"NVML init failed: {e}")
            return False

    def _sample(self) -> Optional[GpuSample]:
        try:
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except pynvml.NVMLError:
                power_draw = 0.0

            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
            except pynvml.NVMLError:
                power_limit = 0.0

            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            except pynvml.NVMLError:
                sm_clock = 0

            try:
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                mem_clock = 0

            return GpuSample(
                timestamp=time.time(),
                gpu_index=self.gpu_index,
                gpu_name=name,
                gpu_uuid=uuid,
                utilization_gpu_pct=float(util.gpu),
                utilization_memory_pct=float(util.memory),
                memory_used_mb=mem.used / 1024**2,
                memory_free_mb=mem.free / 1024**2,
                memory_total_mb=mem.total / 1024**2,
                temperature_c=float(temp),
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                sm_clock_mhz=sm_clock,
                mem_clock_mhz=mem_clock,
            )
        except pynvml.NVMLError as e:
            logger.warning(f"NVML sampling error: {e}")
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample()
            if sample:
                self._samples.append(sample)
            self._stop_event.wait(timeout=self.interval)

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._available = self._init_nvml()
        self._start_time = time.time()
        if self._available:
            self._thread = threading.Thread(target=self._run, daemon=True, name="GpuMonitor")
            self._thread.start()
            logger.info(f"GPU monitor started (index={self.gpu_index}, interval={self.interval}s)")
        else:
            logger.info("GPU monitor running in no-op mode (no GPU or pynvml unavailable)")

    def snapshot(self, since: float) -> dict:
        """Return GPU stats for samples collected since the given timestamp.

        Returns a dict with peak/mean utilization, memory, temperature, power
        for the time window, suitable for embedding in per-concurrency results.
        """
        window = [s for s in self._samples if s.timestamp >= since]
        if not window:
            return {}

        def _peak(attr):
            return max(getattr(s, attr) for s in window)

        def _mean(attr):
            vals = [getattr(s, attr) for s in window]
            return sum(vals) / len(vals)

        return {
            "gpu_samples": len(window),
            "gpu_util_peak_pct": round(_peak("utilization_gpu_pct"), 1),
            "gpu_util_mean_pct": round(_mean("utilization_gpu_pct"), 1),
            "gpu_mem_peak_mb": round(_peak("memory_used_mb"), 0),
            "gpu_mem_mean_mb": round(_mean("memory_used_mb"), 0),
            "gpu_power_peak_w": round(_peak("power_draw_w"), 0),
            "gpu_power_mean_w": round(_mean("power_draw_w"), 0),
            "gpu_temp_peak_c": round(_peak("temperature_c"), 0),
        }

    def stop(self) -> Optional[GpuSummary]:
        duration = time.time() - self._start_time
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
            self._thread = None

        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        if not self._samples:
            logger.info("No GPU samples collected.")
            return None

        samples = self._samples

        def _peak(attr):
            return max(getattr(s, attr) for s in samples)

        def _mean(attr):
            vals = [getattr(s, attr) for s in samples]
            return sum(vals) / len(vals)

        summary = GpuSummary(
            gpu_name=samples[0].gpu_name,
            gpu_uuid=samples[0].gpu_uuid,
            memory_total_mb=samples[0].memory_total_mb,
            sample_count=len(samples),
            duration_sec=round(duration, 2),
            peak_utilization_gpu_pct=_peak("utilization_gpu_pct"),
            mean_utilization_gpu_pct=round(_mean("utilization_gpu_pct"), 2),
            peak_memory_used_mb=round(_peak("memory_used_mb"), 1),
            mean_memory_used_mb=round(_mean("memory_used_mb"), 1),
            peak_temperature_c=_peak("temperature_c"),
            mean_temperature_c=round(_mean("temperature_c"), 1),
            peak_power_draw_w=round(_peak("power_draw_w"), 1),
            mean_power_draw_w=round(_mean("power_draw_w"), 1),
            power_limit_w=samples[0].power_limit_w,
            samples=[asdict(s) for s in samples],
        )
        return summary
