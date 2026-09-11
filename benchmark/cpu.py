"""CPU measurement scoped to a benchmarked task."""
from __future__ import annotations

import os
import psutil
import threading
import time


class CPUMonitor:
    """Measure this process and per-core system load while a task executes.

    Per-core task attribution is not exposed portably by macOS/Linux. The
    overall value uses this process's CPU-time delta; per-core values are
    system samples captured during the same task interval.
    """
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._running = False
        self._core_samples: list[list[float]] = []
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._start_times = None
        self._started_at = 0.0
        self._core_count = os.cpu_count() or 1

    def _monitor(self) -> None:
        while self._running:
            self._core_samples.append(psutil.cpu_percent(interval=None, percpu=True))
            time.sleep(self.interval)

    def start(self) -> None:
        self._start_times = self._process.cpu_times()
        self._started_at = time.perf_counter()
        psutil.cpu_percent(interval=None, percpu=True)  # prime system counters
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join()
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        end_times = self._process.cpu_times()
        process_seconds = ((end_times.user + end_times.system) -
                           (self._start_times.user + self._start_times.system))
        if self._core_samples:
            per_core = [sum(values) / len(values) for values in zip(*self._core_samples)]
        else:
            per_core = [0.0] * self._core_count
        return {
            "process_cpu_percent": 100.0 * process_seconds / (elapsed * self._core_count),
            "process_cpu_percent_one_core": 100.0 * process_seconds / elapsed,
            "per_core_percent": per_core,
            "samples": len(self._core_samples),
        }
