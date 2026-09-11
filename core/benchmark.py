"""
benchmark.py – основний клас для вимірювання продуктивності.
Використовує модулі з benchmark/ для таймерів, пам'яті та CPU.
"""
import logging
import time
import tracemalloc
from typing import Tuple, Any
import numpy as np
from benchmark.cpu import CPUMonitor

logger = logging.getLogger(__name__)

class Benchmark:
    """Обгортка для повторюваних вимірювань часу, пам'яті, CPU."""
    def __init__(self, runs: int = 3, profile_memory: bool = True, profile_cpu: bool = True):
        self.runs = runs
        self.profile_memory = profile_memory
        self.profile_cpu = profile_cpu

    def measure(self, func, *args, name="function", **kwargs) -> Tuple[Any, dict]:
        """
        Виконує func(*args, **kwargs) runs разів і повертає (результат_останнього_виклику, статистика).
        """
        times = []
        memories_peak = []
        cpu_usages = []
        result = None

        for i in range(self.runs):
            # Час
            start = time.perf_counter()
            # Пам'ять
            if self.profile_memory:
                tracemalloc.start()
            # CPU
            cpu_mon = None
            if self.profile_cpu:
                cpu_mon = CPUMonitor(interval=0.05)
                cpu_mon.start()

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if self.profile_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memories_peak.append(peak / 1024)  # KB

            if self.profile_cpu and cpu_mon:
                cpu_stats = cpu_mon.stop()
                avg_cpu = cpu_stats["process_cpu_percent"]
                cpu_usages.append(avg_cpu)

            logger.info(f"{name} запуск {i+1}: час={elapsed:.4f}с, "
                        f"пікова пам'ять={memories_peak[-1] if memories_peak else 'N/A'} KB"
                        + (f", CPU={avg_cpu:.1f}%" if cpu_usages else ""))

        stats = {
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "memory_peak_mean": np.mean(memories_peak) if memories_peak else None,
            "cpu_mean": np.mean(cpu_usages) if cpu_usages else None
        }
        return result, stats
