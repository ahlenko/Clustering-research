"""
benchmark_runner.py – клас для багаторазових вимірювань.
(Фактично використовується через core/benchmark.py)
"""
import numpy as np
from .profiler import profile

class BenchmarkRunner:
    def __init__(self, runs=3):
        self.runs = runs

    def run(self, func, *args, **kwargs):
        times, memories, cpus, per_core_runs = [], [], [], []
        last_result = None
        for _ in range(self.runs):
            res, stats = profile(func, *args, **kwargs)
            last_result = res
            times.append(stats['time_sec'])
            memories.append(stats['memory_peak_kb'])
            cpus.append(stats['cpu_percent'])
            per_core_runs.append(stats['cpu_per_core_percent'])
        agg = {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'memory_peak_mean': np.mean(memories),
            'cpu_mean': np.mean(cpus),
            'cpu_per_core_mean': np.mean(np.asarray(per_core_runs), axis=0).tolist(),
        }
        return last_result, agg
