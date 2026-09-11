"""
profiler.py – агрегація всіх вимірювань (час, пам'ять, CPU).
"""
from .timer import Timer
from .memory import MemoryTracker
from .cpu import CPUMonitor
import time

def profile(func, *args, **kwargs):
    """Запускає func і повертає (результат, словник статистики)."""
    stats = {}
    cpu_mon = CPUMonitor(interval=0.1)

    # CPU
    cpu_mon.start()
    # Пам'ять
    mem = MemoryTracker()
    mem.__enter__()
    # Час
    t = Timer()
    t.__enter__()

    result = func(*args, **kwargs)

    t.__exit__()
    mem.__exit__()
    cpu_stats = cpu_mon.stop()

    stats['time_sec'] = t.elapsed
    stats['memory_peak_kb'] = mem.peak_kb
    stats['cpu_percent'] = cpu_stats['process_cpu_percent']
    stats['cpu_per_core_percent'] = cpu_stats['per_core_percent']
    stats['cpu_samples'] = cpu_stats['samples']
    return result, stats
