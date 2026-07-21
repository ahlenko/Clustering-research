"""
memory.py – вимірювання пікової пам'яті через tracemalloc.
"""
import tracemalloc

class MemoryTracker:
    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *args):
        self.current, self.peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_kb = self.peak / 1024