"""
cpu.py – моніторинг середнього завантаження CPU під час виконання.
"""
import psutil
import threading
import time

class CPUMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._running = False
        self._values = []
        self._thread = None

    def _monitor(self):
        while self._running:
            self._values.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        self._thread.join()
        if self._values:
            return sum(self._values) / len(self._values)
        return 0.0