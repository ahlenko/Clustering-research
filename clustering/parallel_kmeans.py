"""Custom parallel multi-start K-Means clustering.

Several independently seeded K-Means candidates are fitted concurrently and
the solution with the lowest within-cluster inertia is retained.  Unlike a
single initialization, this reduces sensitivity to an unlucky starting point;
unlike sequential restarts, candidate fits can use several CPU cores.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ParallelKMeansClusterer:
    n_clusters: int = 5
    restarts: int = 8
    workers: int = 4
    init: str = "k-means++"
    random_state: int = 42

    def _fit_candidate(self, X: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
        model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=1,
            random_state=seed,
        )
        labels = model.fit_predict(X)
        return float(model.inertia_), labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if self.n_clusters < 2:
            raise ValueError("Кількість кластерів має бути щонайменше 2")
        if self.n_clusters > len(X):
            raise ValueError("Кількість кластерів не може перевищувати кількість документів")
        if self.restarts < 1 or self.workers < 1:
            raise ValueError("Кількість перезапусків і потоків має бути більшою за нуль")

        seeds: Sequence[int] = [self.random_state + index for index in range(self.restarts)]
        max_workers = min(self.workers, self.restarts)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cluster") as executor:
            candidates = list(executor.map(lambda seed: self._fit_candidate(X, seed), seeds))
        _inertia, labels = min(candidates, key=lambda candidate: candidate[0])
        return labels
