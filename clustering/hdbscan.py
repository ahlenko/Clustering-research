import hdbscan
from .base import BaseClusterer
import numpy as np

class HDBSCANClusterer(BaseClusterer):
    def __init__(self, min_cluster_size: int = 5):
        self.model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)