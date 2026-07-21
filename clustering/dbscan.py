from sklearn.cluster import DBSCAN
from .base import BaseClusterer
import numpy as np

class DBSCANClusterer(BaseClusterer):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)