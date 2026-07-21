from sklearn.cluster import Birch
from .base import BaseClusterer
import numpy as np

class BirchClusterer(BaseClusterer):
    def __init__(self, n_clusters: int = 5):
        self.model = Birch(n_clusters=n_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)