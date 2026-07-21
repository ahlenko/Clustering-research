from sklearn.cluster import SpectralClustering
from .base import BaseClusterer
import numpy as np

class SpectralClusterer(BaseClusterer):
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='nearest_neighbors')

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)