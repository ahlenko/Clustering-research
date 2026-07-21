from sklearn.cluster import MiniBatchKMeans
from .base import BaseClusterer
import numpy as np

class MiniBatchKMeansClusterer(BaseClusterer):
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)