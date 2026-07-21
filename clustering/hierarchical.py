from sklearn.cluster import AgglomerativeClustering
from .base import BaseClusterer
import numpy as np

class HierarchicalClusterer(BaseClusterer):
    def __init__(self, n_clusters: int = 5):
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)