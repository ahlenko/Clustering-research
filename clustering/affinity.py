from sklearn.cluster import AffinityPropagation
from .base import BaseClusterer
import numpy as np

class AffinityPropagationClusterer(BaseClusterer):
    def __init__(self, damping: float = 0.9, max_iter: int = 300):
        self.model = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=42)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)