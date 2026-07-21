from abc import ABC, abstractmethod
import numpy as np

class BaseClusterer(ABC):
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        pass