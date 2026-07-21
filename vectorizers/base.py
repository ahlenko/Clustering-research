from abc import ABC, abstractmethod
import numpy as np

class BaseVectorizer(ABC):
    @abstractmethod
    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        """Перетворює список токенізованих текстів у матрицю ознак."""
        pass