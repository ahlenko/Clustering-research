"""
fasttext.py – FastText вектори (середнє) через gensim.
"""
from gensim.models import FastText
from .base import BaseVectorizer
import numpy as np

class FastTextVectorizer(BaseVectorizer):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 workers: int = 4, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        self.model = FastText(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        doc_vectors = []
        for tokens in tokenized_texts:
            vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if vecs:
                doc_vectors.append(np.mean(vecs, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))
        return np.array(doc_vectors)