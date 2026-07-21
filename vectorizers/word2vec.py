"""
word2vec.py – усереднений Word2Vec через gensim.
"""
from gensim.models import Word2Vec
from .base import BaseVectorizer
import numpy as np

class Word2VecVectorizer(BaseVectorizer):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 workers: int = 4, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        # Навчання моделі
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        # Отримання векторів документів як середнє векторів слів
        doc_vectors = []
        for tokens in tokenized_texts:
            vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if vecs:
                doc_vectors.append(np.mean(vecs, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))
        return np.array(doc_vectors)