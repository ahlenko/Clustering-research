"""
doc2vec.py – Doc2Vec (PV-DM) через gensim.
"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .base import BaseVectorizer
import numpy as np

class Doc2VecVectorizer(BaseVectorizer):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 workers: int = 4, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        # Створюємо TaggedDocument
        documents = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(tokenized_texts)]
        self.model = Doc2Vec(
            documents=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        # Отримуємо вектори документів
        doc_vectors = np.array([self.model.dv[str(i)] for i in range(len(tokenized_texts))])
        return doc_vectors