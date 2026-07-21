"""
tfidf.py – TF-IDF векторизація через sklearn.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseVectorizer
import numpy as np

class TfidfVectorizerAdapter(BaseVectorizer):
    def __init__(self, max_features: int = 5000):
        self.model = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        # Склеюємо токени назад у рядки
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        return self.model.fit_transform(texts).toarray()