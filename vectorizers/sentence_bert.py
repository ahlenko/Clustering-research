"""
sentence_bert.py – Sentence-BERT векторизація.
"""
from sentence_transformers import SentenceTransformer
from .base import BaseVectorizer
import numpy as np

class SentenceBERTVectorizer(BaseVectorizer):
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        # Sentence-BERT очікує рядки, а не списки токенів
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        return self.model.encode(texts, show_progress_bar=False)