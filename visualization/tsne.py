"""
tsne.py – додаткові функції для t-SNE.
"""
from sklearn.manifold import TSNE
import numpy as np

def reduce_tsne(X: np.ndarray, n_components=2, perplexity=30, random_state=42):
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(X)