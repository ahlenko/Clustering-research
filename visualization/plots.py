"""
plots.py – візуалізація кластерів зі зниженням розмірності.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_clusters(X: np.ndarray, labels: np.ndarray, title: str = "Clusters",
                  save_path: str = None, method: str = 'tsne', perplexity: int = 30):
    """
    Знижує розмірність до 2D та малює розкид.
    """
    # Якщо вже 2D, не знижуємо
    if X.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()