"""
umap.py – UMAP зниження розмірності (потрібен пакет umap-learn).
"""
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def reduce_umap(X: np.ndarray, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn не встановлено. Виконайте: pip install umap-learn")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)