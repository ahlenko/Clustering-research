"""
internal.py – внутрішні метрики якості кластеризації.
"""
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def evaluate_internal(X: np.ndarray, labels: np.ndarray) -> dict:
    """Повертає словник із внутрішніми метриками."""
    unique_labels = set(labels)
    # Якщо всі об'єкти в одному кластері або забагато шуму (-1)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        return {"silhouette": -1.0, "davies_bouldin": -1.0, "calinski_harabasz": -1.0}
    try:
        sil = silhouette_score(X, labels)
    except:
        sil = -1.0
    try:
        db = davies_bouldin_score(X, labels)
    except:
        db = -1.0
    try:
        ch = calinski_harabasz_score(X, labels)
    except:
        ch = -1.0
    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch
    }