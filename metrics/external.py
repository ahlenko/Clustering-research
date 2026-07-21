"""
external.py – зовнішні метрики (якщо є справжні мітки).
"""
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def evaluate_external(true_labels: list, pred_labels: np.ndarray) -> dict:
    """ARI та NMI."""
    # Фільтруємо шумові точки (label = -1), якщо є
    mask = pred_labels != -1
    if mask.sum() == 0:
        return {"ari": 0.0, "nmi": 0.0}
    true_filtered = [true_labels[i] for i, m in enumerate(mask) if m]
    pred_filtered = pred_labels[mask]
    ari = adjusted_rand_score(true_filtered, pred_filtered)
    nmi = normalized_mutual_info_score(true_filtered, pred_filtered)
    return {"ari": ari, "nmi": nmi}