"""
external.py – зовнішні метрики (якщо є справжні мітки).
"""
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, f1_score, v_measure_score,
)
import numpy as np

def evaluate_external(true_labels: list, pred_labels: np.ndarray) -> dict:
    """F1 (macro/micro), V-measure, ARI and NMI for labelled corpora."""
    # Фільтруємо шумові точки (label = -1), якщо є
    mask = pred_labels != -1
    if mask.sum() == 0:
        return {"f1_macro": 0.0, "f1_micro": 0.0, "v_measure": 0.0,
                "ari": 0.0, "nmi": 0.0}
    true_filtered = [true_labels[i] for i, m in enumerate(mask) if m]
    pred_filtered = pred_labels[mask]
    # Cluster IDs have no intrinsic meaning. Map them to classes optimally
    # before calculating F1, while invariant metrics use raw cluster IDs.
    from scipy.optimize import linear_sum_assignment
    classes, true_idx = np.unique(true_filtered, return_inverse=True)
    clusters, pred_idx = np.unique(pred_filtered, return_inverse=True)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    np.add.at(contingency, (true_idx, pred_idx), 1)
    rows, cols = linear_sum_assignment(contingency, maximize=True)
    mapping = {clusters[col]: classes[row] for row, col in zip(rows, cols)}
    mapped = [mapping.get(cluster, "__unmapped_cluster__") for cluster in pred_filtered]
    return {
        "f1_macro": f1_score(true_filtered, mapped, average="macro", zero_division=0),
        "f1_micro": f1_score(true_filtered, mapped, average="micro", zero_division=0),
        "v_measure": v_measure_score(true_filtered, pred_filtered),
        "ari": adjusted_rand_score(true_filtered, pred_filtered),
        "nmi": normalized_mutual_info_score(true_filtered, pred_filtered),
    }
