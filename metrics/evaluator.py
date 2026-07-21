"""
evaluator.py – (опціонально) клас, що об'єднує внутрішні та зовнішні оцінки.
"""
from .internal import evaluate_internal
from .external import evaluate_external

class Evaluator:
    @staticmethod
    def full_evaluation(X, labels, true_labels=None):
        res = {}
        res.update(evaluate_internal(X, labels))
        if true_labels is not None:
            res.update(evaluate_external(true_labels, labels))
        return res