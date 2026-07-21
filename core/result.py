"""
result.py – структура для збереження результатів одного експерименту.
"""
class ExperimentResult:
    def __init__(self, dataset: str, vectorizer: str, algorithm: str,
                 internal_metrics: dict, external_metrics: dict):
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.algorithm = algorithm
        self.internal_metrics = internal_metrics
        self.external_metrics = external_metrics

    def to_dict(self) -> dict:
        """Перетворює результат у словник для збереження в CSV."""
        base = {
            "dataset": self.dataset,
            "vectorizer": self.vectorizer,
            "algorithm": self.algorithm
        }
        base.update(self.internal_metrics)
        base.update(self.external_metrics)
        return base