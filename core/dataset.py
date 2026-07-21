"""
dataset.py – завантаження текстових датасетів.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class TextDataset:
    """Завантажує CSV/JSON із текстами та опціональними мітками."""
    def __init__(self, filepath: str, text_col: str = "text", label_col: Optional[str] = "label"):
        self.filepath = Path(filepath)
        self.text_col = text_col
        self.label_col = label_col
        self.data: Optional[pd.DataFrame] = None
        self.texts: List[str] = []
        self.labels: Optional[List[str]] = None

    def load(self):
        logger.info(f"Завантаження датасету: {self.filepath}")
        if self.filepath.suffix == ".csv":
            self.data = pd.read_csv(self.filepath)
        elif self.filepath.suffix == ".json":
            self.data = pd.read_json(self.filepath)
        else:
            raise ValueError("Підтримуються тільки .csv та .json")
        self.texts = self.data[self.text_col].astype(str).tolist()
        if self.label_col and self.label_col in self.data.columns:
            self.labels = self.data[self.label_col].astype(str).tolist()
        else:
            self.labels = None
        logger.info(f"Завантажено {len(self.texts)} текстів")
        return self