"""
dataset.py – завантаження текстових датасетів.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class TextDataset:
    """Завантажує TXT, CSV або JSON з текстами та опціональними мітками."""
    def __init__(self, filepath: str, text_col: str = "text", label_col: Optional[str] = "label",
                 max_documents: Optional[int] = None):
        self.filepath = Path(filepath)
        self.text_col = text_col
        self.label_col = label_col
        self.max_documents = max_documents
        self.data: Optional[pd.DataFrame] = None
        self.texts: List[str] = []
        self.labels: Optional[List[str]] = None

    def load(self):
        logger.info(f"Завантаження датасету: {self.filepath}")
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не знайдено: {self.filepath}")
        suffix = self.filepath.suffix.lower()
        if suffix == ".csv":
            self.data = pd.read_csv(self.filepath)
        elif suffix == ".json":
            self.data = pd.read_json(self.filepath)
        elif suffix == ".txt":
            texts = [line.strip() for line in self.filepath.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines() if line.strip()]
            self.data = pd.DataFrame({self.text_col: texts})
        else:
            raise ValueError("Підтримуються формати TXT, CSV та JSON")
        if self.text_col not in self.data.columns:
            raise ValueError(
                f"У файлі немає колонки з текстом «{self.text_col}». "
                f"Доступні колонки: {', '.join(map(str, self.data.columns))}"
            )
        if self.max_documents is not None:
            if self.max_documents < 1:
                raise ValueError("Кількість текстів має бути більшою за нуль")
            self.data = self.data.head(self.max_documents).copy()
        self.texts = self.data[self.text_col].astype(str).tolist()
        if not self.texts:
            raise ValueError("Файл не містить жодного текстового документа")
        if self.label_col and self.label_col in self.data.columns:
            self.labels = self.data[self.label_col].astype(str).tolist()
        else:
            self.labels = None
        logger.info(f"Завантажено {len(self.texts)} текстів")
        return self
