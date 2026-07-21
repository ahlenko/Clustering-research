"""
cleaner.py – очищення тексту від URL, спецсимволів тощо.
"""
import re

class TextCleaner:
    def clean(self, text: str) -> str:
        # До нижнього регістру
        text = text.lower()
        # Видалення URL
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Видалення всього, крім літер української/англійської та пробілів
        text = re.sub(r'[^a-zа-яіїєґ\s]', '', text)
        # Заміна множинних пробілів на один
        text = re.sub(r'\s+', ' ', text).strip()
        return text