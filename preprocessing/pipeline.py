"""
pipeline.py – об'єднує всі кроки препроцесингу.
"""
from .cleaner import TextCleaner
from .tokenizer import SimpleTokenizer
from .stopwords import StopwordRemover
from .lemmatizer import Lemmatizer

class PreprocessingPipeline:
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True,
                 lang: str = "ukrainian", custom_stopwords: set[str] | None = None):
        self.cleaner = TextCleaner()
        self.tokenizer = SimpleTokenizer()
        self.stop_remover = StopwordRemover(lang, custom_stopwords) if remove_stopwords else None
        self.lemmatizer = Lemmatizer(lang) if lemmatize else None

    def process(self, text: str) -> list[str]:
        """Повертає список лематизованих токенів (слів)."""
        text = self.cleaner.clean(text)
        tokens = self.tokenizer.tokenize(text)
        if self.stop_remover:
            tokens = self.stop_remover.remove(tokens)
        if self.lemmatizer:
            tokens = self.lemmatizer.lemmatize(tokens)
        return tokens
