"""
lemmatizer.py – лематизація з використанням pymorphy3.
"""
import pymorphy3

class Lemmatizer:
    def __init__(self, lang: str = "ukrainian"):
        """
        Ініціалізує лематизатор для заданої мови.
        Наразі підтримується тільки українська (lang='ukrainian' або 'uk').
        """
        self.lang = lang
        if lang in ("ukrainian", "uk"):
            morph_lang = "uk"
        else:
            morph_lang = "uk"
        self.morph = pymorphy3.MorphAnalyzer(lang=morph_lang)

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Приймає список токенів (слів) і повертає список їхніх лем.
        Якщо слово не вдається розпізнати, воно залишається без змін.
        """
        lemmas = []
        for token in tokens:
            parsed = self.morph.parse(token)
            if parsed:
                lemmas.append(parsed[0].normal_form)
            else:
                lemmas.append(token)
        return lemmas