"""
lemmatizer.py – лематизація з використанням pymorphy3.
"""
try:
    import pymorphy3
except ImportError:
    pymorphy3 = None

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
        self.morph = pymorphy3.MorphAnalyzer(lang=morph_lang) if pymorphy3 else None

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Приймає список токенів (слів) і повертає список їхніх лем.
        Якщо слово не вдається розпізнати, воно залишається без змін.
        """
        if self.morph is None:
            return tokens
        lemmas = []
        for token in tokens:
            parsed = self.morph.parse(token)
            if parsed:
                lemmas.append(parsed[0].normal_form)
            else:
                lemmas.append(token)
        return lemmas
