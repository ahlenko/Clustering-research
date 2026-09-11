"""
stopwords.py – видалення стоп-слів.
"""
class StopwordRemover:
    # Базовий набір українських стоп-слів
    UKRAINIAN_STOPWORDS = {
        "і", "та", "в", "до", "з", "не", "на", "що", "як", "а",
        "у", "це", "він", "вона", "воно", "вони", "ми", "ви",
        "ти", "я", "ж", "же", "б", "би", "чи", "але", "про",
        "за", "для", "по", "із", "під", "над", "перед", "при",
        "без", "від", "під", "через", "так", "то", "там", "тут",
        "був", "була", "було", "були", "є", "бути", "той", "цей",
        "весь", "все", "вона", "ним", "нею", "себе", "собі",
        "також", "ще", "от", "ну", "ось"
    }

    def __init__(self, lang: str = "ukrainian", custom_stopwords: set[str] | None = None):
        if lang == "ukrainian":
            self.stopwords = set(self.UKRAINIAN_STOPWORDS)
        else:
            self.stopwords = set()  # заглушка
        self.stopwords.update(word.lower().strip() for word in (custom_stopwords or set()) if word.strip())

    def remove(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self.stopwords]
