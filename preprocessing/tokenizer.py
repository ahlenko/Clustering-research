"""
tokenizer.py – простий токенізатор за пробілами.
"""
class SimpleTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()