"""
test_preprocessing.py – простий тест пайплайну.
"""
import unittest
from preprocessing.pipeline import PreprocessingPipeline

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessingPipeline(remove_stopwords=False, lemmatize=False)

    def test_clean_url(self):
        text = "Відвідайте http://example.com та www.site.org сьогодні!"
        tokens = self.pipeline.process(text)
        self.assertNotIn("http", ' '.join(tokens))
        self.assertIn("сьогодні", tokens)

    def test_lowercase(self):
        text = "ВЕЛИКІ літери"
        tokens = self.pipeline.process(text)
        self.assertEqual(tokens, ["великі", "літери"])

if __name__ == "__main__":
    unittest.main()