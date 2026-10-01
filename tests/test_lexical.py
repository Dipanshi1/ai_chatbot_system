import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml import RAW_DATA_PATH, VECTORIZER_PATH, analyze_lexical, clean_text, content_tokens, tokenize


class TestTokenization(unittest.TestCase):
    def test_matches_saved_vectorizer_analyzer_on_knowledge_base(self):
        """Lexical layer must see exactly the tokens the classifier sees."""
        analyzer = joblib.load(VECTORIZER_PATH).build_analyzer()
        questions = pd.read_excel(RAW_DATA_PATH)["User Message"].map(clean_text)
        for q in questions:
            self.assertEqual(list(tokenize(q)), analyzer(q), q)

    def test_single_character_tokens_dropped(self):
        self.assertEqual(tokenize("how do i connect"), ("how", "do", "connect"))

    def test_content_tokens_drop_stopwords(self):
        self.assertEqual(content_tokens(("how", "do", "connect", "to", "college")), ("connect", "college"))


class TestLexicalEvidence(unittest.TestCase):
    VOCAB = {"connect": 0, "financial": 1, "aid": 2, "how": 3}

    def test_low_coverage(self):
        ev = analyze_lexical("how do i connect to college faculty?", self.VOCAB)
        self.assertEqual(ev.content_tokens, ("connect", "college", "faculty"))
        self.assertEqual(ev.in_vocab_content_tokens, ("connect",))
        self.assertEqual(ev.oov_content_tokens, ("college", "faculty"))
        self.assertEqual(ev.in_vocab_content_count, 1)
        self.assertAlmostEqual(ev.content_oov_ratio, 2 / 3)

    def test_stopword_only_in_vocab_tokens_are_not_content(self):
        ev = analyze_lexical("how", self.VOCAB)
        self.assertEqual(ev.content_tokens, ())
        self.assertEqual(ev.in_vocab_content_count, 0)
        self.assertEqual(ev.content_oov_ratio, 1.0)

    def test_all_oov(self):
        ev = analyze_lexical("zzz qqq", self.VOCAB)
        self.assertEqual(ev.in_vocab_content_count, 0)
        self.assertEqual(ev.content_oov_ratio, 1.0)

    def test_full_coverage(self):
        ev = analyze_lexical("financial aid", self.VOCAB)
        self.assertEqual(ev.in_vocab_content_count, 2)
        self.assertEqual(ev.content_oov_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
