"""
Lexical evidence: tokenization, content-token filtering and OOV coverage.

Reports facts only; acceptance/abstention is decided in policy.py.
"""
import re
from typing import Iterable, Mapping, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .schemas import LexicalEvidence

# Identical to sklearn TfidfVectorizer's default token_pattern, so the lexical
# layer sees exactly the tokens the saved classifier vectorizer sees.
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(cleaned: str) -> Tuple[str, ...]:
    """Tokenize already-cleaned (lowercased) text."""
    return tuple(TOKEN_PATTERN.findall(cleaned))


def content_tokens(tokens: Iterable[str]) -> Tuple[str, ...]:
    """Drop English stopwords, preserving order."""
    return tuple(t for t in tokens if t not in ENGLISH_STOP_WORDS)


def analyze_lexical(cleaned: str, vocabulary: Mapping[str, int]) -> LexicalEvidence:
    """Measure how much of the message's content the classifier vocabulary covers."""
    tokens = tokenize(cleaned)
    content = content_tokens(tokens)
    in_vocab = tuple(t for t in content if t in vocabulary)
    oov = tuple(t for t in content if t not in vocabulary)
    oov_ratio = len(oov) / len(content) if content else 1.0

    return LexicalEvidence(
        tokens=tokens,
        content_tokens=content,
        in_vocab_content_tokens=in_vocab,
        oov_content_tokens=oov,
        in_vocab_content_count=len(in_vocab),
        content_oov_ratio=oov_ratio,
    )
