import hashlib
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import RAW_DATA_PATH
from .lexical import content_tokens, tokenize
from .preprocessing import clean_text
from .schemas import RetrievalCandidate


def _sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


class ResponseRetriever:
    """
    Finds the knowledge-base question most similar to a message, restricted to
    the classifier's predicted intent, and reports paraphrase evidence.

    Uses ONE global stopword-free TF-IDF index over all knowledge-base questions
    (stable IDF, comparable scores across intents). Cosine similarity ranks
    candidates; content-token Dice overlap is reported for policy.decide() to
    gate on. This class makes no acceptance decisions.
    """

    def __init__(
        self,
        data_path: Optional[Union[Path, str]] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        if data_path is not None and df is not None:
            raise ValueError("Cannot specify both data_path and df")

        self.df: Optional[pd.DataFrame] = None
        self._questions: list = []
        self._responses: list = []
        self._question_content: list = []
        self._intent_rows: Dict[str, np.ndarray] = {}
        self._row_ids: np.ndarray = np.array([])
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._kb_version: str = ""

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            if df.empty:
                raise ValueError("DataFrame must be non-empty")
            required = ["User Message", "Intent", "Bot Response"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"DataFrame missing required column(s): {missing}")
            if not df.index.is_unique:
                raise ValueError("DataFrame index must be unique")
            if not pd.api.types.is_integer_dtype(df.index):
                raise ValueError("DataFrame index must have integer dtype")

            self.data_path = None
            self.df = df.sort_index().copy()
            csv_bytes = (
                self.df[["User Message", "Intent", "Bot Response"]]
                .sort_index()
                .to_csv(lineterminator="\n")
                .encode("utf-8")
            )
            self._kb_version = f"kb-df-{hashlib.sha256(csv_bytes).hexdigest()[:12]}"
            self._build_index()
        else:
            self.data_path = Path(data_path) if data_path else RAW_DATA_PATH
            self._load_data()
            self._kb_version = f"kb-{_sha256_prefix(self.data_path)}"

    @property
    def kb_version(self) -> str:
        """Version string identifying the knowledge base contents."""
        return self._kb_version

    def _load_data(self) -> None:
        """Load historical question-response data and build the retrieval index."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df = pd.read_excel(self.data_path).reset_index(drop=True)
        self._build_index()

    def _build_index(self) -> None:
        """Fit the global retrieval vectorizer once and map intents to row ids."""
        if self.df is None or self.df.empty:
            return

        self._row_ids = np.array(self.df.index)
        self._questions = [clean_text(q) for q in self.df["User Message"]]
        self._responses = [str(r) for r in self.df["Bot Response"]]
        self._question_content = [
            frozenset(content_tokens(tokenize(q))) for q in self._questions
        ]

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._questions)

        intents = self.df["Intent"].astype(str).to_numpy()
        self._intent_rows = {
            intent: np.flatnonzero(intents == intent) for intent in sorted(set(intents))
        }

    @property
    def intents(self) -> tuple:
        return tuple(self._intent_rows)

    def best_match(self, cleaned: str, intent: Optional[str]) -> Optional[RetrievalCandidate]:
        """
        Best knowledge-base match for `cleaned` within `intent`.

        Returns None if the input is empty, the intent is unknown, or no question
        in the intent shares any (non-stopword) term with the query.
        Ties resolve to the lowest row id.
        """
        cleaned = clean_text(cleaned)
        if not cleaned or intent not in self._intent_rows:
            return None

        rows = self._intent_rows[intent]
        query_vec = self._vectorizer.transform([cleaned])
        similarities = cosine_similarity(query_vec, self._matrix[rows])[0]

        best_pos = int(np.argmax(similarities))  # first max -> lowest row id
        best_cosine = float(similarities[best_pos])
        if best_cosine <= 0.0:
            return None

        pos = int(rows[best_pos])
        row_id = int(self._row_ids[pos])
        query_content = frozenset(content_tokens(tokenize(cleaned)))
        match_content = self._question_content[pos]
        shared = query_content & match_content
        denom = len(query_content) + len(match_content)
        dice = 2 * len(shared) / denom if denom else 0.0

        return RetrievalCandidate(
            row_id=row_id,
            matched_question=self._questions[pos],
            response=self._responses[pos],
            cosine=best_cosine,
            shared_content_tokens=tuple(sorted(shared)),
            dice=dice,
        )


# Module-level singleton instance for convenient reuse
_default_retriever: Optional[ResponseRetriever] = None


def get_retriever() -> ResponseRetriever:
    """Get or initialize the shared ResponseRetriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = ResponseRetriever()
    return _default_retriever
