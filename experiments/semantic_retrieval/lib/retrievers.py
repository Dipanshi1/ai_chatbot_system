"""
B1 retrievers: the Stage A lexical baseline (read-only adapter) and MiniLM dense retrieval.

Both expose the same interface so evaluation and latency measurement are symmetrical:
    encode(query)                      -> query representation
    search(query_repr, k, intent=None) -> list[(row_id, score)]   (rank order)
    rank(query, k, intent=None)        -> search(encode(query), ...)

torch / sentence-transformers are imported lazily inside functions so that importing
this module does not load them (resource probes take RSS0 before that import).
"""
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.preprocessing import clean_text
from src.ml.retrieval import ResponseRetriever

import config

Ranking = List[Tuple[int, float]]


# ---------------------------------------------------------------------------
# Stage A baseline (spec §4)
# ---------------------------------------------------------------------------
class StageARetrieverAdapter:
    """
    Read-only ranking adapter over the ACTUAL Stage A ResponseRetriever(df=train_df).

    Ranks with the retriever instance's own fitted TF-IDF vectorizer and matrix; it never
    refits or mutates them. Matching ResponseRetriever.best_match():
      * rows with cosine <= 0 are not retrieved (a ranking may be shorter than k, or empty);
      * ties resolve to the lowest row_id (stable descending sort over ascending row order);
      * an unknown intent or empty query retrieves nothing.
    """

    name = "stage_a_tfidf"

    def __init__(self, train_df: pd.DataFrame):
        self.retriever = ResponseRetriever(df=train_df)
        r = self.retriever
        self._row_ids = np.asarray(r._row_ids, dtype=np.int64)
        if list(self._row_ids) != sorted(self._row_ids):
            raise RuntimeError("Stage A retriever rows are not in ascending row_id order")
        self._intents = r.df["Intent"].astype(str).to_numpy()

    @property
    def row_ids(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in self._row_ids)

    @property
    def kb_version(self) -> str:
        return self.retriever.kb_version

    def encode(self, query: str):
        cleaned = clean_text(query)
        if not cleaned:
            return None
        return self.retriever._vectorizer.transform([cleaned])

    def scores(self, query_repr) -> np.ndarray:
        """Cosine score for every KB row (positional order = ascending row_id)."""
        if query_repr is None:
            return np.zeros(len(self._row_ids))
        return cosine_similarity(query_repr, self.retriever._matrix)[0]

    def search(self, query_repr, k: int, intent: Optional[str] = None) -> Ranking:
        if query_repr is None:
            return []
        if intent is None:
            positions = np.arange(len(self._row_ids))
        else:
            positions = self.retriever._intent_rows.get(intent)
            if positions is None:
                return []
        sims = self.scores(query_repr)[positions]
        keep = sims > 0.0
        positions, sims = positions[keep], sims[keep]
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(self._row_ids[positions[i]]), float(sims[i])) for i in order]

    def rank(self, query: str, k: int, intent: Optional[str] = None) -> Ranking:
        return self.search(self.encode(query), k, intent)


# ---------------------------------------------------------------------------
# Semantic retrieval (spec §6)
# ---------------------------------------------------------------------------
def order_with_tolerance(row_ids: Sequence[int], scores: Sequence[float], tol: float) -> List[int]:
    """
    Deterministic descending order where near-equal scores are tied (spec §6.1.3).

    Method: sort by descending score (stable over ascending row_id); then walk the
    sorted list, chaining consecutive items whose scores differ by <= tol into one
    tie group, and order each tie group by ascending row_id. Returns positions.
    """
    row_ids = np.asarray(row_ids)
    scores = np.asarray(scores, dtype=np.float64)
    base = np.lexsort((row_ids, -scores))  # primary: -score, secondary: row_id
    result: List[int] = []
    group = [int(base[0])] if len(base) else []
    for pos in base[1:]:
        pos = int(pos)
        if scores[group[-1]] - scores[pos] <= tol:
            group.append(pos)
        else:
            result.extend(sorted(group, key=lambda p: row_ids[p]))
            group = [pos]
    result.extend(sorted(group, key=lambda p: row_ids[p]))
    return result


def require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("B1 semantic runs require HF_HUB_OFFLINE=1 (no network at inference)")


def load_embedding_model():
    """Load the pinned MiniLM revision on CPU with the configured thread count."""
    require_offline()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(config.TORCH_THREADS)
    return SentenceTransformer(config.MODEL_NAME, revision=config.MODEL_REVISION, device="cpu")


def embed_texts(model, texts: Sequence[str]) -> np.ndarray:
    """Embed cleaned texts; unit-normalised float32 rows."""
    emb = model.encode(
        [clean_text(t) for t in texts],
        batch_size=config.ENCODE_BATCH_SIZE,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


class SemanticRetriever:
    """
    Dense retrieval: unit-normalised MiniLM embeddings, dot-product scores over the
    precomputed 160 x 384 float32 KB index (row order = ascending train row_id).
    Every KB row is scored, so a ranking always has min(k, candidates) rows.
    Ties within SEMANTIC_TIE_TOL resolve to ascending row_id (order_with_tolerance).
    """

    name = "b1_semantic_minilm"

    def __init__(self, train_df: pd.DataFrame, model=None, index: Optional[np.ndarray] = None):
        self._row_ids = np.array(sorted(int(i) for i in train_df.index), dtype=np.int64)
        intents = train_df.sort_index()["Intent"].astype(str).to_numpy()
        self._intent_positions = {
            it: np.flatnonzero(intents == it) for it in sorted(set(intents))
        }
        self.model = model if model is not None else load_embedding_model()
        if index is None:
            index = np.load(config.INDEX_PATH)
        self.index = validate_index(index, len(self._row_ids))
        verify_index_order(self.model, self.index, train_df)

    @property
    def row_ids(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in self._row_ids)

    def encode(self, query: str) -> np.ndarray:
        return embed_texts(self.model, [query])[0]

    def scores(self, query_repr: np.ndarray) -> np.ndarray:
        return self.index.astype(np.float64) @ np.asarray(query_repr, dtype=np.float64)

    def search(self, query_repr: np.ndarray, k: int, intent: Optional[str] = None) -> Ranking:
        if intent is None:
            positions = np.arange(len(self._row_ids))
        else:
            positions = self._intent_positions.get(intent)
            if positions is None:
                return []
        sims = self.scores(query_repr)[positions]
        order = order_with_tolerance(self._row_ids[positions], sims, config.SEMANTIC_TIE_TOL)[:k]
        return [(int(self._row_ids[positions[i]]), float(sims[i])) for i in order]

    def rank(self, query: str, k: int, intent: Optional[str] = None) -> Ranking:
        return self.search(self.encode(query), k, intent)


def verify_index_order(model, index: np.ndarray, train_df: pd.DataFrame, atol: float = 1e-4) -> None:
    """Spot-check that index rows follow ascending train row_id by re-embedding a few questions."""
    questions = train_df.sort_index()["User Message"].tolist()
    positions = sorted({0, len(questions) // 2, len(questions) - 1})
    fresh = embed_texts(model, [questions[p] for p in positions])
    if not np.allclose(fresh, index[positions], atol=atol):
        raise ValueError("KB index rows do not match ascending train row_id order")


def validate_index(index: np.ndarray, n_rows: int) -> np.ndarray:
    if index.shape != (n_rows, config.EMBEDDING_DIM):
        raise ValueError(f"Index shape {index.shape} != ({n_rows}, {config.EMBEDDING_DIM})")
    if index.dtype != np.float32:
        raise ValueError(f"Index dtype {index.dtype} != float32")
    norms = np.linalg.norm(index.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=config.UNIT_NORM_TOL):
        raise ValueError("Index rows are not unit-normalised")
    return index
