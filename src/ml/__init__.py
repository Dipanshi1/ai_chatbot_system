import sys
from pathlib import Path

# Ensure project root is in sys.path so both 'src.ml' and 'ml' can be imported anywhere
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from .config import (
    ABSTENTION_THRESHOLD,
    ARTIFACTS_DIR,
    BASE_DIR,
    DATA_DIR,
    MARGIN_THRESHOLD,
    MODEL_PATH,
    POLICY_VERSION,
    RAW_DATA_PATH,
    VECTORIZER_PATH,
)
from .inference import IntentClassifier, get_classifier, predict_intent
from .lexical import analyze_lexical, content_tokens, tokenize
from .pipeline import ChatPipeline, answer, get_pipeline
from .policy import DEFAULT_POLICY, decide, paraphrase_passes
from .preprocessing import clean_text
from .retrieval import ResponseRetriever, get_retriever
from .schemas import (
    AbstainReason,
    ChatResult,
    ClassifierResult,
    Decision,
    LexicalEvidence,
    Outcome,
    PolicyConfig,
    RetrievalCandidate,
)

__all__ = [
    "ABSTENTION_THRESHOLD",
    "ARTIFACTS_DIR",
    "BASE_DIR",
    "DATA_DIR",
    "MARGIN_THRESHOLD",
    "MODEL_PATH",
    "POLICY_VERSION",
    "RAW_DATA_PATH",
    "VECTORIZER_PATH",
    "IntentClassifier",
    "get_classifier",
    "predict_intent",
    "analyze_lexical",
    "content_tokens",
    "tokenize",
    "ChatPipeline",
    "answer",
    "get_pipeline",
    "DEFAULT_POLICY",
    "decide",
    "paraphrase_passes",
    "clean_text",
    "ResponseRetriever",
    "get_retriever",
    "AbstainReason",
    "ChatResult",
    "ClassifierResult",
    "Decision",
    "LexicalEvidence",
    "Outcome",
    "PolicyConfig",
    "RetrievalCandidate",
]
