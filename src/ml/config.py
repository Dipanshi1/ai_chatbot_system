import sys
from pathlib import Path

# Resolve the project root dynamically from this file's location
# File location: <project_root>/src/ml/config.py -> parent.parent.parent is <project_root>
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Ensure project root is in sys.path for portable imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Data directories and paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "AI-Powered Chatbot.xlsx"

# Artifact directories and paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "intent_model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Safety policy (Phase 4)
#
# Every threshold below is a prototype policy constant; not fitted; not optimal.
# None of them were tuned on the 40-example test split.
# ---------------------------------------------------------------------------

# Minimum top-1 minus top-2 LinearSVC decision_function margin.
# Uncalibrated margin, not a probability. 0.20 is a reasonable prototype
# heuristic on this single 40-sample test split.
# Prototype policy constant; not fitted; not optimal.
MARGIN_THRESHOLD = 0.20

# Backward-compatible alias (used by src/evaluate_confidence.py).
ABSTENTION_THRESHOLD = MARGIN_THRESHOLD

# Minimum Dice overlap of content tokens between the query and the matched
# knowledge-base question for a canned answer to be returned.
# Prototype policy constant; not fitted; not optimal.
PARAPHRASE_MIN_DICE = 0.60

# Weak lexical evidence: at most this many in-vocabulary content tokens AND
# at least this content OOV ratio. Downgrades INTENT_FALLBACK to ABSTAINED only.
# Prototype policy constants; not fitted; not optimal.
WEAK_EVIDENCE_MAX_IN_VOCAB_CONTENT = 1
WEAK_EVIDENCE_MIN_OOV_RATIO = 0.5

# Shared tokens that alone do not justify a paraphrase match.
# Prototype policy constant; not fitted; not optimal.
RETRIEVAL_GENERIC_TOKENS = frozenset(
    {"campus", "student", "students", "university", "bot", "help", "request"}
)

POLICY_VERSION = "phase4-v1"

# User-facing message templates
EMPTY_INPUT_MESSAGE = "Please enter a question so I can assist you."
CLARIFICATION_MESSAGE = (
    "I'm not sure what you're asking about. "
    "Could you rephrase or add a little more detail?"
)
INTENT_FALLBACK_TEMPLATE = (
    "I think this is about {area}, but I don't have enough information "
    "to give a reliable answer to that specific question yet."
)
