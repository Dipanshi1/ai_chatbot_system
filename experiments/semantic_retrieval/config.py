"""
B1 Semantic Retrieval Experiment — pre-registered parameters.

Source of truth: B1 Experimental Design Specification, Revision 2 (final).
Every threshold here is an engineering decision gate or prototype target for this
project's benchmark, not a universal scientific criterion. These values are fixed
before any evaluation run (spec §2.2.5) and must not be tuned on results.
"""
from fractions import Fraction
from pathlib import Path

import psutil

SPEC_VERSION = "B1 Revision 2 (final)"
METRIC_DEFINITION_VERSION = "b1-rev2"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent

BENCHMARK_DIR = EXPERIMENT_DIR / "benchmark"
BENCHMARK_PATH = BENCHMARK_DIR / "challenge_set_v1.csv"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "kb_train_embeddings.npy"
RESULTS_DIR = EXPERIMENT_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "b1_head_to_head_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "b1_retrieval_comparison.csv"
FAILURE_CSV_PATH = RESULTS_DIR / "b1_failure_analysis.csv"
REQUIREMENTS_B1_PATH = EXPERIMENT_DIR / "requirements-b1.txt"
DECISIONS_PATH = REPO_ROOT / "DECISIONS.md"
# Reviewer aid lives outside the repository on purpose.
REVIEW_PACK_PATH = Path("/tmp/b1_review_pack.md")

# Stage A files whose hashes must never change during B1.
STAGE_A_PROTECTED_FILES = (
    REPO_ROOT / "artifacts" / "intent_model.pkl",
    REPO_ROOT / "artifacts" / "tfidf_vectorizer.pkl",
    REPO_ROOT / "artifacts" / "manifest.json",
    REPO_ROOT / "data" / "raw" / "AI-Powered Chatbot.xlsx",
)

# ---------------------------------------------------------------------------
# Embedding model (spec §5.3, §5.5)
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIM = 384
NORMALIZE_EMBEDDINGS = True
ENCODE_BATCH_SIZE = 32
_PHYSICAL_CORES = psutil.cpu_count(logical=False) or 1
TORCH_THREADS = min(4, _PHYSICAL_CORES)

# ---------------------------------------------------------------------------
# Ranking (spec §4, §6.1, §7)
# ---------------------------------------------------------------------------
SEED = 42
TOP_K_DEPTHS = (1, 3, 5)
MRR_DEPTH = 10
RETRIEVAL_DEPTH = 10  # rankings are produced to depth 10 (max of K and MRR depth)
SEMANTIC_TIE_TOL = 1e-6
PARITY_COSINE_TOL = 1e-12
UNIT_NORM_TOL = 1e-5

# ---------------------------------------------------------------------------
# Benchmark composition (spec §3.1, §3.3)
# ---------------------------------------------------------------------------
BENCHMARK_SIZE = 72
CATEGORY_COUNTS = {
    "synonym_substitution": 22,
    "syntactic_inversion": 18,
    "slang_informal": 18,
    "distractor_negative": 14,
}
POSITIVE_CATEGORIES = ("synonym_substitution", "syntactic_inversion", "slang_informal")
NEGATIVE_CATEGORY = "distractor_negative"
PRIMARY_MIN = 40
ELIGIBLE_MIN_TRAIN_EXAMPLES = 3
INTENT_COVERAGE_RANGE = (12, 15)  # "where feasible" — reported, not enforced
REVIEW_STATUSES = ("draft", "verified")
BENCHMARK_COLUMNS = (
    "challenge_id",
    "query_text",
    "target_intent",
    "acceptable_row_ids",
    "preferred_row_id",
    "difficulty_category",
    "dice_to_preferred",
    "max_dice_to_acceptable",
    "max_dice_to_any_kb_row",
    "dice_bucket",
    "review_status",
    "notes",
)
DICE_DECIMALS = 6

# ---------------------------------------------------------------------------
# Lexical-overlap population and buckets (spec §3.4, R2-1)
# Canonical variable: max_dice_to_acceptable (non-gating benchmark-definition field).
# Exact rational boundaries; comparisons are exact.
# ---------------------------------------------------------------------------
PRIMARY_DICE_LT = Fraction(3, 5)  # 0.60: P* = positive queries with max_dice_to_acceptable < 0.60
BUCKET_EDGES = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))  # 0.25, 0.50, 0.75
B3_SPLIT = Fraction(3, 5)  # B3a: 0.50 <= D < 0.60 (in P*); B3b: 0.60 <= D < 0.75

# ---------------------------------------------------------------------------
# Decision gates (spec §13) — engineering gates, not universal scientific claims
# ---------------------------------------------------------------------------
T1_PP = 15       # ΔAcceptableRecall@1 >= +15 percentage points (primary)
T2_MRR = 0.15    # ΔAcceptableMRR@10 >= +0.15
S_PP = 5         # ΔWrongIntent@1 <= +5 percentage points (safety guardrail)
WILCOXON_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Statistics (spec §8.2)
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI = (2.5, 97.5)  # percentile method
WILCOXON = {"alternative": "two-sided", "zero_method": "wilcox"}

# ---------------------------------------------------------------------------
# Resources (spec §11) — measured, never assumed
# ---------------------------------------------------------------------------
LATENCY_CALLS = 100
LATENCY_P50_TARGET_MS = 20
LATENCY_P95_LIMIT_MS = 50          # gating
INCREMENTAL_RSS_LIMIT_MB = 500     # gating
RESOURCE_REPEATS = 3
# Reported, non-gating targets/limits preserved from Revision 1
MODEL_LOAD_TARGET_S, MODEL_LOAD_LIMIT_S = 1.0, 2.5
INDEX_BUILD_TARGET_S, INDEX_BUILD_LIMIT_S = 0.5, 1.5
INDEX_SIZE_TARGET_BYTES, INDEX_SIZE_LIMIT_BYTES = 500 * 1024, 2 * 1024 * 1024

# ---------------------------------------------------------------------------
# Failure-mode signatures (spec §10)
# ---------------------------------------------------------------------------
F1_COSINE_GT = 0.75
F4_COSINE_LT = 0.35
F3_MAX_TOKENS = 2

# ---------------------------------------------------------------------------
# Benchmark freeze (set in Phase 2 by build_benchmark.py freeze)
# ---------------------------------------------------------------------------
BENCHMARK_SHA256 = "b80b9b95d1f085a1f1537077f72ae485da2ab2cdb6d12cbdc92928f2b184bd6b"
