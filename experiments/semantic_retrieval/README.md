# B1 — Semantic Retrieval Experiment

**Status:** Phase 1 complete; benchmark human-reviewed (72/72 explicit decisions) and **frozen**
(sha256 `b80b9b95d1f085a1f1537077f72ae485da2ab2cdb6d12cbdc92928f2b184bd6b`, recorded in `config.py` and
`DECISIONS.md`). Phase 3 evaluation is **not implemented**: it is blocked until the authoritative
Revision 2 specification is available in the repository.
No retrieval evaluation has been run.

**Source of truth:** B1 Experimental Design Specification, **Revision 2 (final)**.
This is an isolated experiment; Stage A (`src/`, `artifacts/`, `tests/`, `reports/`, `app.py`,
Stage A requirements) is frozen and is not modified.

## Methodology summary

- **Question:** can dense semantic retrieval meaningfully improve retrieval for student-support
  queries with weak lexical overlap with the knowledge base, at the exact-row and intent level,
  without an unacceptable increase in wrong-intent retrieval?
- **Comparison:** the actual Stage A `ResponseRetriever(df=train_df)` (stopword-free TF-IDF,
  ranked through a read-only adapter with exact parity to `best_match`) vs.
  `sentence-transformers/all-MiniLM-L6-v2` (pinned revision, CPU, unit-normalised embeddings,
  dot-product search over a 160 × 384 float32 NumPy matrix).
  **No vector database or external retrieval server is required.**
- **Knowledge base:** the 160 canonical training rows only. The 40 Stage A test rows are never
  indexed, embedded, labelled or used as drafting seeds.
- **Benchmark:** 72 human-reviewed queries — 22 synonym substitution, 18 syntactic inversion,
  18 slang/informal (positive), 14 distractor/negative probes.
- **Primary population P\*:** positive queries for which no acceptable KB answer has Dice ≥ 0.60
  (`max_dice_to_acceptable` < 0.60, at least 40 queries). `max_dice_to_acceptable` is a non-gating
  benchmark-definition field; `dice_to_preferred` and `max_dice_to_any_kb_row` are diagnostics.
  No Dice field is a retrieval acceptance gate.
- **Primary evaluation:** Pass 1 (global search over the 160-row KB). Pass 2 (oracle intent) is
  diagnostic; Pass 3 (Stage A predicted intent) is integration context.
- **Primary metric:** AcceptableRecall@1. Engineering decision gates on P\* / Pass 1:
  ΔAcceptableRecall@1 ≥ +15 pp, ΔAcceptableMRR@10 ≥ +0.15, ΔWrongIntent@1 ≤ +5 pp.
  Paired percentile bootstrap (1,000 resamples, seed 42); Wilcoxon signed-rank as supporting
  evidence only. Resource gates: p95 end-to-end latency ≤ 50 ms, incremental peak RSS ≤ 500 MB.
- **Interpretation rule:** "semantic retrieval is better" is never claimed from raw Recall@1 alone
  (spec §13.4).

All pre-registered parameters live in [config.py](config.py).

## Layout

| Path | Purpose |
|---|---|
| `requirements-b1.txt` | B1-only dependencies (includes `../../requirements.txt`; torch pinned to `+cpu`) |
| `config.py` | Pre-registered parameters, paths, model revision |
| `lib/` | Shared code: data split & leakage guards, exact Dice, retrievers, metrics, statistics, resource probe, provenance, benchmark validation |
| `benchmark/build_benchmark.py` | `review-pack`, `annotate`, `validate`, `freeze`, `verify-freeze` |
| `benchmark/challenge_set_v1.csv` | Frozen challenge benchmark (72 human-verified queries) |
| `benchmark/challenge_set_v1_draft90.csv`, `human_review_decisions.json`, review/audit files | Audit evidence: original 90-row draft, recorded human decisions, review reports |
| `scripts/01_build_index.py` | Builds `artifacts/kb_train_embeddings.npy` (git-ignored, regenerable) |
| `scripts/02_run_eval.py`, `scripts/03_analyze_errors.py` | Phase 3 — not implemented (blocked on the authoritative specification) |
| `tests/test_semantic_retriever.py` | B1 unit tests |
| `results/` | Phase 3 outputs |

## Setup

From the repository root (Python 3.12):

```bash
python3 -m venv experiments/semantic_retrieval/.venv-b1
experiments/semantic_retrieval/.venv-b1/bin/pip install -r experiments/semantic_retrieval/requirements-b1.txt
```

Download the pinned model once (the only network use):

```bash
experiments/semantic_retrieval/.venv-b1/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
```

Every later command runs with `HF_HUB_OFFLINE=1`.

## Commands

Build the KB index:

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/scripts/01_build_index.py
```

Benchmark tool (`review-pack` | `annotate` | `validate` | `freeze` | `verify-freeze`):

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/benchmark/build_benchmark.py review-pack
```

B1 tests:

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python -m unittest discover -s experiments/semantic_retrieval/tests -v
```

Stage A tests must keep passing in the Stage A environment:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Findings

None yet. No evaluation has been run; no B1 results exist.
