# Phase 5E Engineering Decisions

This document records architectural, documentation, and verification decisions established during Phase 5E (Final Documentation & Fresh-Clone Verification).

---

## 1. Documentation Structure & Terminology
- **Decision**: Adhere strictly to the requested 21-section structure for `README.md`.
- **Terminology**: Never refer to the chatbot as an LLM, RAG, or deep learning system. Describe the model accurately as a classical `LinearSVC` intent classifier over `TfidfVectorizer` paired with lexical overlap retrieval and deterministic policy gating.
- **Acronym Policy**: The term "HCL" is preserved as "HCL Student Support Chatbot / Workflow" without expanding it into any unverified or invented acronym.

## 2. Distinction Between Educational Walkthrough and Production Engine
- **Decision**: In `README.md` and `docs/architecture.md`, explicitly separate `notebooks/hcl/01_hcl_student_support_workflow.ipynb` from `src/ml/`.
- **Rationale**: The notebook is an educational, reproducible companion designed for transparent walkthroughs. It calls tested library functions directly and runs candidate training in memory. It is not an alternative or competing production implementation.

## 3. Fresh-Clone Verification Environment
- **Decision**: To verify fresh clone usability without disturbing the project's existing `.venv`, create a temporary virtual environment at `/tmp/hcl-chatbot-test-venv`.
- **Scope**: Install dependencies from `requirements.txt`, run the full test suite (`python -m unittest discover -s tests -v`), run `python src/evaluate.py`, and test headless Streamlit startup (`streamlit run app.py --server.headless true`). Remove the temporary environment after verification.

## 4. Protected Canonical Artifacts
- **Decision**: The existing canonical artifacts (`artifacts/intent_model.pkl`, `artifacts/tfidf_vectorizer.pkl`, and `artifacts/manifest.json`) are immutable release baselines. Normal training (`python src/train.py`) executes entirely in memory. Disk writes require `--save` and `--overwrite` flags to guard against accidental replacement.

## 5. Security & Secret Exposure
- **Decision**: Confirm via `git grep` and `git ls-files` that no API keys, credentials, tokens, or environment files are tracked. Document the zero-secret status in release verification reports.

---

## B1 Pre-registration & Benchmark Freeze (2026-10-28)
- **Specification**: B1 Revision 2 (final); metric definitions `b1-rev2`.
- **Benchmark**: `experiments/semantic_retrieval/benchmark/challenge_set_v1.csv`, sha256 `b80b9b95d1f085a1f1537077f72ae485da2ab2cdb6d12cbdc92928f2b184bd6b`.
- **Population counts**: Q=72, P=58, P*=58, N=14;
  slices {'B1': 36, 'B2': 20, 'B3a': 2, 'NEG': 14}; categories {'distractor_negative': 14, 'slang_informal': 18, 'synonym_substitution': 22, 'syntactic_inversion': 18};
  intents {'academic_advising': 7, 'academic_policies': 7, 'academic_support': 4, 'accessibility': 4, 'administration': 6, 'campus_facilities': 7, 'career_services': 3, 'course_information': 1, 'course_registration': 4, 'financial_aid': 5, 'housing': 6, 'mental_health': 5, 'research_support': 3, 'safety': 4, 'student_services': 2, 'technical_issue': 4}.
- **Canonical lexical-overlap variable**: `max_dice_to_acceptable` — a non-gating benchmark-definition
  field. P* = positive queries with max_dice_to_acceptable < 0.60.
  Buckets B1 < 0.25 <= B2 < 0.50 <= B3 < 0.75 <= B4; B3 split at
  0.60 into B3a (in P*) and B3b. `dice_to_preferred` and
  `max_dice_to_any_kb_row` are purely diagnostic. No Dice field is a retrieval acceptance gate.
- **Primary comparison**: Stage A `ResponseRetriever(df=train_df)` vs `sentence-transformers/all-MiniLM-L6-v2`
  (revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) on P*, Pass 1 (global search over the 160 training rows).
- **Decision gates (engineering, not universal claims)**: T1 ΔAcceptableRecall@1 >= +15 pp;
  T2 ΔAcceptableMRR@10 >= +0.15; S ΔWrongIntent@1 <= +5 pp on P*.
- **Outcome categories (assigned in order)**: Regression; Meaningful improvement; Modest improvement;
  No meaningful difference / inconclusive — per spec §13.2.
- **Statistics**: paired bootstrap, 1000 resamples, seed 42,
  percentile CI (2.5, 97.5); Wilcoxon signed-rank {'alternative': 'two-sided', 'zero_method': 'wilcox'} (supporting only).
- **Resource gates**: end-to-end p95 <= 50 ms (p50 target <= 20 ms);
  incremental peak RSS <= 500 MB; 3 fresh-process repeats.
- **Environment pins**: `requirements-b1.txt` sha256 `2fd8c97f2be269cbfdedc83653617be888e1ae82492652546e2373c71dc3a88b`.

---

## Stage A UI error-handling fix (2026-10-28)
- **Decision (user-authorized submission fix)**: `app.py` no longer renders raw exception text. Query
  failures show only the generic error message; startup failures show a generic message with the local
  verification command. Details are logged server-side via `logging`.
- **Rationale**: user-facing output must not expose stack traces, paths or internal details
  (`CLAUDE.md` §7). No ML behaviour, artifacts or dependencies changed.
- **Tests**: new `tests/test_app_e2e.py` (Streamlit `AppTest`): startup, supported query, fallback,
  over-long input, example button + clear, and two leak-prevention tests (query and startup failures).
