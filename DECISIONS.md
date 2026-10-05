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
