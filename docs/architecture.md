# System Architecture & Technical Design

This document details the architectural design, component interactions, data flows, and safety mechanisms of the **HCL Student Support Chatbot**.

---

## 1. High-Level System Architecture

The chatbot is structured around a modular, feed-forward architecture where intent prediction, evidence gathering, and decision gating are strictly decoupled.

```text
                               +-----------------------------------+
                               |        User Input Query           |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     Text Normalization            |
                               |    (src.ml.preprocessing)         |
                               +-----------------------------------+
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
       +----------------------------+                          +----------------------------+
       |   Intent Classification    |                          |     Lexical Analysis       |
       |    (src.ml.inference)      |                          |     (src.ml.lexical)       |
       |  - TfidfVectorizer         |                          |  - Content tokenization    |
       |  - LinearSVC decision func |                          |  - In-vocab / OOV counts   |
       +----------------------------+                          +----------------------------+
                     |                                                       |
                     v                                                       |
       +----------------------------+                                        |
       |  Knowledge Base Retrieval  |                                        |
       |    (src.ml.retrieval)      |                                        |
       |  - Filter by intent        |                                        |
       |  - TF-IDF Cosine ranking   |                                        |
       |  - Content-token Dice      |                                        |
       +----------------------------+                                        |
                     |                                                       |
                     +---------------------------+---------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |       Safety Decision Policy      |  <-- Sole decision point
                               |        (src.ml.policy)            |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |         ChatResult Assembly       |
                               |        (src.ml.pipeline)          |
                               +-----------------------------------+
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
       +----------------------------+                          +----------------------------+
       |   Terminal CLI Interface   |                          |   Streamlit Web Interface  |
       |       (src/chatbot.py)     |                          |          (app.py)          |
       +----------------------------+                          +----------------------------+
```

---

## 2. Component Decoupling & Responsibilities

Each module under `src/ml/` possesses a single, well-defined responsibility with strict boundaries regarding decision-making:

| Module | Primary Responsibility | Makes Decisions? |
| :--- | :--- | :---: |
| [`preprocessing.py`](../src/ml/preprocessing.py) | Lowercasing, whitespace collapse, trimming. | **No** |
| [`lexical.py`](../src/ml/lexical.py) | Token extraction, stopword filtering, vocabulary coverage assessment. | **No** |
| [`inference.py`](../src/ml/inference.py) | Vectorization and LinearSVC scoring; computes top-1 minus top-2 margin. | **No** |
| [`retrieval.py`](../src/ml/retrieval.py) | Candidate search filtered by predicted intent; cosine similarity and Dice overlap. | **No** |
| [`policy.py`](../src/ml/policy.py) | Evaluates complete evidence against thresholds; emits final outcome and reason. | **YES (Sole Decision Authority)** |
| [`pipeline.py`](../src/ml/pipeline.py) | Orchestration of evidence gathering, response assembly, and version stamps. | **No** |
| [`schemas.py`](../src/ml/schemas.py) | Immutable dataclasses defining evidence, candidates, decisions, and results. | **No** |
| [`config.py`](../src/ml/config.py) | Centralized paths, random seeds, threshold constants, and message templates. | **No** |

---

## 3. Training Pipeline & Artifact Flow

The training pipeline is isolated in `src/ml/training.py` and driven via `src/train.py`:

```text
data/raw/AI-Powered Chatbot.xlsx  (canonical workbook; see Section 8)
       |
       v
load_dataset()  --> Schema validation (User Message, Intent, Bot Response)
       |
       v
split_dataset() --> Deterministic unstratified split (160 train, 40 test; seed=42)
       |
       v
clean_text()    --> Standardized query text
       |
       v
fit_vectorizer()--> TfidfVectorizer fitted strictly on 160 training rows
       |
       v
build_models()  --> Fits LogisticRegression, LinearSVC, MultinomialNB
       |
       +---------> In-Memory Evaluation (Default: writes nothing to disk)
       |
       +---------> Optional Persistence (--save --overwrite):
                     - artifacts/intent_model.pkl
                     - artifacts/tfidf_vectorizer.pkl
                     - artifacts/manifest.json (SHA-256 digests & environment metadata)
```

### Safety Features of the Training Pipeline
- **Default In-Memory Mode**: Executing `python src/train.py` runs entirely in memory without writing any files.
- **Protected Overwrite**: Artifacts can only be written to disk when `--save` is supplied; if canonical files already exist, `--overwrite` is explicitly required.
- **Automated Manifest Synchronization**: `write_manifest()` immediately computes and records the SHA-256 hashes of saved artifacts.

---

## 4. Inference & Response Retrieval Flow

During runtime inference (`src.ml.pipeline.ChatPipeline.answer`):
1. **Input Normalization**:
   The input string is passed through `clean_text()`. If the resulting string is empty, the pipeline immediately returns an `ABSTAINED` outcome with reason `EMPTY_INPUT`.
2. **Lexical Analysis**:
   `analyze_lexical()` tokenizes the query using the vectorizer's regex pattern, filters English stopwords to identify content tokens, and calculates the in-vocabulary vs out-of-vocabulary (OOV) ratio.
3. **Intent Classification**:
   `IntentClassifier.predict()` projects the cleaned text through `tfidf_vectorizer.pkl` and computes signed decision distances using `intent_model.pkl`. The top-1 predicted intent and decision margin ($\text{score}_{\text{top-1}} - \text{score}_{\text{top-2}}$) are recorded.
4. **Knowledge Base Retrieval**:
   `ResponseRetriever.best_match()` queries only the subset of records matching the predicted intent. At runtime the index is built directly from all 200 workbook rows (Section 8). It calculates:
   - Cosine similarity between TF-IDF representations.
   - Content-token Dice overlap coefficient:
     $$\text{Dice}(Q, M) = \frac{2 \times |Q \cap M|}{|Q| + |M|}$$
     Where $Q$ are query content tokens and $M$ are matched question content tokens.
   - Shared non-generic content tokens (excluding campus stop tokens like *student*, *university*, *help*).

---

## 5. Policy Decision Flow & Abstention

The decision logic in `src.ml.policy.decide()` executes deterministic threshold checks:

```text
Is input empty?
   YES -> ABSTAINED (EMPTY_INPUT)
   NO  ->
Is there at least 1 in-vocabulary content token?
   NO  -> ABSTAINED (NO_LEXICAL_EVIDENCE)
   YES ->
Is classification margin >= 0.20?
   NO  -> ABSTAINED (LOW_MARGIN)
   YES ->
Does retrieval candidate meet paraphrase criteria?
(Cosine > 0 AND Dice >= 0.60 AND at least 1 shared specific content token)
   YES -> ANSWERED (Return canned response)
   NO  ->
Is lexical evidence weak? (<= 1 in-vocab content token AND OOV ratio >= 0.50)
   YES -> ABSTAINED (WEAK_LEXICAL_EVIDENCE)
   NO  -> INTENT_FALLBACK (NO_RELIABLE_MATCH; names the likely support area)
```

### Outcome Definitions:
- **`ANSWERED`**: High intent margin and a paraphrase match. The matched response from the knowledge base is returned.
- **`INTENT_FALLBACK`**: High intent margin, but no reliable question match in the knowledge base. The system names the general topic without speculating on specific details.
- **`ABSTAINED`**: Ambiguous intent, low margin, or weak vocabulary match. The system declines to answer and requests clarification.

---

## 6. Streamlit Presentation Layer

The web interface in `app.py` is a **presentation layer around the pipeline**:
- **No ML Decision Logic**: `app.py` contains no ML logic or confidence thresholds. It applies the support layer (Section 9), then calls `pipeline.answer()` with redacted text unless a fixed triage reply applies.
- **Cached Loading**: Uses `@st.cache_resource` on `load_cached_pipeline()` to run `verify_artifacts()` and initialize `ChatPipeline` once per server session.
- **Input Guardrails**: Validates input length against `MAX_MESSAGE_CHARS = 500` before calling the pipeline.
- **Conversation State**: Manages dialogue history in `st.session_state.messages` and provides a "Clear Conversation" button.
- **Transparent Diagnostics**: Provides an expandable technical drawer exposing uncalibrated decision margins, cosine similarities, content-token overlap, and model versions.

---

## 7. Relationship Between Notebook and Production Pipeline

To maintain architectural integrity, the repository strictly delineates between the educational walkthrough and the production inference engine:

```text
+-----------------------------------------------------------+
|                      Shared Core                          |
|                       (src/ml/)                           |
+-----------------------------------------------------------+
               ^                             ^
               |                             | (Direct Dependency)
               |                             |
+------------------------------+  +------------------------------+
|     Educational Notebook     |  |    Production Interfaces     |
|   (notebooks/hcl/01_*.ipynb) |  |   - Streamlit (app.py)       |
|                              |  |   - CLI (src/chatbot.py)     |
| - Pedagogical explanations   |  |   - Eval CLI (src/evaluate)  |
| - Inline visual EDA          |  +------------------------------+
| - In-memory training demo    |
| - Static report inspection   |
+------------------------------+
```

### Key Distinctions:
1. **The Notebook is NOT a Production Implementation**: The notebook exists solely for educational walkthrough, visual inspection, and verification of scientific concepts.
2. **No Code Duplication**: The notebook imports data loading, preprocessing, in-memory training, and evaluation directly from `src.ml`.
3. **No Artifact Creation**: The notebook executes candidate model training strictly in memory; it never calls `save_artifacts()` and never overwrites canonical files.
4. **Reproducibility Bridge**: The notebook uses `check_saved_model_equivalence()` to prove that the educational in-memory training produces results identical (40/40) to the committed production artifacts.

---

## 8. Data Sources & Knowledge Base

`data/raw/AI-Powered Chatbot.xlsx` is the unified canonical source (200 rows, 10 columns). No separate knowledge-base file exists; `ResponseRetriever` reads the workbook directly.

```text
                 data/raw/AI-Powered Chatbot.xlsx
                                |
        +-----------------------+------------------------+
        |                       |                        |
        v                       v                        v
  Training source         Evaluation KB              Runtime KB
  160 rows                160 rows (train_df)        200 rows (all)
  User Message -> X       ResponseRetriever(df=...)  ResponseRetriever()
  Intent       -> y       metrics only               live answers, admin viewer
```

- **Training role**: only `User Message` and `Intent` from the 160 training rows reach the model. `Bot Response` is in `FORBIDDEN_FEATURE_COLUMNS`.
- **Runtime FAQ role**: retrieval matches on `User Message` and returns the paired `Bot Response`.
- **Row-count distinction**: published full-pipeline metrics (1 answered, 21 fallback, 18 abstained of 40) use the 160-row evaluation KB. The staff viewer defaults to the 200 runtime records and labels the 160-row evaluation subset separately. `get_pipeline()` builds `ResponseRetriever()` with no training-split filter, so the live index also contains the 40 held-out rows.

### Canned-Response Remediation

All 200 `Bot Response` entries were rewritten to remove offers the single-turn system could not fulfil (`DECISIONS.md`, 2026-10-31). Because `Bot Response` is not a feature, the remediation changed only returned text and the workbook-derived KB version labels:

| Item | Status against committed baseline |
| :--- | :--- |
| `Bot Response` | All 200 entries changed |
| `User Message`, `Intent`, `Topic`, all metadata | Identical |
| 160 training-row identities | Identical |
| Vectorizer vocabulary and IDF values | Identical |
| Classifier predictions and retrieval rankings | Unchanged |
| Serialized artifacts and manifest | Unchanged; no retraining |

### Separate HCL Dataset

The HCL-supplied dataset (`dataset.csv`, 500 rows, 6 columns; `session_id`, `user_query_type`, `intent`, `confidence_score`, `escalated`, `user_rating`; SHA-256 `cbf1fbd353623819c462ad76e4277c264b7dcfe41af5dc91ecb44c4d743f7526`) is separate from the workbook. It was downloaded from the internship course LMS, is not stored in this repository, is not loaded by any code, and contains no message text.

---

## 9. Student Support Layer

`src/support/` wraps the pipeline and imports nothing from `src.ml`, so it cannot change classification, retrieval or policy gates.

```text
Student message
   -> validate_user_message()
   -> redact()                  regex PII placeholders outside both vocabularies
   -> detect_signals()
        crisis         -> fixed guidance, pipeline skipped
        human request  -> escalation form, pipeline skipped
        opt-out phrase -> points to the sidebar toggle, pipeline skipped
        none           -> pipeline.answer(redacted text)
                          -> analyze_sentiment()
                          -> offer escalation if no reliable answer and negative
```

| Module | Responsibility |
| :--- | :--- |
| `redaction.py` | Pattern-based PII redaction; not exhaustive. |
| `triage.py` | Phrase-based crisis and request signals; not exhaustive. |
| `sentiment.py` | Lexicon heuristic with unmeasured accuracy; only informs escalation offers. |
| `store.py` | Local SQLite tickets and feedback with redaction on write and 30-day expiry. |
| `session.py` | Session opt-out, escalation and feedback state. |
| `admin_auth.py` | Single shared scrypt-hashed staff password; fails closed when unset. Not RBAC. |
| `analytics.py` | Aggregates from stored tickets and feedback only, not total chat volume. |
| `export.py` | Admin-only CSV export with redaction and formula neutralization. |

The chat is not monitored, tickets hold no contact details or transcripts, and the crisis guidance contains no phone numbers. `src/chatbot.py` calls the pipeline directly and does not use this layer.
