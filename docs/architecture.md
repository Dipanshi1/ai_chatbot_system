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
| [`preprocessing.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/preprocessing.py) | Lowercasing, whitespace collapse, trimming. | **No** |
| [`lexical.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/lexical.py) | Token extraction, stopword filtering, vocabulary coverage assessment. | **No** |
| [`inference.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/inference.py) | Vectorization and LinearSVC scoring; computes top-1 minus top-2 margin. | **No** |
| [`retrieval.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/retrieval.py) | Candidate search filtered by predicted intent; cosine similarity and Dice overlap. | **No** |
| [`policy.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/policy.py) | Evaluates complete evidence against thresholds; emits final outcome and reason. | **YES (Sole Decision Authority)** |
| [`pipeline.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/pipeline.py) | Orchestration of evidence gathering, response assembly, and version stamps. | **No** |
| [`schemas.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/schemas.py) | Immutable dataclasses defining evidence, candidates, decisions, and results. | **No** |
| [`config.py`](file:///home/dipanshi/University%20AI%20Chatbot/src/ml/config.py) | Centralized paths, random seeds, threshold constants, and message templates. | **No** |

---

## 3. Training Pipeline & Artifact Flow

The training pipeline is isolated in `src/ml/training.py` and driven via `src/train.py`:

```text
data/raw/AI-Powered Chatbot.xlsx
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
   `ResponseRetriever.best_match()` queries only the subset of historical records matching the predicted intent. It calculates:
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
Is classification margin >= 0.20?
   NO  -> ABSTAINED (LOW_MARGIN)
   YES ->
Is lexical evidence weak? (<= 1 in-vocab content token AND OOV ratio >= 0.50)
   YES -> ABSTAINED (WEAK_LEXICAL_EVIDENCE)
   NO  ->
Does retrieval candidate meet paraphrase criteria?
(Cosine > 0 AND Dice >= 0.60 AND at least 1 shared specific content token)
   YES -> ANSWERED (Return official canned response)
   NO  -> INTENT_FALLBACK (Return department fallback message)
```

### Outcome Definitions:
- **`ANSWERED`**: High intent margin and verified paraphrase match. The matched response from the knowledge base is returned.
- **`INTENT_FALLBACK`**: High intent margin, but no reliable question match in the knowledge base. The system names the general topic without speculating on specific details.
- **`ABSTAINED`**: Ambiguous intent, low margin, or weak vocabulary match. The system declines to answer and requests clarification.

---

## 6. Streamlit Presentation Layer

The web interface in `app.py` is a **thin presentation layer**:
- **Zero Decision Logic**: `app.py` contains no ML logic, confidence thresholds, or routing heuristics. It calls `answer(user_input)` directly.
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
