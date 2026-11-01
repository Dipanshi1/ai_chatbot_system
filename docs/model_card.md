# Model Card: HCL Student Support Intent Classifier

Following model card best practices, this document provides essential documentation on the training, intended usage, performance metrics, limitations, and safety considerations of the intent classification model deployed in the HCL Student Support Chatbot.

---

## 1. Model Details

- **Model Name**: HCL Student Support Intent Classifier (`linear_svm`)
- **Model Type**: Linear Support Vector Classifier (`sklearn.svm.LinearSVC`)
- **Feature Representation**: Term Frequency-Inverse Document Frequency (`sklearn.feature_extraction.text.TfidfVectorizer`)
  - Token Pattern: `(?u)\b\w\w+\b` (unigrams; default `ngram_range`)
  - Vocabulary Size: 360 features
  - Sublinear TF Scaling: False
  - Normalization: L2 Euclidean norm
- **Model Parameters**:
  - Penalty: L2
  - Loss: Squared Hinge
  - Dual: Auto
  - Tolerance: 1e-4
  - $C$: 1.0
  - Random State: 42 in the training pipeline; the saved artifact predates this setting and is prediction-equivalent (40/40) to a seeded refit, not bit-reproducible
- **Model Fingerprint**: `linsvc-a473f4580f16+tfidf-5f62a51adde8`
- **Model Artifact File**: `artifacts/intent_model.pkl` (SHA-256: `a473f4580f168c0ef0c7ea0c4812c4b1ca94aebce9f107d25d442c5fb30fa211`)
- **Vectorizer Artifact File**: `artifacts/tfidf_vectorizer.pkl` (SHA-256: `5f62a51adde831fbc634872c95672fd403d50b9c715266376315731d4de250f7`)
- **Date**: October 2026
- **License / Context**: Educational research baseline

---

## 2. Intended Use

- **Primary Intended Use**: Educational and demonstration baseline for automated routing of student support queries into 26 university operational intents, paired with safety-gated retrieval.
- **Intended Users**: University students and administrative staff exploring machine learning prototypes for support triage.
- **Intended Domain**: University campus administration (financial aid, academic advising, facilities, registration, IT support).

### Out-of-Scope & Prohibited Uses
- **High-Stakes Decision Making**: Do NOT use this model as an authoritative source of university policy, academic standing determinations, graduation verification, visa compliance, or disciplinary decisions.
- **Crisis or Emergency Intervention**: This model is NOT trained or equipped to handle mental health emergencies, physical safety crises, or immediate campus security reports. The Streamlit app adds phrase-based crisis guidance before the model runs; that is a non-exhaustive safety net, and the chat is not monitored.
- **Direct Unfiltered Response Generation**: The model does not generate text; predictions must always be evaluated through the safety policy before communicating answers to students.

---

## 3. Dataset Characteristics

- **Source**: `data/raw/AI-Powered Chatbot.xlsx`
- **Total Volume**: 200 interaction records
- **Number of Classes**: 26 fine-grained intent classes
- **Class Imbalance**: High variance in class frequency:
  - Dominant classes: `campus_facilities` (23 samples), `student_services` (20 samples)
  - Moderate classes: `academic_policies` (10 samples), `academic_advising` (10 samples)
  - Singleton classes: `admissions`, `exam_schedule`, `grades`, `library`, `schedule` (1 sample each)
- **Features Used**: `User Message` (sole feature)
- **Target Variable**: `Intent` (categorical label)
- **Precluded Features**: `Bot Response`, `Topic`, `Sentiment Score`, `Sentiment Label`, `Intent Confidence`, `User ID`, `Conversation ID`, `Timestamp`

---

## 4. Performance & Evaluation Metrics

All metrics reflect strictly held-out evaluation on the canonical 80/20 train/test split (160 train rows, 40 test rows; seed 42):

### Classification Performance (Test Set, N=40)
- **Accuracy**: **0.500000** (20 / 40 correct)
- **95% Wilson Score Confidence Interval**: **[0.351995, 0.648005]**
- **Macro F1 (Present Classes, N=18: labels found in test truth or predictions)**: **0.334127**
- **Macro F1 (All 26 Classes)**: **0.231319**
- **Weighted F1**: **0.510000**
- **Baseline Comparison (Majority Class Baseline)**: 0.175000 (7 / 40 correct)

### Repeated-Split Analysis (30 Seeds, Seeds 0–29)
- **Mean Test Accuracy**: **0.405833**
- **Standard Deviation**: **0.063884**
- **Accuracy Range**: **[0.250000, 0.525000]**
- **Median Accuracy**: **0.412500**
- **Relative Standing**: Linear SVM achieved the strictly best or tied-best accuracy in **30 of 30** repeated random splits.

### Full Chatbot Pipeline Performance (Train-Only Knowledge Base, N=40)
- **`ANSWERED` Outcome**: 1 / 40 (2.5%)
- **`INTENT_FALLBACK` Outcome**: 21 / 40 (52.5%)
- **`ABSTAINED` Outcome**: 18 / 40 (45.0%)
- **Committed Intent Accuracy**: **72.7273%** (16 / 22 correct among answered/fallback; 95% Wilson CI [0.518483, 0.868492])
- **Wrong-Intent Answers Delivered**: **0 / 40 (0.0%)**
- **Provenance**: stored 2026-10-05 reports from the pre-remediation workbook (model `linsvc-a473f4580f16+tfidf-5f62a51adde8`, policy `phase4-v1`, KB `kb-df-db0915f97b6b`). Answer correctness is not measured.

---

## 5. Limitations & Known Biases

1. **Extreme Data Sparsity**: With only 200 total examples, multiple classes have 1 or 2 training instances, severely limiting the statistical power of the classifier.
2. **High Sensitivity to Split Seed**: Performance ranges from 25.0% to 52.5% depending on which examples land in the test partition. Canonical seed 42 (50.0%) represents an optimistic upper-quartile split.
3. **No Calibrated Probabilities**: Linear SVM produces unbounded hyperplane distances (`decision_function`), not probabilities. The decision margin is an uncalibrated geometric heuristic.
4. **Lexical Brittleness**: Queries containing synonyms, misspellings, or phrasing absent from the 360-term training vocabulary experience degraded performance.
5. **Absence of Multi-Turn State**: The model evaluates each user utterance in complete isolation; it does not maintain conversational memory or contextual state.
6. **Static Response Knowledge**: Canned responses live in the `Bot Response` column of the canonical workbook. All 200 were remediated on 2026-10-31 without retraining, and the index rebuilds from the workbook at startup.
7. **Evaluation vs Runtime Knowledge Base**: Pipeline metrics use the 160-row training-split KB; the live chatbot searches all 200 workbook rows.

---

## 6. Safety & Gating Mechanisms

To mitigate the limitations above, the model output is strictly insulated by the three-tier policy in `src.ml.policy`:
- **Low Margin Gating**: If the difference between top-1 and top-2 LinearSVC scores is $< 0.20$, the system automatically abstains (`ABSTAINED: low_margin`).
- **Lexical Coverage Gating**: If a query contains $\le 1$ in-vocabulary content token and an OOV ratio $\ge 0.50$, the system refuses to guess (`ABSTAINED: weak_lexical_evidence`).
- **Paraphrase Verification**: An answer is only returned if the query achieves both positive cosine similarity and $\ge 0.60$ content-token Dice overlap with a knowledge-base entry. Otherwise, it names the likely support area (`INTENT_FALLBACK`) or abstains.

This design returned 0 wrong-intent answers across the 40 evaluation queries. That is a measured gating result on one small split, not a guarantee that answers are correct.
