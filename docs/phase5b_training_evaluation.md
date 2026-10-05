# Phase 5B — Training and Evaluation Methodology

## 1. Executive Summary & Problem Scope

This document details the training and evaluation framework for the University Student Support AI Chatbot implemented in Phase 5B. The primary objective of Phase 5B is to establish an honest, reproducible, and leakage-free evaluation standard for intent classification and retrieval-based answer gating.

### Key Dataset Characteristics
- **Total Dataset Size**: Exactly 200 interaction rows.
- **Number of Intent Classes**: 26 distinct categories.
- **Severe Class Imbalance**: Intent support ranges from 1 example (singleton classes such as `grades`, `housing`, `mental_health`, `schedule`, `study_abroad`) to 23 examples (`campus_facilities`).
- **Canonical Evaluation Split**: 80/20 train/test split (160 training rows, 40 testing rows) with `SPLIT_RANDOM_STATE = 42`.
- **Stratification Rationale**: Stratified splitting is intentionally omitted because scikit-learn cannot stratify on classes with support < 2 (singletons), and force-stratifying would distort the distribution of small classes.

---

## 2. Model Training Architecture

The training library (`src/ml/training.py`) implements clean, in-memory training adhering to the following constraints:

1. **Feature Source**: Only `User Message` is extracted as the text feature. All other columns (`Intent Confidence`, `Topic`, `Sentiment Score`, `Sentiment Label`, `Bot Response`, `User ID`, `Conversation ID`, `Timestamp`) are explicitly forbidden from the classifier feature space.
2. **Preprocessing**: Input text is processed strictly using the frozen `clean_text()` preprocessor (lowercasing, whitespace normalization).
3. **TF-IDF Vectorization**: `TfidfVectorizer()` is fitted exclusively on the 160 training samples. Default scikit-learn settings are preserved.
4. **Candidate Models**:
   - **Logistic Regression**: `LogisticRegression(max_iter=1000, random_state=42)`
   - **Linear SVM**: `LinearSVC(random_state=42)` (Selected canonical model)
   - **Multinomial Naive Bayes**: `MultinomialNB()`
5. **No Optimization Claims**:
   - No hyperparameter tuning was conducted.
   - No confidence calibration (e.g. Platt scaling, isotonic regression) was applied.
   - No class-weight balancing (`class_weight=None`) was used.
   - No resampling (oversampling or undersampling) was performed.
   - No synthetic data generation was used.
   - No claim of optimal performance is made.

---

## 3. Canonical Evaluation Results (Seed 42, N = 40)

Evaluated on the single 40-sample canonical test split:

| Model | Correct / Total | Accuracy | 95% Wilson CI | Macro F1 (Present) | Macro F1 (All 26) | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** (Selected) | **20 / 40** | **0.500000** | [0.351995, 0.648005] | **0.334127** | **0.231319** | **0.510000** |
| **Logistic Regression** | 17 / 40 | 0.425000 | [0.285094, 0.578049] | 0.237061 | 0.145884 | 0.373772 |
| **Multinomial NB** | 15 / 40 | 0.375000 | [0.242230, 0.529676] | 0.119737 | 0.073684 | 0.286579 |
| **Majority Baseline** | 7 / 40 | 0.175000 | [0.087454, 0.319500] | 0.018617 | 0.011457 | 0.052128 |

### Notes on Metrics:
- **Majority Baseline**: In the 160-sample training set, `campus_facilities` and `student_services` tie at 16 instances each. Alphabetical tie-breaking selects `campus_facilities`, yielding 7/40 (17.5%) test accuracy (`student_services` would achieve 4/40 = 10.0%).
- **Macro F1 Present vs. All Classes**:
  - `macro_f1_present` evaluates only the 16–18 classes actually present in the test ground truth or predicted by the model.
  - `macro_f1_all_classes` evaluates across all 26 classes with `zero_division=0`, properly penalizing unrepresented classes.

---

## 4. Supplementary Repeated-Split Analysis (30 Seeds)

The canonical test split of 40 samples produces wide 95% Wilson confidence intervals (e.g. [0.352, 0.648] for Linear SVM). To quantify split variance without leaking test data, a supplementary analysis over 30 independent random train/test splits (seeds 0 to 29) was performed in memory:

| Model | Mean Accuracy | Std Dev (ddof=1) | Min | Median | Max | Seeds Best (Strict) | Seeds Best or Tied | At/Above Canonical (0.500) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **0.405833** | **0.063884** | **0.250000** | **0.412500** | **0.525000** | **29 / 30** | **30 / 30** | **10.0% (3/30)** |
| **Logistic Regression** | 0.258333 | 0.068018 | 0.125000 | 0.275000 | 0.375000 | 0 / 30 | 0 / 30 | 0.0% (0/30) |
| **Multinomial NB** | 0.206667 | 0.077663 | 0.075000 | 0.225000 | 0.375000 | 0 / 30 | 1 / 30 | 3.3% (1/30) |

### Key Insight:
- The canonical Seed 42 accuracy of **0.500** for Linear SVM is in the **upper decile** of the empirical split distribution (mean = 0.406). Seed 42 is an optimistic split.
- Repeated-split analysis is strictly **supplementary context** to communicate sampling variance. It is **not cross-validation** and was not used to select hyperparameters or models.

---

## 5. Leakage-Safe Pipeline Evaluation (Train-Only Knowledge Base)

### Why Full-KB Evaluation is Leaky
If the retrieval knowledge base contains all 200 rows of the dataset, test queries can retrieve the exact question-response pair from the knowledge base, artificially inflating paraphrase match rates and masking retrieval limitations. Evaluating a pipeline against a knowledge base that contains held-out test data is severe data leakage.

### Train-Only Knowledge Base Protocol
To evaluate the end-to-end chatbot pipeline honestly:
1. The retrieval index is constructed exclusively from the **160 training rows** (`ResponseRetriever(df=train_df)`).
2. Strict automated leakage assertions are verified prior to scoring:
   - KB size == 160
   - KB size < 200
   - Test indices in KB == 0
   - Test question overlap == 0
   - Test response overlap == 0
3. All 40 canonical test queries are evaluated through `ChatPipeline`:

### Pipeline Outcomes on Canonical Test Split (N = 40):
- **ANSWERED (Canned Answer Returned)**: 1 query (2.5%)
- **INTENT_FALLBACK (Area Identified, No Match)**: 21 queries (52.5%)
- **ABSTAINED (Clarification Requested)**: 18 queries (45.0%)
- **Intent Coverage (ANSWERED + FALLBACK)**: 22 / 40 (55.0%)

### Safety Gating Integrity:
- **Wrong-intent ANSWERED**: **0** (The chatbot never returned a canned response under an incorrect intent).
- **Wrong-area FALLBACK**: **6** (6 queries fell back to an intent area that did not match the ground-truth intent).
- **Committed Intent Accuracy**: **16 / 22 (72.7%)** with 95% Wilson CI [0.518, 0.868].
- **Predicted Intent Accuracy**: **20 / 40 (50.0%)** with 95% Wilson CI [0.352, 0.648].
- **Held-Out Response Returned**: **0** (No test sample ever retrieved its own held-out answer).

---

## 6. Saved Model Equivalence

The committed canonical model (`artifacts/intent_model.pkl`) was trained during earlier project phases before `random_state=42` was pinned explicitly in constructor arguments.

- **Equivalence Status**: **Prediction-Equivalent**
- **Test Set Agreement**: **40 / 40 (100.0%)** predictions match between the saved model and in-memory seeded refit.
- **Maximum Absolute Weight Difference**: < 1.6e-5 (well within the 1e-4 tolerance).
- **Maximum Intercept Difference**: < 1.3e-5.
- **Vocabulary & IDF**: Exact match.
- **Decision**: The canonical model is **NOT overwritten**. The correct scientific claim is "prediction-equivalent", not "bit-reproducible".

---

## 7. Limitations, Disclaimers, and Explicit Negative Claims

1. **Answer Correctness Is Not Measured**:
   - The evaluation framework measures intent accuracy, paraphrase similarity, and gate transitions. It does **not** score semantic factual correctness of answers beyond knowledge base retrieval matching.
2. **Small Dataset & High Variance**:
   - With $N = 200$ and $N_{test} = 40$, sampling variance is substantial. Point estimates should always be interpreted alongside confidence intervals.
3. **No Optimization Claims**:
   - Model thresholds (`MARGIN_THRESHOLD = 0.20`, `PARAPHRASE_MIN_DICE = 0.60`) are prototype heuristic constants established in Phase 4. They were **not tuned** on test splits or cross-validation folds.
4. **No Calibration**:
   - Linear SVM decision margins are raw geometric distances, not calibrated probabilities.
