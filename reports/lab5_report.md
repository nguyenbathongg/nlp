# Lab 5 Report: Text Classification with Supervised Learning

**Date:** November 10, 2025

---

## 1. Implementation Steps

This lab implements text classification with supervised learning using Twitter Financial News Sentiment dataset (9,543 tweets: Bearish, Bullish, Neutral).

### Task 1: TextClassifier Class
**File:** `src/models/text_classifier.py`

Implemented `TextClassifier` with 4 methods:
- `__init__(vectorizer)`: Initialize with vectorizer
- `fit(texts, labels)`: Train LogisticRegression (solver='lbfgs', max_iter=1000)
- `predict(texts)`: Return predicted labels
- `evaluate(y_true, y_pred)`: Calculate accuracy, precision, recall, F1 (weighted average)

### Task 2: Basic Test
**File:** `test/lab5_test.py`

- Load dataset → Split 80/20 → Train with CountVectorizer → Predict → Evaluate
- Uses RegexTokenizer from Lab 1

### Task 3: Spark ML Pipeline
**File:** `test/lab5_spark_sentiment_analysis.py`

5-stage pipeline: Tokenizer → StopWordsRemover → HashingTF (10k features) → IDF → LogisticRegression

### Task 4: Model Improvements
**File:** `test/lab5_improvement_test.py`

Implemented 3 improvements:
1. **TfidfVectorizer** (`src/representations/tfidf_vectorizer.py`): Custom TF-IDF with L2 normalization
2. **Word2Vec**: GloVe embeddings (glove-wiki-gigaword-50), document = average word vectors
3. **Naive Bayes**: MultinomialNB with TF-IDF features

---


## 2. Code Execution Guide

**Prerequisites:**
```bash
pip install scikit-learn datasets pyspark gensim
```

**Run Basic Test (Task 1 & 2):**
```bash
python test/lab5_test.py
```

**Run Spark Pipeline (Task 3):**
```
python test/lab5_spark_sentiment_analysis.py
```

**Run Improvement Experiments (Task 4):**
```bash
python test/lab5_improvement_test.py
```

---

## 3. Result Analysis

### Performance Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Baseline (Count+LR)** | **82.08%** | **81.38%** |
| Spark Pipeline | 68.47% | 68.82% |
| TF-IDF + LR | 68.83% | 61.72% |
| Word2Vec + LR | 71.03% | 66.85% |
| TF-IDF + Naive Bayes | 67.78% | 59.74% |

### Analysis

**Baseline Won (82.08%):**
- Financial tweets have domain-specific terms ($AAPL, $TSLA) → Count features preserve exact matches
- Company names are highly predictive → TF-IDF down-weights them
- Short texts (tweets) suit bag-of-words better than complex embeddings

**Why Others Underperformed:**

1. **TF-IDF (-13.25%)**: Down-weighted important financial terms (company names, stock symbols)
2. **Word2Vec (-11.05%)**: GloVe trained on Wikipedia, lacks financial domain knowledge
3. **Naive Bayes (-14.30%)**: Independence assumption violated, sensitive to 65% Neutral class imbalance
4. **Spark (-13.61%)**: HashingTF collisions, fewer iterations (maxIter=10)

**Key Insight:** Simpler models (Count+LR) outperform complex ones when vocabulary is domain-specific and features are highly predictive.

---

## 4. Challenges and Solutions

**Challenge 1: Java 24 incompatible with PySpark**
- Solution: Set JAVA_HOME to Java 17 before running Spark

**Challenge 2: CSV data had invalid sentiment values**
- Solution: Filter with regex `^-?[0-9]+$` before casting to int

**Challenge 3: TfidfVectorizer missing `fit()` method**
- Solution: Added `fit()` that calls `fit_transform()` internally

**Challenge 4: WordEmbedder doesn't accept tokenizer**
- Solution: Store tokenizer separately, manually tokenize in Word2VecClassifier

**Challenge 5: Imbalanced dataset (65% Neutral)**
- Solution: Use weighted metrics (precision, recall, F1) instead of macro averaging

---

## 5. References

- **scikit-learn**: Machine learning library - https://scikit-learn.org/
- **PySpark MLlib**: Spark machine learning library - https://spark.apache.org/mllib/
- **HuggingFace datasets**: Twitter Financial News Sentiment - https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
- **Gensim**: Word2Vec and GloVe embeddings - https://radimrehurek.com/gensim/
