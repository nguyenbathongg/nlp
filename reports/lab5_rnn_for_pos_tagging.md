# Lab 5 Report: RNN for Part-of-Speech Tagging

**Date:** November 17, 2025

---

## 1. Implementation Steps

Implements RNN for POS tagging using Universal Dependencies English-EWT dataset (12,544 train sentences, 2,001 dev, 17 POS tags).

### Task 1: Data Loading
- **`load_conllu()`**: Parses CoNLL-U format, extracts word (column 2) and tag (column 4), filters comments/multi-word tokens
- **`build_vocab()`**: Creates word_to_ix (19,674 words + `<UNK>`), tag_to_ix (17 tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X)

### Task 2: Dataset & DataLoader
- **`POSDataset`**: Converts (word, tag) pairs to tensor indices, handles `<UNK>` for unknown words
- **`collate_fn`**: Pads variable-length sequences using `pad_sequence`, padding_value=0
- DataLoaders: 392 train batches, 63 dev batches (batch_size=32)

### Task 3: Model Architecture
```python
Embedding(19,674 → 100) → RNN(100 → 128) → Linear(128 → 17)
Total parameters: 1,999,033
```

### Task 4: Training
- Optimizer: Adam (lr=0.001), Loss: CrossEntropyLoss(ignore_index=0)
- Training loop: zero_grad → forward → loss → backward → step
- 10 epochs, saves best model to `best_pos_model.pt`

### Task 5: Evaluation
- Token-level accuracy (ignores padding)
- Per-tag classification report (precision/recall/F1)
- `predict_sentence()` function for raw text input

---

## 2. Code Execution Guide

**Install:**
```bash
pip install torch numpy scikit-learn matplotlib
```

**Run:**
```bash
jupyter notebook lab5_pos_tagging.ipynb
# Click: Cell -> Run All (total: ~6 minutes)
```

---

## 3. Result Analysis

### Overall Performance

| Metric | Train | Dev |
|--------|-------|-----|
| Accuracy | 97.41% | 89.59% |
| Loss | 0.0851 | 0.4122 |

**Training Progress:** Accuracy improved from 79.12% (epoch 1) to 89.59% (epoch 10). Dev loss increased after epoch 8 (0.3847 to 0.4122), indicating slight overfitting.

### Per-Tag Accuracy

**High Accuracy (>92%):** PUNCT (95.2%), DET (94.1%), ADP (93.8%), PRON (92.7%)  
**Medium Accuracy (85-92%):** NOUN (91.3%), VERB (90.8%), ADJ (88.2%), AUX (87.5%)  
**Low Accuracy (<85%):** X (76.4%), SYM (78.9%), NUM (82.1%)

**Common Errors:**
- ADJ and NOUN confusion: "American" can be both
- ADP and SCONJ confusion: "as" ambiguous between preposition/conjunction
- VERB and NOUN confusion: "run" depends on context

### Why RNN Works

**Strengths:**
- Sequential context: Captures left-to-right word dependencies
- Hidden state memory: Previous words inform current predictions
- Parameter sharing: Generalizes across all positions
- Sufficient data: 12.5K training sentences

**Limitations:**
- Unidirectional: Only sees past context
- Vanishing gradient: Long sequences lose early information
- Vanilla RNN: Weaker than LSTM/BiLSTM-CRF (95-97% SOTA)

---

## 4. Challenges and Solutions

1. **Variable-Length Sequences:** Custom `collate_fn` with `pad_sequence`, `ignore_index=0` in loss
2. **Overfitting:** Early stopping saves best model on dev accuracy
3. **Unknown Words:** `<UNK>` token handles OOV words (3-5% of dev set)
4. **Class Imbalance:** Rare tags (X, SYM) have lower accuracy due to limited training data

---

## 5. References

