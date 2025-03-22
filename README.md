# ANLP Slot Labeling Project

## Project Description

This project implements three sequence labeling models for slot filling:

1. **Most Frequent Tag (MLE)**: Baseline using maximum likelihood estimation
2. **Logistic Regression**: Uses SpaCy word embeddings
3. **Enhanced Model**: Logistic regression with linguistic features:
   - POS tags (one-hot encoded)
   - Dependency relations (one-hot encoded)
   - Lemmatized forms
   - Morphological features (binarized)
   - Context windows (±2 tokens)
   - Numeric modifier detection
   - Head token features

## Usage

```bash
# Baseline model
python label_slots.py -m most_frequent_tag

# Logistic regression with BIO constraints
python label_slots.py -m logistic_regression -p bio_tags

# Enhanced model with full features
python label_slots.py -m my_model -p bio_tags
```

## Model Architecture

**Feature Engineering:**

* 300D SpaCy word vectors
* POS & Dependency one-hot encoding
* Morphological feature binarization
* Contextual features from adjacent tokens
* Special handling of numeric modifiers
* Head token relationships

**BIO Enforcement:**

* Validates I- follows B/I-
* Automatic correction of invalid sequences

## Evaluation Metrics

1. **Token-level** :

* Precision/Recall/F1 for individual BIO tags

1. **Span-level** :

* Strict boundary matching
* Slot-level classification metrics

## Report Documents

* `assignment2.pdf`: Original task specifications
* `submission.pdf`: Technical report with implementation details
