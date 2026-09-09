# Email Spam Classification using Multinomial Naive Bayes

An end-to-end Machine Learning pipeline for email spam classification (Spam vs. Ham) using Natural Language Processing (NLP) techniques and a custom, from-scratch implementation of the Multinomial Naive Bayes algorithm.

## Overview
- **Task:** Binary Text Classification (Spam / Ham)
- **Dataset:** Kaggle Spam vs Ham Emails (~5,700 samples)
- **Core Model:** From-scratch Multinomial Naive Bayes with Laplace Smoothing
- **Evaluation:** Cross-Validation & Test Set Evaluation (Accuracy, Precision, Recall, F1-Score)

## Key Results
- **Test Accuracy:** 97.9%
- **F1-Score (Spam Class):** 95.7%
- **Recall (Spam Detection):** 98.2%
- **Precision:** 93.4%

## Architecture & Pipeline
1. **Text Preprocessing:** Regular expression cleaning, lowercasing, stop-phrase removal, and whitespace normalization.
2. **Feature Extraction:** Evaluated both CountVectorizer (Bag-of-Words) and TfidfVectorizer across multiple n-gram and vocabulary size configurations.
3. **Model Implementation:** Custom `MyMultinomialNaiveBayes` classifier computing log-priors and smoothed log-likelihood probabilities to prevent numerical underflow.
4. **Validation:** 3-fold cross-validation grid search to select the optimal feature representation.

## Tech Stack
- Python 3
- NumPy & Pandas
- Scikit-Learn (data splitting and evaluation metrics)
- Google Colab
