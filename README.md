# Hate Speech Classification

## Overview
This project implements a machine learning pipeline to classify tweets as containing hate speech or not. It demonstrates text preprocessing, feature extraction using various vectorization techniques, and evaluation of different classification models.

## Dataset
The dataset consists of labeled tweets collected from Twitter, with binary classification:
- `0`: No hate speech
- `1`: Contains hate speech

The dataset is imbalanced, with significantly fewer hate speech examples than non-hate speech ones.

## Project Structure
- Data loading and exploration
- Data cleaning and preprocessing
- Feature extraction using various vectorization methods
- Model training and evaluation
- Hyperparameter tuning

## Preprocessing Pipeline
The project uses custom scikit-learn transformers for consistent preprocessing:
1. **TextCleaner**: Handles text cleaning operations including:
   - HTML entity decoding
   - Username removal
   - URL removal
   - Emoji conversion
   - Case normalization
   - Number replacement
   - Special character removal

2. **Vectorizer**: Implements multiple text vectorization strategies:
   - Bag of Words (BOW)
   - TF-IDF
   - Word2Vec
   - GloVe
   - FastText
   - CNN-based embeddings

## Models Evaluated
- Random Forest Classifier
- Gradient Boosting Classifier
- Gaussian Naive Bayes

## Key Findings
1. The dataset shows significant class imbalance
2. Different vectorization techniques yield varying results depending on the model
3. TF-IDF vectorization with Random Forest classifier achieved the best Macro F1 score
4. Traditional machine learning models can effectively handle this NLP task with proper preprocessing

## Performance Evaluation
The models were evaluated using Macro F1 score to account for class imbalance, providing a balanced measure of performance across both classes.

## Best Model Configuration
After grid search hyperparameter tuning, the optimal model was a Random Forest Classifier with TF-IDF vectorization.

## Conclusion
This project demonstrates the effectiveness of a well-designed NLP pipeline for hate speech classification. TF-IDF vectorization combined with Random Forest classification proved most effective for this task. The custom transformer approach provides a clean, modular implementation that can be extended to other text classification problems.
