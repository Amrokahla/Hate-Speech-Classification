# Hate Speech Classification

## Overview
This project implements two approaches to classify tweets as containing hate speech or not:
1. Traditional Machine Learning Models
2. Recurrent Neural Networks (RNNs)

Both approaches demonstrate effective methods for detecting harmful content in social media text.

## Dataset
The dataset consists of labeled tweets with binary classification:
- `0`: No hate speech
- `1`: Contains hate speech

The dataset is imbalanced, with significantly fewer hate speech examples than non-hate speech ones.

## Approach 1: Traditional ML Models

### Preprocessing Pipeline
- Text cleaning (removing URLs, usernames, special characters)
- Converting emojis to text
- Case normalization
- Number replacement

### Feature Extraction
- Bag of Words (BOW)
- TF-IDF Vectorization

### Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Naive Bayes

### Results
The best performance was achieved using TF-IDF vectorization with a Random Forest Classifier.

## Approach 2: RNN Models

### Preprocessing
- Text cleaning (similar to ML approach)
- Tokenization
- Sequence padding

### Models
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM

### Word Embeddings
- Word2Vec
- GloVe
- FastText

### Results
Bidirectional LSTM with pre-trained GloVe embeddings provided the best performance among neural models.

## Evaluation
Both approaches were evaluated using Macro F1 score to account for class imbalance.

## Conclusion
- Traditional ML models with TF-IDF offer a simple yet effective solution
- RNN models can capture sequential patterns in text but require more computational resources
- Both approaches demonstrate viable methods for hate speech detection with different trade-offs in terms of complexity and performance
