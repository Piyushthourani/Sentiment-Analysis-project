# Sentiment-Analysis-project

This project trains an RNN-based sentiment classifier on the IMDb movie review
dataset so you can practice core NLP preprocessing and sequence modeling.

## What is included

- IMDb dataset loading (`tensorflow.keras.datasets.imdb`)
- NLP preprocessing:
  - vocabulary size limiting
  - sequence padding/truncation
- RNN models:
  - Embedding layer
  - LSTM, GRU, and Bidirectional LSTM
  - Dense output for binary sentiment classification
- Training, validation, evaluation, and model export
- **Streamlit Web App** for interactive sentiment analysis

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features
- 🎬 Interactive IMDB sentiment analyzer
- 📝 Enter custom movie reviews or select examples
- 📊 Get real-time sentiment predictions with confidence scores
- 📋 View preprocessed/cleaned text
- ✅ Model Info showing architecture and test accuracy (~87%)

## Train the model

```bash
python train_imdb_rnn.py --epochs 3 --batch-size 64
```

## Useful options

```bash
python train_imdb_rnn.py --help
```

Example:

```bash
python train_imdb_rnn.py \
  --num-words 10000 \
  --max-len 200 \
  --embedding-dim 128 \
  --rnn-units 64 \
  --epochs 3 \
  --model-out imdb_sentiment_rnn.keras
```