# Sentiment-Analysis-project

This project trains an RNN-based sentiment classifier on the IMDb movie review
dataset so you can practice core NLP preprocessing and sequence modeling.

## What is included

- IMDb dataset loading (`tensorflow.keras.datasets.imdb`)
- NLP preprocessing:
  - vocabulary size limiting
  - sequence padding/truncation
- RNN model:
  - Embedding layer
  - Bidirectional LSTM
  - Dense output for binary sentiment classification
- Training, validation, evaluation, and model export

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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