import argparse
import random
import sys
from typing import Any, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an IMDb sentiment analysis model with NLP preprocessing + RNN."
    )
    parser.add_argument("--num-words", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--max-len", type=int, default=200, help="Sequence length after padding")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--rnn-units", type=int, default=64, help="LSTM units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model-out",
        default="imdb_sentiment_rnn.keras",
        help="Path for exported model",
    )
    return parser.parse_args()


def load_and_preprocess_imdb(
    tf, num_words: int, max_len: int
) -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_len, padding="post", truncating="post"
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_len, padding="post", truncating="post"
    )
    return x_train, y_train, x_test, y_test


def build_model(tf, vocab_size: int, max_len: int, embedding_dim: int, rnn_units: int, dropout: float):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def split_train_validation(
    x: Any, y: Any, validation_split: float
) -> Tuple[Any, Any, Any, Any]:
    split_index = int(len(x) * (1 - validation_split))
    if split_index <= 0 or split_index >= len(x):
        raise ValueError("--validation-split must keep at least one sample in train and validation sets")
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]


def main() -> int:
    args = parse_args()

    try:
        import tensorflow as tf
    except ImportError:
        print(
            "TensorFlow is not installed. Run `pip install -r requirements.txt` and retry.",
            file=sys.stderr,
        )
        return 1

    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x_train, y_train, x_test, y_test = load_and_preprocess_imdb(
        tf=tf, num_words=args.num_words, max_len=args.max_len
    )

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train, y_train, validation_split=args.validation_split
    )

    model = build_model(
        tf=tf,
        vocab_size=args.num_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        dropout=args.dropout,
    )

    model.summary()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    test_loss, test_accuracy, test_auc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    model.save(args.model_out)
    print(f"Saved model to: {args.model_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
