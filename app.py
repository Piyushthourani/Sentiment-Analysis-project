import re
import string
import pickle

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------
# Load Model and Tokenizer
# ----------------------------

model = load_model("models/gru_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ----------------------------
# NLP Setup
# ----------------------------

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

MAX_LEN = 200


# ----------------------------
# Text Cleaning
# ----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ----------------------------
# Prediction Function
# ----------------------------

def predict_sentiment(text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("IMDb Sentiment Analysis")

st.write(
    "Enter a movie review and predict its sentiment."
)

review = st.text_area(
    "Movie Review",
    height=200
)

if st.button("Analyze Sentiment"):

    if review.strip():

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        st.write(
            f"**Sentiment:** {sentiment}"
        )

        st.write(
            f"**Confidence:** {confidence:.2%}"
        )

    else:
        st.warning(
            "Please enter a review."
        )