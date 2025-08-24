import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ------------------------------
# Load IMDb dataset
# ------------------------------
st.title("🎬 IMDb Sentiment Analysis App")

st.write("This app predicts whether a movie review is **Positive** or **Negative** using a trained LSTM model on the IMDb dataset.")

# Parameters
vocab_size = 10000  # Top 10k words
maxlen = 200        # Max review length

@st.cache_resource
def train_model():
    # Load data
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    # Build model
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model (only few epochs to keep it lightweight)
    model.fit(X_train, y_train, epochs=2, batch_size=128, validation_data=(X_test, y_test), verbose=1)

    return model, X_train, y_train

with st.spinner("Training model... (this may take 2-3 minutes on first run)"):
    model, X_train, y_train = train_model()

# ------------------------------
# Input Review
# ------------------------------
word_index = imdb.get_word_index()

# Map word index to word
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

def encode_review(text):
    tokens = text.lower().split()
    encoded = [1]  # start token
    for word in tokens:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # unknown word
    return pad_sequences([encoded], maxlen=maxlen)

st.subheader("🔎 Try it yourself")
user_input = st.text_area("Enter a movie review:", "This movie was amazing, I really loved it!")

if st.button("Predict Sentiment"):
    encoded_input = encode_review(user_input)
    prediction = model.predict(encoded_input)[0][0]
    sentiment = "😊 Positive" if prediction > 0.5 else "☹️ Negative"
    st.write(f"**Prediction:** {sentiment} (Confidence: {prediction:.2f})")
