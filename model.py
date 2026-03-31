import streamlit as st
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# set the tab title
st.set_page_config("Next words generation on amazon food reviews")

# Set the page title
st.title("Next words generation")

# Set header
st.subheader("By Vaishnavi Badade")


model = load_model("TextGenerationModel.keras")


text_input=st.text_input("Review")

tokenizer = Tokenizer()


# Reverse mapping : index -> word
index_word = {i : word for word,i in tokenizer.word_index.items()}
max_sequence_length =6
def sample_with_temperature(preds, temperature=0.8, top_k=5):
    preds = np.asarray(preds).astype("float64")
    # [0.87,0.09,0.56,0.44,0.37,.............,0.89,0.32,...]

    # Select top k probabilities
    top_indices = np.argsort(preds)[-top_k:]
    # argsort : [0.09,0.32,0.37,0.44,0.56,0.87,0.89,........]
    # index of top k proabilities : [index]
    top_probs = preds[top_indices]
    #top k probs

    # Apply temperature scaling
    top_probs = np.log(top_probs + 1e-10) / temperature
    exp_probs = np.exp(top_probs)
    top_probs = exp_probs / np.sum(exp_probs)

    return np.random.choice(top_indices, p=top_probs)

# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Generate next words")

def generate_text(seed_text, next_words=20):
    output_text = seed_text
    generated_words = []

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding='pre'
        ) # [0,0,0,189,45]

        predicted_probs = model.predict(token_list, verbose=0)[0] # [[0.89,0.07,.....,]] = [0.89,0.07,.....,]

        predicted_index = sample_with_temperature(
            predicted_probs,
            temperature=0.8,
            top_k=5
        )

        next_word = index_word.get(predicted_index, "")

        # avoid immediate repetition
        if next_word in generated_words[-3:]:
            continue

        generated_words.append(next_word)
        output_text += " " + next_word

    return output_text
if submit:
    generated_text=generate_text(text_input, next_words=15)
   
    st.subheader(generated_text)
