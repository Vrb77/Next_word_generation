import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from pathlib import Path
image = Image.open('Templates/word_generation.jpg')

st.set_page_config(
    page_title="Next words generation on amazon food reviews",
    page_icon=Image.open('Templates/Word_generation_icon.png'),
)
st.set_page_config(page_title="Next words generation on amazon food reviews")
st.title("Next words generation on amazon food reviews")
st.image(image)
st.subheader("By Vaishnavi Badade")

# Load model
model = load_model("TextGenerationModel.keras")

# ✅ Load the saved tokenizer (not a fresh one!)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Reverse mapping: index -> word
index_word = {i: word for word, i in tokenizer.word_index.items()}

# ✅ Must match training value
max_sequence_length = 6  # Change this to match your training

def sample_with_temperature(preds, temperature=0.5, top_k=10):
    preds = np.asarray(preds).astype("float64")
    top_indices = np.argsort(preds)[-top_k:]
    top_probs = preds[top_indices]
    top_probs = np.log(top_probs + 1e-10) / temperature
    exp_probs = np.exp(top_probs)
    top_probs = exp_probs / np.sum(exp_probs)
    return np.random.choice(top_indices, p=top_probs)

def generate_text(seed_text, next_words=20):
    output_text = seed_text
    generated_words = []
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding='pre'
        )
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted_probs, temperature=0.5, top_k=10)
        next_word = index_word.get(predicted_index, "")
        if next_word in generated_words[-3:]:
            continue
        generated_words.append(next_word)
        output_text += " " + next_word
    return output_text

text_input = st.text_input("Enter some text to generate next words")
submit = st.button("Generate next words")

if submit:
    if not text_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Generating..."):
            generated_text = generate_text(text_input, next_words=15)
        st.success("Generated Text:",generated_text)
        
