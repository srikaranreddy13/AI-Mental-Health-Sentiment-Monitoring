import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mental Health Monitor",
    page_icon="🧠",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.result-box {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model_files():

    model = tf.keras.models.load_model(
        "mental_health_rnn_model.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder


try:

    model, tokenizer, encoder = load_model_files()

except Exception as e:

    st.error(f"Error Loading Files: {e}")

    st.stop()


# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# =====================================================
# PREDICTION FUNCTION
# =====================================================

MAX_LENGTH = 50

def predict_sentiment(text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    prediction = model.predict(padded)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    sentiment = encoder.inverse_transform(
        [predicted_class]
    )[0]

    return sentiment, confidence


# =====================================================
# TITLE
# =====================================================

st.markdown(
    "<div class='title'>🧠 AI Mental Health Sentiment Monitoring</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning based Emotion Detection using Simple RNN</div>",
    unsafe_allow_html=True
)


# =====================================================
# TEXT INPUT
# =====================================================

user_input = st.text_area(
    "Enter Your Message",
    height=200,
    placeholder="Example: I feel stressed and emotionally tired..."
)


# =====================================================
# BUTTON
# =====================================================

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        sentiment, confidence = predict_sentiment(
            user_input
        )

        confidence_percent = round(
            confidence * 100,
            2
        )

        st.markdown(f"""
        <div class='result-box'>

        <h2>Predicted Emotion</h2>

        <h1 style='color:#38bdf8;'>
        {sentiment}
        </h1>

        <h3>Confidence Score</h3>

        <h2>
        {confidence_percent}%
        </h2>

        </div>
        """, unsafe_allow_html=True)

        if sentiment.lower() in [
            "depression",
            "suicidal",
            "anxiety",
            "stress"
        ]:

            st.error(
                "⚠ Negative emotional state detected."
            )

        else:

            st.success(
                "✅ Positive emotional state detected."
            )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 About")

st.sidebar.info("""
This project uses:
- NLP
- TensorFlow
- Simple RNN
- Streamlit

to detect emotional sentiment from text.
""")

st.sidebar.title("🧪 Example Inputs")

examples = [
    "I feel lonely and depressed",
    "I am mentally exhausted",
    "Today is a beautiful day",
    "I feel calm and happy"
]

for ex in examples:
    st.sidebar.write("•", ex)