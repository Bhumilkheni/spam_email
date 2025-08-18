import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("model/vectorizer.pkl", "rb"))
model = pickle.load(open("model/model.pkl", "rb"))

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(-45deg, #00c6ff, #0072ff, #ff4b4b, #ff1e1e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .main-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        max-width: 600px;
        margin: auto;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    h1 {
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .stTextArea textarea {
        border: 2px solid rgba(255,255,255,0.5);
        border-radius: 10px;
        background-color: rgba(255,255,255,0.15);
        color: black;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #00ff7f;
        box-shadow: 0 0 10px #00ff7f;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 0;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff1e1e;
        transform: scale(1.05);
        box-shadow: 0px 4px 15px rgba(255,0,0,0.6);
    }

    .result-good {
        color: #00ff7f;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0px 0px 10px #00ff7f;
    }
    .result-bad {
        color: #ff4b4b;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0px 0px 10px #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>📧 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message:")

if st.button("🔍 Predict"):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div class='result-bad'>🚨 Spam Detected!</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-good'>✅ Not Spam</div>", unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
