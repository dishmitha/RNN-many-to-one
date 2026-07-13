# RNN PRACTICAL (Many-to-One)
# SMS Spam Detection using Simple RNN
# Dataset: spam.csv

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIGURATION ---------------- #

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50

# ---------------- CLEAN TEXT ---------------- #

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- TRAIN MODEL ---------------- #

def train_model():

    print("Training model...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    df["text"] = df["text"].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    Y = df["label"].values

    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(SimpleRNN(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    model.save(MODEL)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print("\nAccuracy:", accuracy)

    predictions = (model.predict(x_test) > 0.5).astype(int).flatten()

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nAccuracy Score")
    print(accuracy_score(y_test, predictions))

# ---------------- PREDICT SMS ---------------- #

def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    probability = model.predict(sequence, verbose=0)[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", probability

# Train only once
if not os.path.exists(MODEL):
    train_model()

# ================================================== #
#                  STREAMLIT UI                       #
# ================================================== #

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="✉️",
    layout="centered",
)

# ---------------- CUSTOM STYLING ---------------- #
# Palette: warm ivory background, deep teal + burnt terracotta accents

st.markdown(
    """
    <style>
    :root {
        --bg-color:       #FBF6EF;
        --card-color:     #FFFFFF;
        --teal:           #1F5C57;
        --teal-dark:      #143E3B;
        --terracotta:     #C9663B;
        --terracotta-dark:#A64F2A;
        --gold:           #D9A441;
        --text-dark:      #2E2A26;
        --text-muted:     #6B645C;
        --border-soft:    #E6DFD3;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.85); }
        70%  { opacity: 1; transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes borderGlow {
        0%   { box-shadow: 0 0 6px rgba(201, 102, 59, 0.25), 0 4px 14px rgba(46, 42, 38, 0.06); border-color: var(--terracotta); }
        50%  { box-shadow: 0 0 18px rgba(31, 92, 87, 0.35), 0 4px 14px rgba(46, 42, 38, 0.06); border-color: var(--teal); }
        100% { box-shadow: 0 0 6px rgba(201, 102, 59, 0.25), 0 4px 14px rgba(46, 42, 38, 0.06); border-color: var(--terracotta); }
    }
    @keyframes underlineGrow {
        from { width: 0%; }
        to   { width: 70%; }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-4px); }
        40% { transform: translateX(4px); }
        60% { transform: translateX(-3px); }
        80% { transform: translateX(3px); }
    }

    .stApp {
        background-color: var(--bg-color) !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .app-title-wrap {
        text-align: center;
        padding: 1.6rem 1rem 1rem 1rem;
        margin-bottom: 1.4rem;
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(20, 62, 59, 0.25);
        animation: fadeInDown 0.7s ease-out;
    }
    .app-title-wrap h1 {
        color: #FBF6EF !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: 0.3px;
    }
    .app-title-wrap p {
        color: #E3D9C6 !important;
        font-size: 1rem;
        margin: 0;
        font-style: italic;
    }

    /* ---- Card built with a REAL Streamlit container (st.container(key=...)) ---- */
    /* Targets the wrapper Streamlit generates for a keyed container, so styling   */
    /* actually wraps the widgets inside it instead of floating as an empty div.   */
    div[class*="st-key-board_card"] {
        background-color: var(--card-color) !important;
        border: 3px solid var(--terracotta);
        border-radius: 20px;
        padding: 1.8rem 1.8rem 1.4rem 1.8rem;
        margin-bottom: 1.4rem;
        animation: fadeInUp 0.6s ease-out, borderGlow 3.5s ease-in-out infinite;
    }

    .card-heading {
        text-align: center;
        margin-bottom: 1.1rem;
    }
    .card-heading h3 {
        display: inline-block;
        color: var(--teal-dark) !important;
        font-weight: 900 !important;
        font-size: 1.9rem !important;
        letter-spacing: 0.4px;
        margin: 0 !important;
        text-shadow: 0 1px 0 rgba(255,255,255,0.6);
    }
    .card-heading .underline {
        display: block;
        height: 4px;
        width: 70%;
        margin: 0.4rem auto 0 auto;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--terracotta), var(--gold), var(--teal));
        animation: underlineGrow 1s ease-out;
    }

    .stTextArea textarea {
        background-color: #FDFBF7 !important;
        border: 2px solid var(--border-soft) !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        font-size: 1.05rem !important;
        padding: 0.8rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 3px rgba(201, 102, 59, 0.18) !important;
    }
    .stTextArea label p {
        color: var(--teal-dark) !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        text-align: center;
    }

    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        background: linear-gradient(135deg, var(--terracotta) 0%, var(--terracotta-dark) 100%);
        color: #FFF8F0;
        border: none;
        border-radius: 30px;
        padding: 0.7rem 2.8rem;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 12px rgba(166, 79, 42, 0.35);
        transition: all 0.2s ease-in-out;
        margin-top: 0.6rem;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 10px 22px rgba(166, 79, 42, 0.45);
        background: linear-gradient(135deg, var(--terracotta-dark) 0%, var(--terracotta) 100%);
        color: #FFFFFF;
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }

    .result-spam {
        background: linear-gradient(135deg, #F6D9CC 0%, #F0C2AC 100%);
        border: 3px solid var(--terracotta);
        border-radius: 18px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 18px rgba(166, 79, 42, 0.25);
        animation: popIn 0.5s ease-out, shake 0.6s ease-in-out 0.5s;
    }
    .result-spam h2 {
        color: var(--terracotta-dark);
        margin-bottom: 0.3rem;
        font-size: 1.8rem;
        font-weight: 900;
    }

    .result-ham {
        background: linear-gradient(135deg, #DCEAE3 0%, #C6DED3 100%);
        border: 3px solid var(--teal);
        border-radius: 18px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 18px rgba(31, 92, 87, 0.25);
        animation: popIn 0.5s ease-out;
    }
    .result-ham h2 {
        color: var(--teal-dark);
        margin-bottom: 0.3rem;
        font-size: 1.8rem;
        font-weight: 900;
    }

    .result-confidence {
        color: var(--text-muted);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .result-confidence span {
        color: var(--gold);
        font-weight: 900;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        text-align: center;
        animation: shake 0.5s ease-in-out;
    }

    .app-footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-soft);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ---------------- #

st.markdown(
    """
    <div class="app-title-wrap">
        <h1>✉️ SMS Spam Detector</h1>
        <p>Powered by a Simple RNN &nbsp;•&nbsp; Many-to-One Sequence Classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- INPUT CARD ---------------- #
# Using a real Streamlit container(key=...) instead of a raw unclosed <div> —
# this is what actually makes the card render correctly and removes the
# empty ghost box that used to show up before "Check a Message".

with st.container(key="board_card"):

    st.markdown(
        """
        <div class="card-heading">
            <h3>🔍 Check a Message</h3>
            <span class="underline"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    message = st.text_area(
        "Enter the SMS message you'd like to analyze",
        height=130,
        placeholder="e.g. Congratulations! You've won a free prize, click here to claim now...",
    )

    predict_clicked = st.button("Analyze Message")

# ---------------- RESULT ---------------- #

if predict_clicked:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message before analyzing.")
    else:
        prediction, probability = predict_sms(message)
        confidence = round(probability * 100, 2) if prediction == "Spam" else round((1 - probability) * 100, 2)

        if prediction == "Spam":
            st.markdown(
                f"""
                <div class="result-spam">
                    <h2>🚫 Spam Detected</h2>
                    <p class="result-confidence">Confidence: <span>{confidence}%</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-ham">
                    <h2>✅ Looks Safe (Ham)</h2>
                    <p class="result-confidence">Confidence: <span>{confidence}%</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

