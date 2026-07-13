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

    # Load Dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")

    # Keep only required columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels to numbers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    Y = df["label"].values

    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    # Save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Build Model
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

    # Train Model
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Save Model
    model.save(MODEL)

    # Evaluate
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
# (deliberately avoiding the usual blue/purple gradient look)

st.markdown(
    """
    <style>
    :root {
        --bg-color:       #FBF6EF;   /* warm ivory */
        --card-color:     #FFFFFF;
        --teal:           #1F5C57;   /* deep teal - primary */
        --teal-dark:      #143E3B;
        --terracotta:     #C9663B;   /* burnt terracotta - accent */
        --terracotta-dark:#A64F2A;
        --gold:           #D9A441;   /* muted gold - highlight */
        --text-dark:      #2E2A26;
        --text-muted:     #6B645C;
        --border-soft:    #E6DFD3;
    }

    /* Overall app background */
    .stApp {
        background-color: var(--bg-color);
    }

    /* Hide default Streamlit chrome for a cleaner board look */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Main title block */
    .app-title-wrap {
        text-align: center;
        padding: 1.6rem 1rem 1rem 1rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(20, 62, 59, 0.25);
    }
    .app-title-wrap h1 {
        color: #FBF6EF !important;
        font-size: 2.1rem;
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

    /* Card container for the input section */
    .board-card {
        background-color: var(--card-color);
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        padding: 1.8rem 1.8rem 1.4rem 1.8rem;
        box-shadow: 0 4px 14px rgba(46, 42, 38, 0.06);
        margin-bottom: 1.4rem;
    }

    .board-card h3 {
        color: var(--teal-dark);
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        background-color: #FDFBF7 !important;
        border: 2px solid var(--border-soft) !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        font-size: 1rem !important;
        padding: 0.8rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 2px rgba(201, 102, 59, 0.15) !important;
    }
    .stTextArea label {
        color: var(--teal-dark) !important;
        font-weight: 600 !important;
        text-align: center;
    }

    /* Button styling */
    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        background: linear-gradient(135deg, var(--terracotta) 0%, var(--terracotta-dark) 100%);
        color: #FFF8F0;
        border: none;
        border-radius: 30px;
        padding: 0.65rem 2.6rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 12px rgba(166, 79, 42, 0.35);
        transition: all 0.2s ease-in-out;
        margin-top: 0.4rem;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(166, 79, 42, 0.45);
        background: linear-gradient(135deg, var(--terracotta-dark) 0%, var(--terracotta) 100%);
        color: #FFFFFF;
    }
    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* Result card - Spam */
    .result-spam {
        background: linear-gradient(135deg, #F6D9CC 0%, #F0C2AC 100%);
        border: 2px solid var(--terracotta);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 14px rgba(166, 79, 42, 0.18);
    }
    .result-spam h2 {
        color: var(--terracotta-dark);
        margin-bottom: 0.3rem;
        font-size: 1.6rem;
    }

    /* Result card - Ham */
    .result-ham {
        background: linear-gradient(135deg, #DCEAE3 0%, #C6DED3 100%);
        border: 2px solid var(--teal);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 14px rgba(31, 92, 87, 0.18);
    }
    .result-ham h2 {
        color: var(--teal-dark);
        margin-bottom: 0.3rem;
        font-size: 1.6rem;
    }

    .result-confidence {
        color: var(--text-muted);
        font-size: 1rem;
        font-weight: 600;
    }

    .result-confidence span {
        color: var(--gold);
        font-weight: 800;
    }

    /* Warning box override */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        text-align: center;
    }

    /* Footer */
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

st.markdown('<div class="board-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Check a Message", unsafe_allow_html=True)

message = st.text_area(
    "Enter the SMS message you'd like to analyze",
    height=130,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim now...",
)

predict_clicked = st.button("Analyze Message")

st.markdown("</div>", unsafe_allow_html=True)

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

# ---------------- FOOTER ---------------- #

st.markdown(
    """
    <div class="app-footer">
        Built with Streamlit &amp; TensorFlow · Simple RNN Architecture
    </div>
    """,
    unsafe_allow_html=True,
)
