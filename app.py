import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector for Reddit Titles")
user_input = st.text_input("Enter a Reddit post title", key="input_text")

if st.button("Predict", key="predict_button"):
    if user_input.strip() == "":
        st.warning("Please enter a title.")
    else:
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        prob = model.predict_proba(vectorized)[0][1]
        label = "Fake News" if prob > 0.5 else "Real News"

        st.subheader("Result")
        st.write(f"Prediction: 🎯 **{label}**")
        st.write(f"Fake News Probability: `{prob:.2f}`")

