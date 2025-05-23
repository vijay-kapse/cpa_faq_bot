import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
    return tokenizer, model

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("faq.csv")
    questions = df['Question Text'].fillna("").tolist()
    answers = df['Answer Text'].fillna("").tolist()
    corpus = [q + " " + a for q, a in zip(questions, answers)]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return df, corpus, vectorizer, X

# Load everything
tokenizer, model = load_model_and_tokenizer()
df, corpus, vectorizer, X = load_and_prepare_data()

# Streamlit UI
st.title("📘 SaaS Accounting FAQ Bot (Flan-T5)")
st.write("Ask me anything related to SaaS accounting!")

user_query = st.text_input("Your question:")

if user_query:
    # Find the most relevant context
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    best_match_idx = similarity.argmax()
    context = corpus[best_match_idx]

    # Format prompt
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=150)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown("**Bot:** " + response.strip())
