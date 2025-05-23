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

# Load resources
tokenizer, model = load_model_and_tokenizer()
df, corpus, vectorizer, X = load_and_prepare_data()

# Streamlit UI
st.title("📘 SaaS Accounting FAQ Bot (Flan-T5)")
st.write("Ask a question about SaaS accounting (ASC 606, revenue recognition, contracts, etc.)")


st.subheader("💡 Suggested Questions")
col1, col2 = st.columns(2)

with col1:
    if st.button("💬 How do I record revenue from annual subscriptions collected upfront?"):
        st.session_state.user_query = "How do I record revenue from annual subscriptions collected upfront?"

with col2:
    if st.button("💬 Should I recognize deferred revenue for prepaid SaaS contracts?"):
        st.session_state.user_query = "Should I recognize deferred revenue for prepaid SaaS contracts?"

# Initialize session state if not already set
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

user_query = st.text_input("Your question:", value=st.session_state.user_query)


user_query = st.text_input("Your question:")

if user_query:
    if len(user_query.strip()) < 5:
        st.warning("Please enter a more specific question.")
    else:
        # Find the most relevant context
        query_vec = vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, X).flatten()
        best_match_idx = similarity.argmax()
        context = corpus[best_match_idx]

        # Improved prompt
        prompt = f"""
You are a helpful AI assistant specialized in SaaS accounting (e.g., ASC 606, revenue recognition, subscription billing).
Use the provided context to give a concise and accurate answer to the user's question.

Context:
{context}

Question:
{user_query}

Answer:"""

        # Generate answer
        inputs = tokenizer(prompt.strip(), return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=150)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("**Bot:** " + response.strip())




st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.markdown("<center>Made with ❤️ by Shota and Vijay</center>", unsafe_allow_html=True)
