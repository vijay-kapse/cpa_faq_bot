# 📘 SaaS Accounting FAQ Bot (Flan-T5-powered)

[![Streamlit App](https://img.shields.io/badge/Live%20App-cpabot.streamlit.app-blue?logo=streamlit)](https://cpabot.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Streamlit-powered chatbot trained on SaaS accounting FAQs, built using **Google's Flan-T5** model and **TF-IDF** context retrieval. Ask questions related to **ASC 606**, **deferred revenue**, **Stripe accounting**, and more.

> 💡 Ideal for early-stage SaaS founders, bookkeepers, and accounting teams.

---

## ✨ Features

- ✅ Natural language Q&A over curated SaaS accounting knowledge
- ✅ Powered by [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)
- ✅ Uses TF-IDF for FAQ context retrieval
- ✅ Streamlit UI with suggested questions
- ✅ Deployable on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🖥️ Try It Out

👉 [cpabot.streamlit.app](https://cpabot.streamlit.app/)

Ask questions like:
- _How do I record Stripe fees?_
- _When should I recognize deferred revenue for SaaS contracts?_
- _What is the correct entry for annual billing?_

---

## 🛠 How It Works

1. **TF-IDF Vectorizer** retrieves the closest matching FAQ context.
2. **Flan-T5 model** uses that context to generate a natural language answer.
3. Suggested prompts help users get started quickly.

---

