## 📖 PickABook
## AI-Powered Hybrid Book Recommendation System

PickABook is an end-to-end AI-powered hybrid recommendation system that suggests books based on semantic similarity and personalized reading preferences. The application combines traditional NLP, deep learning, vector search, Large Language Models, and a RESTful backend to deliver fast, explainable, and personalized recommendations through an interactive web interface.

## ✨ Features:

📚 Content-Based Recommendation (TF-IDF + Cosine Similarity)
🧠 Deep Learning Semantic Search (Sentence Transformers)
⚡ FAISS Vector Database for high-speed similarity search
👤 Personalized User-to-Book Recommendation
🤖 AI-generated recommendation explanations (Hugging Face LLM)
📖 Automatic Book Cover Retrieval
🔎 Intelligent Book Search
🌐 REST API built with FastAPI
🐳 Dockerized application
🎨 Interactive Streamlit interface
🚀 End-to-End AI recommendation pipeline

## 🏆 Key Achievements

- Built an end-to-end AI-powered hybrid book recommendation system from scratch.
- Combined TF-IDF, Sentence Transformers, FAISS, and LLMs into a single recommendation pipeline.
- Developed personalized user-to-book recommendations using semantic embeddings.
- Designed and exposed recommendation services through a FastAPI REST API.
- Containerized the application using Docker for portable deployment.
- Built an interactive Streamlit interface with AI-generated recommendation explanations.

## 🏗️ System Architecture:

                User
                  │
                  ▼
            Streamlit UI
                  │
             HTTP Requests
                  │
                  ▼
             FastAPI Backend
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     TF-IDF    FAISS     Hugging Face
        │         │            │
        └─────────┴────────────┘
                  │
          Recommendation Engine
                  │
                  ▼
        Goodreads Dataset


## 🛠 Tech Stack:

Category	Technologies
Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn
Deep Learning	Sentence Transformers
Semantic Search	FAISS
NLP	TF-IDF
LLM	Hugging Face Inference API
Backend	FastAPI
Frontend	Streamlit
Containerization	Docker
Version Control	Git & GitHub

## 📂 Project Structure:

BookRec_Project/
│
├── app.py                     
├── api/
│   ├── main.py
│   ├── routes.py 
│   ├── schemas.py 
│
├── src/
│   ├── preprocessing.py
│   ├── recommendation.py
│   ├── recommendation_dl.py
│   ├── user_recommendation.py
│   ├── recommendation_llm.py
│   └── llm.py
│
├── data/
│
├── notebooks/
│
│── test_api.py 
├── requirements.txt
├── Dockerfile
├── start.sh
├── .env
├── README.md
└── .gitignore



## 🚀 Installation:

Clone repository
git clone https://github.com/sahrademir/PickABook.git

cd PickABook
Install dependencies
pip install -r requirements.txt
Configure environment variables

Create a .env file:

HF_TOKEN=your_huggingface_token
Download model files

Due to GitHub file size limitations, the FAISS index and metadata files are hosted separately.

Download:

books_index.faiss
books_metadata.pkl

- [Download books_index.faiss](https://drive.google.com/file/d/1rDPGhtlzkNkUA3kTCIbUInRgfV-OrMTQ/view?usp=drive_link)
- [Download books_metadata.pkl](https://drive.google.com/file/d/1Lnr21ciGeK33zTS04ci5Mk6QKRi4Zz82/view?usp=drive_link)

Place them inside:

data/

before running the application.

Run with Streamlit
streamlit run app.py
Run API
uvicorn api.main:app --reload

API documentation:

http://127.0.0.1:8000/docs

Run with Docker
docker build -t pickabook .
docker run -p 8501:8501 pickabook

## 🎬 Application Demo

<img width="400" height="214" alt="Ekran Kaydı 2026-07-17 101319" src="https://github.com/user-attachments/assets/10085915-52e2-4232-afbb-17a96c9e9913" />


## 📊 Dataset:

Dataset used:

Goodreads Books Dataset (Kaggle)




