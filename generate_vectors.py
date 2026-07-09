import os
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.preprocessing import load_data

def main():
    print("🚀 Loading clean data...")
    df= load_data()

    if 'ISBN' not in df.columns:
        print("⚠️Warning: ISBN column not found! Check your preprocessing.py file.")
    else:
        print("✓ The ISBN column was successfully detected.")
        
    print("🧠 The Transformer model is being loaded and vectors are being calculated...  ")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    descriptions = df["Description"].fillna("").tolist()

    embeddings = model.encode(descriptions, show_progress_bar=True)

    print("🗄️FAISS index is being created and vectors are being written to disk...")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs('data', exist_ok=True)
    faiss.write_index(index, "data/books_index.faiss")
    with open("data/books_metadata.pkl", "wb") as f:
        pickle.dump(df, f)

    print("\n✨ SUCCESS! The FAISS index and metadata have been permanently saved to the 'data/' folder.")

if __name__ == "__main__":
    main()

    





