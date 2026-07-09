import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def build_dl_model(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Deep Learning Vectors (Embeddings) are being calculated... This process may take some time.")
    dl_embeddings = model.encode(df["Description"].tolist(), show_progress_bar=True)

    indices = pd.Series(df.index, index=df["Name"]).drop_duplicates()

    return dl_embeddings, indices

def recommend_dl(book_name, df, dl_embeddings, indices, top_n=6):
    if book_name not in indices:
        return None
    
    idx= indices[book_name]

    target_embedding = dl_embeddings[idx].reshape(1,-1)
    sim_scores = cosine_similarity(target_embedding, dl_embeddings).flatten()
    sim_df = pd.DataFrame({'Similarity': sim_scores, 'Name': df['Name'], 'Authors':df['Authors'], 'Rating': df['Rating']})
    sim_df= sim_df[sim_df['Name'] != book_name]
    return sim_df.sort_values(by='Similarity', ascending=False).head(top_n)



