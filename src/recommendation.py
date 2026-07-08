import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Features"])

    indices = pd.Series(df.index, index=df["Name"]).drop_duplicates()

    return tfidf_matrix, indices

def recommend(book_name, df, tfidf_matrix, indices, top_n=10):
    # Partial search
    matches = df[
        df["Name"].str.contains(book_name, case=False, na=False)
    ]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = similar_indices[1:top_n + 1]

    recommendations = df.iloc[similar_indices][
        ["Name", "Authors", "Rating"]
    ].copy()

    recommendations["Similarity"] = (
        similarity_scores[similar_indices] * 100
    ).round(2)

    return recommendations

    