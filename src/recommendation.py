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
        df["Name"].str.contains(book_name, case=False, na=False, regex=False)
    ]

    if matches.empty:
        return None

    df_idx = matches.index[0]

    if isinstance(indices, dict):
        matrix_row_idx = indices.get(df_idx, None)
    else:
        try:
            matrix_row_idx = df.index.get_loc(df_idx)
        except:
            matrix_row_idx = list(df.index).index(df_idx)

    if matrix_row_idx is None:
        matrix_row_idx = df_idx

    similarity_scores = cosine_similarity(
        tfidf_matrix[matrix_row_idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [i for i in similar_indices if i != matrix_row_idx][:top_n]
    

    recommendations = df.iloc[similar_indices][
        ["Name", "Authors", "Rating"]
    ].copy()

    recommendations["Similarity"] = (
        similarity_scores[similar_indices] * 100
    ).round(2)

    return recommendations

