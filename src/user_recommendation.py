import numpy as np
import pandas as pd
import faiss

def recommend_user_profile(favorite_titles, df, index, top_n=6):
    matched_indices = []
    for title in favorite_titles:
        matched = df[df['Name'] == title]
        if not matched.empty:
            pos_idx = df.index.get_loc(matched.index[0])
            matched_indices.append(pos_idx)

    if not matched_indices:
        return None
    
    vectors = []
    for idx in matched_indices:
        # FAISS indeksinden vektörü çekiyoruz
        vec = index.reconstruct(int(idx))
        vectors.append(vec)

    user_profile_vector = np.mean(vectors, axis=0).reshape(1,-1)
    faiss.normalize_L2(user_profile_vector)

    search_k = top_n + len(favorite_titles)
    distances, indices = index.search(user_profile_vector, search_k)

    sim_indices = indices[0]
    sim_distances = distances[0]

    min_d, max_d = min(sim_distances), max(sim_distances)
    span = (max_d - min_d) if (max_d - min_d) > 0 else 1

    recommended_records = []
    for idx, dist in zip(sim_indices, sim_distances):
        if idx >= len(df):
            continue

        book_title = df.iloc[idx]['Name']

        if any(fav.lower() in book_title.lower() for fav in favorite_titles):
            continue

        similarity = float(dist)
        row_copy = df.iloc[idx].copy()
        row_copy['Similarity'] = similarity
        recommended_records.append(row_copy)

        if len(recommended_records) == top_n:
            break

    if not recommended_records:
        return None
    
    return pd.DataFrame(recommended_records)