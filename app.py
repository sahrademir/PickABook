import streamlit as st
import pandas as pd
import urllib.parse
import faiss
import pickle
import numpy as np
from src.preprocessing import load_data
from src.recommendation import build_model, recommend
from src.recommendation_llm import get_llm_explanation
from src.user_recommendation import recommend_user_profile

def recommend_user_profile_direct(selected_books, data, faiss_index, top_n=6):
    import numpy as np
    
    # Precise 0-based positional indices from the current dataframe array
    matched_positions = [int(pos) for pos in np.where(data['Name'].isin(selected_books))[0]]
    
    ntotal = faiss_index.ntotal
    valid_positions = [pos for pos in matched_positions if pos < ntotal]
    
    if not valid_positions:
        return data.head(top_n)
        
    vectors = [faiss_index.reconstruct(pos) for pos in valid_positions]
    profile_vector = np.mean(vectors, axis=0).reshape(1, -1)
    
    distances, indices = faiss_index.search(profile_vector, top_n + len(selected_books))
    
    rec_indices = [idx for idx in indices[0] if idx not in valid_positions and idx < len(data)][:top_n]
    
    return data.iloc[rec_indices].copy()

st.set_page_config(
    page_title="PickABook | AI Recommendation", 
    page_icon="📖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .book-card {
        background-color: #1f293d;
        padding: 15px;
        border-radius: 12px;
        border-bottom: 4px solid #ff4b4b;
        margin-bottom: 20px;
        transition: transform 0.2s;
        text-align: center;
        min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .book-card:hover {
        transform: translateY(-5px);
        background-color: #26354e;
    }
    .cover-container{
        height: 220px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    }
    .book-cover{
        max-height: 100%;
        max-width: 100%;
        border-radius: 6px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);        
    }
    .book-title {
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;        
     }
    .book-author { color: #a3b3cc; font-size: 13px; font-style: italic; margin-bottom:10px; }
    .metric-container{
            background: #161b26;
            padding: 8px;
            border-radius: 8px;
            margin-top: auto;
            }
    .metric-text { font-size: 12px; font-weight: 600; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

def open_details(book_name, book_authors, book_rating, img_url, t_name, t_auth):
    st.session_state.selected_book = {
        "title": book_name,
        "author": book_authors,
        "cover": img_url,
        "rating": book_rating,
        "target_title": t_name,
        "target_author": t_auth
    }
    st.session_state.page = "details"

@st.cache_resource
def init_all_models():
    with open("data/books_metadata.pkl", "rb") as f:
        data = pickle.load(f)
    
    faiss_index = faiss.read_index("data/books_index.faiss")
    
    data['Display_Name'] = data['Name'] + " (by " + data['Authors'] + ")"
    
    tfidf_matrix, tfidf_indices = build_model(data)
    
    isbn_col = None
    for col in ['ISBN', 'isbn', 'ISBN13', 'Isbn']:
        if col in data.columns:
            isbn_col = col
            break
            
    return data, faiss_index, tfidf_matrix, tfidf_indices, isbn_col

data, faiss_index, tfidf_matrix, tfidf_indices, isbn_column = init_all_models()

def recommend_faiss(target_title, df, index, top_n=6):
    try:
        idx = df[df['Name'] == target_title].index[0]
        target_vector = index.reconstruct(int(idx)).reshape(1, -1)
        distances, indices = index.search(target_vector, top_n + 1)
        
        sim_indices = indices[0][1:]
        sim_distances = distances[0][1:]
        
        min_d, max_d = min(sim_distances), max(sim_distances)
        span = (max_d - min_d) if (max_d - min_d) > 0 else 1
        similarities = [0.95 - ((d - min_d) / span) * 0.4 for d in sim_distances]
        
        recommended_df = df.iloc[sim_indices].copy()
        recommended_df['Similarity'] = similarities
        return recommended_df
    except Exception as e:
        return None

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "profile_triggered" not in st.session_state:
    st.session_state.profile_triggered = False

if st.session_state.page == "home":

    st.title("📖 PickABook")
    st.markdown("### *AI-Powered Content-Based Book Recommendation Engine*")
    st.markdown("---")

    rec_mode = st.radio(
        "Choose Recommendation Mode:",
        ["Similar Book", "My Favorite Books"],
        horizontal=True
    )
    st.markdown("---")

    search_query = ""
    selected_display_names = []

    if rec_mode == "Similar Book":
        st.session_state.profile_triggered = False
        col_search, col_model = st.columns([2, 1])
        with col_search:
            search_query = st.text_input(
                "What is the last book you read and enjoyed?",
                placeholder="Type a book title or keyword...",
                key="book_search_input"
            )
        with col_model:
            model_choice = st.radio(
                "Select the Recommendation Algorithm:",
                ["TF-IDF (Word-Matching Based)", "Deep Learning (Semantic Search + FAISS)"],
                key="algo_single"
            )
            
    else:
        st.markdown("#### Build Your Reading Profile")
        st.markdown("Select at least 2-3 books you loved. We will blend their embeddings to find your perfect matches!")
        
        search_query = st.text_input("Type a book name to search in our library:", key="profile_search_input")
        filtered_books = []
        if search_query:
            filtered_books = data[data['Name'].str.contains(search_query, case=False, na=False)]['Name'].tolist()

        selected_display_names = st.multiselect(
            "Select your favorite books from search results:",
            options=filtered_books[:10],
            max_selections=5,
            key="profile_book_multiselect"
        )

        model_choice = "Deep Learning (Semantic Search + FAISS)"

    results = None
    target_book_name = ""
    target_book_author = ""

    if rec_mode == "Similar Book" and search_query != "":
        matched_books = data[data["Name"].str.lower().str.contains(search_query.lower(), na=False)]
        
        if not matched_books.empty:
            target_book_name = matched_books.sort_values(by="Rating", ascending=False).iloc[0]["Name"]
            target_book_author = matched_books.sort_values(by="Rating", ascending=False).iloc[0]["Authors"]
            
            if target_book_name != search_query:
                st.info(f"Best match for your search: **{target_book_name}**")
                
            st.markdown(f"#### **'{target_book_name}'** Best Recommendations Similar to Your Book:")
            
            if "TF-IDF" in model_choice:
                with st.spinner("Calculating TF-IDF..."):
                    raw_results = recommend(target_book_name, data, tfidf_matrix, tfidf_indices, top_n=6)
                    if raw_results is not None and not raw_results.empty:
                        results = data[data['Name'].isin(raw_results['Name'])].copy()
                        results = results.merge(raw_results[['Name', 'Similarity']], on='Name', how='left')
                        results['Similarity'] = 0.5 + (results['Similarity'] / results['Similarity'].max() * 0.45) if results['Similarity'].max() > 0 else results['Similarity']
                    else:
                        results = None
            else:
                with st.spinner("FAISS lightspeed search..."):
                    results = recommend_faiss(target_book_name, data, faiss_index, top_n=6)

    elif rec_mode == "My Favorite Books":
        if len(selected_display_names) > 0:
            if "profile_triggered" not in st.session_state:
                st.session_state.profile_triggered = False

            if st.button("Generate Profile Recommendations", use_container_width=True):
                st.session_state.profile_triggered = True
                if "profile_results" in st.session_state:
                    del st.session_state.profile_results

            if st.session_state.profile_triggered:
                selected_favorites = data[data['Name'].isin(selected_display_names)]['Name'].tolist()
                st.markdown("#### 🧬 Personalized Recommendations Based on Your Favorites:")
            
                if "profile_results" not in st.session_state:
                    with st.spinner("Blending embeddings and scanning library..."):
                        if "profile_results" not in st.session_state:
                            with st.spinner("Blending embeddings and scanning library..."):
                                st.session_state.profile_results = recommend_user_profile_direct(selected_favorites, data, faiss_index, top_n=6)

                results = st.session_state.profile_results
        else:
            st.session_state.profile_triggered = False
            if "profile_results" in st.session_state:
                del st.session_state.profile_results

    if results is not None and not results.empty:
        st.write("")
        col1, col2, col3 = st.columns(3)
        
        for idx, row in results.reset_index().iterrows():
            current_col = [col1, col2, col3][idx % 3]
            
            with current_col:
                if 'Similarity' in row:
                    similarity_pct = int(row['Similarity'] * 100)
                    similarity_text = f"🎯 {similarity_pct}% Match"
                else:
                    similarity_text = "🧬 Profile Match"

                placeholder_url = "https://via.placeholder.com/180x240/1f293d/ffffff?text=No+Cover"
                cover_url = placeholder_url
                
                if isbn_column is not None and isbn_column in row and pd.notna(row[isbn_column]):
                    raw_isbn = str(row[isbn_column]).strip()
                    if raw_isbn.endswith('.0'):
                        raw_isbn = raw_isbn[:-2]
                    
                    clean_isbn = "".join(c for c in raw_isbn if c.isalnum())
                    
                    if len(clean_isbn) >= 9:
                        encoded_placeholder = urllib.parse.quote_plus(placeholder_url)
                        cover_url = f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-M.jpg?default={encoded_placeholder}"
                
                card_html = f"""
                <div class="book-card">
                    <div class="cover-container">
                        <img class="book-cover" src="{cover_url}">
                    </div>
                    <div class="book-title">📘 {row['Name']}</div>
                    <div class="book-author">✍️ {row['Authors']}</div>
                    <div class="metric-container">
                        <div class="metric-text" style="margin-bottom: 5px;">⭐ {row['Rating']:.2f} / 5</div>
                        <div class="metric-text">🧬 Profile Fit: {similarity_text}%</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if 'Similarity' in row:
                    progress_value = min(max(float(row['Similarity']), 0.0), 1.0)
                else:
                    progress_value = 0.95

                st.progress(progress_value)
                
                st.button(
                    "✨ Why This Book?", 
                    key=f"btn_{idx}", 
                    use_container_width=True,
                    on_click=open_details,
                    args=(row['Name'], row['Authors'], row['Rating'], cover_url, target_book_name, target_book_author)
                )
                st.write("") 
    elif (rec_mode == "Similar Book" and search_query != "") or (rec_mode == "My Favorite Books" and len(selected_display_names) > 0 and st.session_state.profile_triggered):
        st.error("No books found matching your criteria.")

else:
    book = st.session_state.selected_book

    st.title("📖 Why do these books look similar to each other?")
    st.markdown("---")

    col_left, col_right = st.columns([1,2])

    with col_left:
        st.image(book['cover'], width=220)
        st.subheader(book['title'])
        st.markdown(f"**✍️ Author:** {book['author']}")
        st.markdown(f"**⭐ Global Rating:** {book['rating']:.2f} / 5")

        st.write("")
        if st.button("⬅️ Back to Recommendations", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_book = None
            st.rerun()

    with col_right:
        st.info(f" Connection Path: This book was recommended because you read and liked **'{book['target_title']}'** by *{book['target_author']}*.")

        st.markdown("### Llama-3 AI Semantic Explanation")
        with st.spinner("Analyzing thematic layers and writing styles via Llama-3..."):
            explanation = get_llm_explanation(
                target_title=book['target_title'],
                rec_title=book['title'],
                target_author=book['target_author'],
                rec_author=book['author']
            )
            
        st.success(explanation)
        
        st.markdown(f"---")
        st.caption("🤖 *Note: This explanation is fully generated in real-time by Meta-Llama-3-8B-Instruct model deployed on Hugging Face Inference API infrastructure.*")