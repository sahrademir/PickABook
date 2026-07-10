import streamlit as st
import pandas as pd
import urllib.parse
import faiss
import pickle
from src.preprocessing import load_data
from src.recommendation import build_model, recommend
from src.recommendation_llm import get_llm_explanation

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

# --- GLOBAL ON_CLICK FONKSİYONU (Döngünün dışına, en güvenli yere aldık!) ---
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

# --- 1. SCREEN: HOME ---
if st.session_state.page == "home":

    st.title("📖 PickABook")
    st.markdown("### *AI-Powered Content-Based Book Recommendation Engine*")
    st.markdown("---")

    col_search, col_model = st.columns([2,1])

    with col_search:
        search_query = st.text_input(
            "What is the last book you read and enjoyed?",
            placeholder="Type a book title or keyword...",
            key="book_search_input"
        )

    with col_model:
        model_choice = st.radio(
            "Select the Recommendation Algorithm:",
            ["TF-IDF (Word-Matching Based)", "Deep Learning (Semantic Search + FAISS)"]
        )

    if search_query != "":
        st.write("") 
        
        matched_books = data[data["Name"].str.lower().str.contains(search_query.lower(), na=False)]
        
        if not matched_books.empty:
            target_book_name = matched_books.sort_values(by="Rating", ascending=False).iloc[0]["Name"]
            target_book_author = matched_books.sort_values(by="Rating", ascending=False).iloc[0]["Authors"]
            
            if target_book_name != search_query:
                st.info(f"🔍 Best match for your search: **{target_book_name}**")
                
            st.markdown(f"#### 🎯 **'{target_book_name}'** Best Recommendations Similar to Your Book:")
            
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
                
            if results is not None and not results.empty:
                st.write("")
                col1, col2, col3 = st.columns(3)
                
                for idx, row in results.reset_index().iterrows():
                    current_col = [col1, col2, col3][idx % 3]
                    
                    with current_col:
                        similarity_pct = int(row['Similarity'] * 100)
                        placeholder_url = "https://via.placeholder.com/180x240/1f293d/ffffff?text=No+Cover"
                        
                        if isbn_column in row and pd.notna(row[isbn_column]) and str(row[isbn_column]).strip() != "":
                            cover_url = f"https://covers.openlibrary.org/b/isbn/{str(row[isbn_column]).strip()}-M.jpg?default=false"
                        else:
                            cover_url = placeholder_url
                        
                        card_html = f"""
                        <div class="book-card">
                            <div class="cover-container">
                                <img class="book-cover" src="{cover_url}" onerror="this.onerror=null;this.src='{placeholder_url}';">
                            </div>
                            <div class="book-title">{row['Name']}</div>
                            <div class="book-author">✍️ {row['Authors']}</div>
                            <div class="metric-container">
                                <div class="metric-text" style="margin-bottom: 5px;">⭐ {row['Rating']:.2f} / 5</div>
                                <div class="metric-text">🧬 Similarity: {similarity_pct}%</div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        st.progress(min(max(row['Similarity'], 0.0), 1.0))
                        
                        # Artık temizlenmiş ve üst satıra taşınmış global fonksiyona bağlanan buton:
                        st.button(
                            "✨ Why This Book?", 
                            key=f"btn_{idx}", 
                            use_container_width=True,
                            on_click=open_details,
                            args=(row['Name'], row['Authors'], row['Rating'], cover_url, target_book_name, target_book_author)
                        )
                        st.write("") 
            else:
                st.error("An error occurred.")
        else:
            st.error("No books found.")

# --- 2. SECREEN: DETAILS ---
else:
    book = st.session_state.selected_book

    st.title("📖 Why do these books similar to each other? ")
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
        st.info(f"🔍 Connection Path: This book was recommended because you read and liked **'{book['target_title']}'** by *{book['target_author']}*.")

        st.markdown("### 🧠 Llama-3 AI Semantic Explanation")
        st.markdown("We've sent both books' metadata to the open-source Llama-3 model on Hugging Face to dissect their underlying narrative bond:")
        
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