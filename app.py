import streamlit as st
import pandas as pd
import urllib.parse
from src.preprocessing import load_data
from src.recommendation import build_model, recommend
from src.recommendation_dl import build_dl_model, recommend_dl

# 1.Wide screen and modern header 
st.set_page_config(
    page_title="PickABook | AI Recommendation", 
    page_icon="📖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Part
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
        min-height: 400px;
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
        display.flex;
        justify-content:center;
        align-items: center;
        margin-bottom: 15px
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

# 2. Application Launch  ve Cache
@st.cache_resource
def init_all_models():
    data = load_data()
    tfidf_matrix, indices = build_model(data)
    dl_embeddings, dl_indices = build_dl_model(data)
    return data, tfidf_matrix, indices, dl_embeddings, dl_indices

data, tfidf_matrix, tfidf_indices, dl_embeddings, dl_indices = init_all_models()

# 3. Hero Section
st.title("📖 PickABook")
st.markdown("### *AI-Powered Content-Based Book Recommendation Engine*")
st.markdown("---")

col_search, col_model = st.columns([2,1])

with col_search:
    search_query = st.text_input(
        "What is the last book you read and enjoyed?",
        placeholder="Type the full book title and press Enter...",
        key="book_search_input"
    )

with col_model:
    model_choice = st.radio(
        "🧠 Select the Recommendation Algorithm:",
        ["TF-IDF (Word-Matching Based)", "Deep Learning (Semantic Search)"],
        help="TF-IDF looks only at word similarities. Deep Learning, on the other hand, uses a Transformer model to focus on the meaning of the sentence, even if the words differ."
    )


#5.Generates a cover URL from Open Library
def get_cover_url(ISBN):
    encoded_isbn = urllib.parse.quote(ISBN)
    return f"https://covers.openlibrary.org/b/isbn/{ISBN}-M.jpg?default=false"

# 6.Suggestions Listing Area
if search_query != "":
    st.write("") 
    
    matched_books = data[data["Name"].str.lower().str.contains(search_query.lower(), na=False)]
    
    if not matched_books.empty:
       
        target_book_name = matched_books.sort_values(by="Rating", ascending=False).iloc[0]["Name"]
        
        if target_book_name != search_query:
            st.info(f"🔍 '{search_query}' The best match for your search has been selected: **{target_book_name}**")
            
        st.markdown(f"#### 🎯 **'{target_book_name}' Best Recommendations Similar to Your Book:**")
        
        if "TF-IDF" in model_choice:
            with st.spinner("The TF-IDF matrix is ​​being calculated..."):
                results = recommend(target_book_name, data, tfidf_matrix, tfidf_indices, top_n=6)
        else:
            with st.spinner("The Transformer performs deep semantic search..."):
                results = recommend_dl(target_book_name, data, dl_embeddings, dl_indices, top_n=6)
            
        if results is not None and not results.empty:
            st.write("")
            col1, col2, col3 = st.columns(3)
            
            for idx, row in results.reset_index().iterrows():
                current_col = [col1, col2, col3][idx % 3]
                
                with current_col:
                    similarity_pct = int(row['Similarity'] * 100)
                    placeholder_url = "https://via.placeholder.com/180x240/1f293d/ffffff?text=No+Cover"
                    
                    if 'ISBN' in row and pd.notna(row['ISBN']):
                        cover_url = f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg?default=false"
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
                    st.write("") 
        else:
            st.error("An error occurred while calculating recommendations.")
    else:
        st.error("No books matching the word you entered were found. Please try another word.")