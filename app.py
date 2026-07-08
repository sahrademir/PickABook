import streamlit as st
import pandas as pd
from src.preprocessing import load_data
from src.recommendation import build_model, recommend

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
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateY(-2px);
        background-color: #26354e;
    }
    .book-title { color: #ffffff; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
    .book-author { color: #a3b3cc; font-size: 14px; font-style: italic; }
    .metric-text { font-size: 13px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# 2. Application Launch  ve Cache
@st.cache_resource
def init_app():
    data = load_data()
    tfidf_matrix, indices = build_model(data)
    return data, tfidf_matrix, indices

df, tfidf_matrix, indices = init_app()

# 3. Hero Section
st.title("📖 PickABook")
st.markdown("### *AI-Powered Content-Based Book Recommendation Engine*")
st.markdown("---")

# 4. Autocomplete & Enter 

search_query = st.text_input(
    "What is the last book you read and enjoyed?",
    placeholder="Write a book name..."
)

# 5.Suggestions Listing Area
if search_query != "":
    st.write("") 
    st.markdown(f"#### 🎯 **'{search_query}'** Best Recommendations Similar to Your Book:")
    
    with st.spinner("Mathematical vectors are being compared, and recommendations are being prepared..."):
        results = recommend(search_query, df, tfidf_matrix, indices, top_n=6)
        
    if results is not None and not results.empty:
        st.write("")
        
        col1, col2, col3 = st.columns(3)
        
        for idx, row in results.reset_index().iterrows():
            current_col = [col1, col2, col3][idx % 3]
            
            with current_col:
                
                similarity_pct = int(row['Similarity'])
                
                # HTML Card Design
                card_html = f"""
                <div class="book-card">
                    <div class="book-title">📘 {row['Name']}</div>
                    <div class="book-author">✍️ {row['Authors']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
               # Star Rating and Similarity Progress Bar
                col_score, col_progress = st.columns([1, 1.2])
                with col_score:
                    st.markdown(f"<span class='metric-text'>⭐ {row['Rating']:.2f} / 5</span>", unsafe_allow_html=True)
                with col_progress:
                    st.markdown(f"<span class='metric-text'>🧬 Similarity: {similarity_pct}%</span>", unsafe_allow_html=True)
                    st.progress(min(max(row['Similarity'], 0.0), 1.0))
                
                st.write("") 
                
    else:
        st.error("I'm sorry, an error occurred while this book was being processed in our current subset.")
