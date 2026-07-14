import pickle
import faiss
from fastapi import APIRouter, HTTPException
from api.schemas import (
    ContentBasedRequest, 
    SemanticSearchRequest, 
    UserRecommendationRequest, 
    RecommendationResponse
)

from src.recommendation import recommend as content_recommend, build_model
from src.recommendation_dl import recommend_dl 
from src.recommendation_llm import get_llm_explanation
from src.user_recommendation import recommend_user_profile

GLOBAL_DATA = None
GLOBAL_FAISS_INDEX = None
GLOBAL_TFIDF_MATRIX = None
GLOBAL_TFIDF_INDICES = None

try:
    print("Loading backend models, please wait...")
    
    
    with open("data/books_metadata.pkl", "rb") as f:
        GLOBAL_DATA = pickle.load(f)
    
    GLOBAL_FAISS_INDEX = faiss.read_index("data/books_index.faiss")
    
    GLOBAL_TFIDF_MATRIX, GLOBAL_TFIDF_INDICES = build_model(GLOBAL_DATA)
    
    print("Backend models successfully loaded and ready! 🚀")
except Exception as e:
    print(f"⚠️ Critical Error loading models on backend: {e}")
    GLOBAL_DATA = None
    GLOBAL_FAISS_INDEX = None
    GLOBAL_TFIDF_MATRIX = None
    GLOBAL_TFIDF_INDICES = None

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


# ==================== 1. TF-IDF (CONTENT-BASED) ROUTE ====================
@router.post("/content", response_model=RecommendationResponse)
def get_content_recommendations(request: ContentBasedRequest):
    if GLOBAL_DATA is None or GLOBAL_TFIDF_MATRIX is None:
        raise HTTPException(status_code=500, detail="TF-IDF model or metadata is not loaded on server.")
        
    try:
        raw_results = content_recommend(
            request.book_title, 
            df=GLOBAL_DATA, 
            tfidf_matrix=GLOBAL_TFIDF_MATRIX, 
            indices=GLOBAL_TFIDF_INDICES, 
            top_n=request.top_n
        )

        if raw_results is None or raw_results.empty:
            raise HTTPException(status_code=404, detail=f"Book '{request.book_title}' not found.")

        target_row = GLOBAL_DATA[GLOBAL_DATA["Name"] == request.book_title]
        target_author = target_row["Authors"].values[0] if not target_row.empty else "Unknown Author"
        
        formatted_results = []
        for _, row in raw_results.iterrows():
            rec_author = row.get("Authors", "Unknown Author")

            try:
                explanation = get_llm_explanation(
                    request.book_title,
                    target_author,
                    row.get("Name"),
                    rec_author
                )
            except Exception as e:
                print(f"⚠️ LLM API Error: {e}")
                explanation = "We cannot reach the llm right now."
            
            raw_sim = float(row.get("Similarity", 0.0))
            if 0.0 < raw_sim <= 1.0:
                sim_score = raw_sim * 100.0
            else:
                sim_score = raw_sim

            formatted_results.append({
                "title": row.get("Name", "Unknown Title"),
                "isbn": row.get("ISBN") or row.get("isbn") or None,
                "similarity_score": round(sim_score, 2),
                "llm_explanation": explanation
            })
            
        return {
            "status": "success",
            "method": "Content-Based (TF-IDF)",
            "results": formatted_results
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 2. DEEP LEARNING (FAISS SEMANTIC) ROUTE ====================
@router.post("/semantic", response_model=RecommendationResponse)
def get_semantic_recommendations(request: SemanticSearchRequest):
    if GLOBAL_DATA is None or GLOBAL_FAISS_INDEX is None:
        raise HTTPException(status_code=500, detail="FAISS models or metadata are not loaded on server.")
        
    try:
        
        matched_books = GLOBAL_DATA[GLOBAL_DATA['Name'] == request.query_text]
        if matched_books.empty:
            raise HTTPException(status_code=404, detail=f"Book '{request.query_text}' not found in database.")
            
        target_idx = matched_books.index[0]
        target_author = matched_books.iloc[0]["Authors"]
        
        
        target_vector = GLOBAL_FAISS_INDEX.reconstruct(int(target_idx)).reshape(1, -1)
        distances, indices = GLOBAL_FAISS_INDEX.search(target_vector, request.top_n + 1)
        
        
        sim_indices = indices[0][1:]
        sim_distances = distances[0][1:]
        
        min_d, max_d = min(sim_distances), max(sim_distances)
        span = (max_d - min_d) if (max_d - min_d) > 0 else 1
        similarities = [0.95 - ((d - min_d) / span) * 0.4 for d in sim_distances]
        
        raw_results = GLOBAL_DATA.iloc[sim_indices].copy()
        raw_results['Similarity'] = similarities
        
        formatted_results = []
        for idx, row in raw_results.iterrows():
            rec_author = row.get("Authors", "Unknown Author")
            
            try:
                explanation = get_llm_explanation(
                    request.query_text,
                    target_author,
                    row.get("Name"),
                    rec_author
                )
            except Exception as e:
                print(f"⚠️ LLM API Error: {e}")
                explanation = "We cannot reach the llm right now."
                
            sim_pct = int(row['Similarity'] * 100)
            
            formatted_results.append({
                "title": row.get("Name", "Unknown Title"),
                "isbn": row.get("ISBN") or row.get("isbn") or None,
                "similarity_score": float(sim_pct),
                "llm_explanation": explanation
            })
            
        return {
            "status": "success",
            "method": "Deep Learning (FAISS Semantic)",
            "results": formatted_results
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 3. USER FAVORITES PORT ====================
@router.post("/user-blend", response_model=RecommendationResponse)
def get_user_recommendations(request: UserRecommendationRequest):
    if GLOBAL_DATA is None or GLOBAL_FAISS_INDEX is None:
        raise HTTPException(status_code=500, detail="FAISS models or metadata are not loaded on server.")
        
    try:
        raw_results = recommend_user_profile(
            request.favorite_books,     # 1. favorite_titles
            GLOBAL_DATA,                # 2. df
            GLOBAL_FAISS_INDEX,         # 3. index (FAISS index nesnesi!)
            top_n=request.top_n         # 4. top_n
        )
        
        if raw_results is None or raw_results.empty:
            raise HTTPException(status_code=404, detail="No personalized recommendations could be generated.")
            
        ref_book = request.favorite_books[0] if request.favorite_books else "Unknown Book"
        ref_row = GLOBAL_DATA[GLOBAL_DATA["Name"] == ref_book]
        ref_author = ref_row["Authors"].values[0] if not ref_row.empty else "Unknown Author"

        formatted_results = []
        for _, row in raw_results.iterrows():
            rec_author = row.get("Authors", "Unknown Author")
            
            try:
                explanation = get_llm_explanation(
                    ref_book,
                    ref_author,
                    row.get("Name"),
                    rec_author
                )
            except Exception as e:
                print(f"⚠️ LLM API Error: {e}")
                explanation = f"Recommended based on your library reading preferences like {ref_book}."

            raw_sim = float(row.get("Similarity", 0.0))
            
            if raw_sim <= 1.0:
                sim_pct = (1.0 - raw_sim) * 100.0
            else:
                sim_pct = raw_sim
                
            formatted_results.append({
                "title": row.get("Name", "Unknown Title"),
                "isbn": row.get("ISBN") or row.get("isbn") or None,
                "similarity_score": round(max(0.0, min(100.0, sim_pct)), 2),
                "llm_explanation": explanation
            })
            
        return {
            "status": "success",
            "method": "User Profile (FAISS Semantic)",
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))