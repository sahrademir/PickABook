import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as recommendation_router

app = FastAPI(
    tittle="PickABook AI Recommendation Engine",
    description="Microservice backend for hybrid, semantic and blendedd book recommendations.",
    version="1.0.0"
)

# CORS MIDDLEWARE (Streamlit Connection Permisson)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(recommendation_router)

@app.get("/")
def root():
    return{
        "message": "Welcome to PickABook AI Recommendation API",
        "status": "healthy",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)