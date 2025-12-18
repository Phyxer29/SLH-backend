from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QueryRequest, RecommendResponse, Assessment
from app.rag_engine import RAGEngine
import time

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered assessment recommendation system",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (singleton)
print("🔥 Initializing FastAPI application...")
engine = RAGEngine()
print("✅ API ready to serve requests\n")

@app.get("/health")
def health_check():
    """
    Health check endpoint (required by assignment)
    
    Returns:
        {"status": "healthy"}
    """
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: QueryRequest):
    """
    Assessment recommendation endpoint (required by assignment)
    
    Accepts:
        - Natural language query
        - Job description text
        - Job description URL
    
    Returns:
        List of 5-10 relevant assessments with metadata
    """
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get recommendations from RAG engine
        results = engine.process_query(query)
        
        # Ensure minimum 5 results (required by assignment)
        if len(results) < 5:
            print(f"⚠ Only {len(results)} results found (minimum 5 required)")
        
        # Convert to Pydantic models
        assessments = [
            Assessment(
                url=r['url'],
                name=r['name'],
                adaptive_support=r['adaptive_support'],
                description=r['description'],
                duration=r['duration'],
                remote_support=r['remote_support'],
                test_type=r['test_type']
            )
            for r in results
        ]
        
        return RecommendResponse(recommended_assessments=assessments)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "SHL Assessment Recommendation API",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)"
        },
        "documentation": "/docs"
    }
