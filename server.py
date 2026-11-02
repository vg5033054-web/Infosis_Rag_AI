import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from rag_engine import RAGEngine

app = FastAPI(title="Infosys Enterprise RAG AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine singleton
engine = RAGEngine()
engine.load_or_build_index()

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = 5
    mode: Optional[str] = "hybrid"
    fiscal_year: Optional[str] = "all"
    conversation_history: Optional[List[Dict[str, str]]] = None
    temperature: Optional[float] = 0.2

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "hybrid"
    fiscal_year: Optional[str] = "all"

class UpdateKeyRequest(BaseModel):
    api_key: str

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "is_indexed": engine.is_indexed,
        "stats": engine.indexing_stats,
        "models": config.MODELS,
        "default_model": config.DEFAULT_MODEL,
        "has_api_key": bool(engine.api_key and len(engine.api_key) > 10)
    }

@app.get("/api/models")
def get_models():
    return {
        "models": config.MODELS,
        "default": config.DEFAULT_MODEL
    }

@app.post("/api/search")
def search_documents(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = engine.search(
        query=req.query,
        top_k=req.top_k or 5,
        mode=req.mode or "hybrid",
        fiscal_year=req.fiscal_year or "all"
    )
    return {
        "query": req.query,
        "results_count": len(results),
        "results": results
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    response = engine.answer_query(
        query=req.query,
        model=req.model or config.DEFAULT_MODEL,
        top_k=req.top_k or 5,
        mode=req.mode or "hybrid",
        fiscal_year=req.fiscal_year or "all",
        conversation_history=req.conversation_history or [],
        temperature=req.temperature or 0.2
    )
    return response

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator():
        stream_gen = engine.answer_query_stream(
            query=req.query,
            model=req.model or config.DEFAULT_MODEL,
            top_k=req.top_k or 5,
            mode=req.mode or "hybrid",
            fiscal_year=req.fiscal_year or "all",
            conversation_history=req.conversation_history or [],
            temperature=req.temperature or 0.2
        )
        
        for item in stream_gen:
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0.005)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/key")
def update_key(req: UpdateKeyRequest):
    key = req.api_key.strip()
    if not key or len(key) < 10:
        raise HTTPException(status_code=400, detail="Invalid API key provided")
    
    # Test key with a fast call
    try:
        from groq import Groq
        test_client = Groq(api_key=key)
        test_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        engine.api_key = key
        engine.groq_client = test_client
        return {"status": "success", "message": "Groq API key validated and updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to validate Groq key: {str(e)}")

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index_page():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Infosys RAG Chatbot is initializing...</h1>"

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 64)
    print("  Infosys Enterprise RAG AI Assistant Web Server")
    print("  Web Interface: http://localhost:8000 (or http://127.0.0.1:8000)")
    print("  Grounded in 752 Pages across FY24-25 and FY25-26 Reports")
    print("=" * 64 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

