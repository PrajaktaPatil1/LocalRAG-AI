import os
import uuid
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import settings
from .document_pipeline import DocumentParser, TextChunker
from .vector_store import HybridVectorStore
from .reranker import Reranker
from .query_router import QueryRouter
from .llm_service import OllamaService

app = FastAPI(
    title=settings.APP_NAME,
    description="Hybrid RAG + Ollama Engine with BM25, Cross-Encoder Reranking, Relevance Gating, and Streaming",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
vector_store = HybridVectorStore()
reranker = Reranker()
query_router = QueryRouter(vector_store=vector_store, reranker=reranker)
ollama_service = OllamaService()
chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)


class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = settings.DEFAULT_LLM_MODEL
    relevance_threshold: Optional[float] = settings.RELEVANCE_THRESHOLD
    top_k_retrieval: Optional[int] = settings.TOP_K_RETRIEVAL
    top_k_rerank: Optional[int] = settings.TOP_K_RERANK
    temperature: Optional[float] = 0.2
    force_mode: Optional[str] = "auto"  # "auto", "rag_only", "direct_only"


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "indexed_chunks": vector_store.count(),
        "ollama_url": settings.OLLAMA_BASE_URL
    }


@app.get("/api/models")
async def get_models():
    models = await ollama_service.list_models()
    return {"models": models, "default": settings.DEFAULT_LLM_MODEL}


@app.get("/api/documents")
def list_documents():
    docs = vector_store.list_documents()
    total_chunks = vector_store.count()
    return {"documents": docs, "total_chunks": total_chunks}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        doc_id = str(uuid.uuid4())[:8]
        safe_filename = os.path.basename(file.filename)
        saved_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{safe_filename}")
        
        try:
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process & chunk
            chunks = chunker.process_document(saved_path, doc_id=doc_id)
            if not chunks:
                results.append({
                    "filename": file.filename,
                    "doc_id": doc_id,
                    "status": "warning",
                    "message": "No readable text extracted from file."
                })
                continue
                
            # Index into Chroma + BM25
            vector_store.add_chunks(chunks)
            
            results.append({
                "filename": file.filename,
                "doc_id": doc_id,
                "status": "success",
                "chunks_created": len(chunks),
                "pages": max([c.page or 1 for c in chunks])
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            
    return {"uploaded": results, "total_chunks_now": vector_store.count()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        vector_store.delete_document(doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted.", "total_chunks_now": vector_store.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def execute_query(req: QueryRequest):
    """Non-streaming query execution."""
    # 1. Relevance Gating & Routing
    decision = query_router.route(
        query=req.query,
        relevance_threshold=req.relevance_threshold or settings.RELEVANCE_THRESHOLD,
        top_k_retrieval=req.top_k_retrieval or settings.TOP_K_RETRIEVAL,
        top_k_rerank=req.top_k_rerank or settings.TOP_K_RERANK,
        force_mode=req.force_mode or "auto"
    )

    citations = []
    if decision.mode == "rag_grounded":
        messages = ollama_service.format_grounded_prompt(req.query, decision.chunks)
        citations = ollama_service.extract_citations(decision.chunks)
    else:
        messages = ollama_service.format_direct_prompt(req.query, decision.decision_reason)

    # Collect full LLM answer
    full_answer = []
    target_model = req.model or settings.DEFAULT_LLM_MODEL
    async for token in ollama_service.stream_chat(target_model, messages, temperature=req.temperature or 0.2):
        full_answer.append(token)

    return {
        "answer": "".join(full_answer),
        "mode": decision.mode,
        "relevance_score": decision.relevance_score,
        "decision_reason": decision.decision_reason,
        "citations": citations
    }


@app.post("/api/query/stream")
async def execute_query_stream(req: QueryRequest):
    """
    Streaming query endpoint yielding Server-Sent Events (SSE).
    Emits metadata first, then streamed response tokens, followed by citation details.
    """
    # 1. Adaptive Relevance Gating & Smart Routing
    decision = query_router.route(
        query=req.query,
        relevance_threshold=req.relevance_threshold or settings.RELEVANCE_THRESHOLD,
        top_k_retrieval=req.top_k_retrieval or settings.TOP_K_RETRIEVAL,
        top_k_rerank=req.top_k_rerank or settings.TOP_K_RERANK,
        force_mode=req.force_mode or "auto"
    )

    citations = []
    if decision.mode == "rag_grounded":
        messages = ollama_service.format_grounded_prompt(req.query, decision.chunks)
        citations = ollama_service.extract_citations(decision.chunks)
    else:
        messages = ollama_service.format_direct_prompt(req.query, decision.decision_reason)

    target_model = req.model or settings.DEFAULT_LLM_MODEL

    async def event_generator():
        # Step 1: Send metadata packet
        meta_event = {
            "type": "meta",
            "mode": decision.mode,
            "relevance_score": decision.relevance_score,
            "threshold": decision.threshold_used,
            "decision_reason": decision.decision_reason,
            "citations": citations,
            "model": target_model
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # Step 2: Stream LLM tokens
        async for token in ollama_service.stream_chat(target_model, messages, temperature=req.temperature or 0.2):
            token_event = {
                "type": "token",
                "content": token
            }
            yield f"data: {json.dumps(token_event)}\n\n"

        # Step 3: Send completion packet
        done_event = {"type": "done"}
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
