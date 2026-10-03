# 🧠 Hybrid RAG with Adaptive Routing, Reranker & Streaming Response

A production-grade, 100% local **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **Streamlit**, **ChromaDB + BM25 Hybrid Search**, **Cross-Encoder Reranking**, **Relevance Gating**, and **Ollama**.

---

## 🏗️ System Architecture

```
                    ┌────────────────────────┐
                    │   Streamlit Web UI     │ (Real-time Token Streaming & Citations)
                    └───────────┬────────────┘
                                │ SSE Stream
                                ▼
                    ┌────────────────────────┐
                    │     FastAPI Server     │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
       DOCUMENT PIPELINE                   QUERY PIPELINE
              │                                   │
       Document Parser                            │
    (PDF / DOCX / TXT / MD)                       │
              │                                   │
       Recursive Chunking                         │
              │                                   │
       Local Embeddings                           │
              │                                   │
              ▼                                   ▼
     ┌──────────────────┐               ┌──────────────────┐
     │  Chroma (Dense)  │               │   Dense Vector   │
     │        +         │◄──────────────┤        +         │ (Hybrid Search via RRF)
     │   BM25 (Sparse)  │               │    BM25 Search   │
     └──────────────────┘               └─────────┬────────┘
                                                  │
                                                  ▼
                                            Top-K Candidates
                                                  │
                                                  ▼
                                          Cross-Encoder Reranker
                                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │    Relevance Gate    │
                                       └──────────┬───────────┘
                                                  │
                   ┌──────────────────────────────┴──────────────────────────────┐
                   │ (Score >= Threshold)                                        │ (Score < Threshold or No Docs)
                   ▼                                                             ▼
         📄 [RAG GROUNDED MODE]                                       🤖 [DIRECT LLM MODE]
      Build Context + Citations Prompt                               Direct General Knowledge Prompt
                   │                                                             │
                   └───────────────────────┬─────────────────────────────────────┘
                                           │
                                           ▼
                                     🦙 Ollama LLM
                                (Async Token Streaming)
                                           │
                                           ▼
                             Grounded Answer + Citations
```

---

## 🌟 Key Features

1. **Hybrid Search (Dense + Sparse via RRF)**:
   - **Dense Vectors**: Semantic vector retrieval using local embeddings (`all-MiniLM-L6-v2` / HuggingFace).
   - **Sparse BM25**: Exact keyword and token-level matching via `rank_bm25`.
   - **Reciprocal Rank Fusion (RRF)** combines rankings seamlessly:
     $$RRF\_Score(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$

2. **Cross-Encoder Reranker**:
   - Re-evaluates top hybrid candidates using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to capture deep query-document cross-attention.

3. **Adaptive Relevance Gating (Smart Routing)**:
   - **Contextual Query**: When the query matches uploaded document content with confidence above the relevance threshold, it performs strict **Grounded RAG** with verified citations (`[Source: file.pdf, Page: X]`).
   - **General Query**: When the user asks a general question or something outside the documents, the relevance gate detects low retrieval scores and routes smoothly to **Direct LLM Mode** without hallucinating or breaking.

4. **Real-time Streaming Responses**:
   - FastAPI emits **Server-Sent Events (SSE)** with structured metadata, streamed token-by-token generation, and final citation cards.

5. **Multi-Format Ingestion**:
   - Ingests **PDF**, **DOCX**, **TXT**, **MD**, and **CSV** files with automatic page and paragraph tracking.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- Pull your preferred model (e.g. `llama3.2` or `mistral`):
  ```bash
  ollama pull llama3.2
  ```

### 2. Run with One Command
```bash
./run.sh
```

### 3. Manual Start
```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Backend (Terminal 1)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start Frontend (Terminal 2)
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser!

---

## 🔌 API Endpoints

- `GET /api/health` — Health check & indexed chunk count
- `GET /api/models` — Installed Ollama models
- `GET /api/documents` — List indexed documents and chunk statistics
- `POST /api/documents/upload` — Upload & batch-index files
- `DELETE /api/documents/{doc_id}` — Delete a document from vector & BM25 index
- `POST /api/query/stream` — Real-time streaming RAG query endpoint (SSE)
