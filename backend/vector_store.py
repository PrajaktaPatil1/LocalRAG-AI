import os
import re
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .config import settings
from .document_pipeline import DocumentChunk

class EmbeddingManager:
    """Manages local or Ollama embeddings."""
    def __init__(self, model_name: str = settings.DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._st_model = None
        self._loaded = False

    def _ensure_model(self):
        if not self._loaded:
            self._loaded = True
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
                print(f"[Info] Embedding model loaded: {self.model_name}")
            except Exception as e:
                print(f"[Warning] SentenceTransformer load failed ({e}). Falling back to simple embeddings.")
                self._st_model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        self._ensure_model()
        if self._st_model is not None:
            embedding = self._st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        return [0.0] * 384


class HybridVectorStore:
    """Combines ChromaDB (Dense Vector Search) with BM25 (Sparse Keyword Search) using RRF."""
    
    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = EmbeddingManager(settings.DEFAULT_EMBED_MODEL)
        
        # BM25 In-memory Index
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[Dict[str, Any]] = []  # [{chunk_id, content, metadata}]
        self._rebuild_bm25_from_chroma()

    def _tokenize(self, text: str) -> List[str]:
        """Simple lower-case alphanumeric tokenization."""
        return re.findall(r'\w+', text.lower())

    def _rebuild_bm25_from_chroma(self):
        """Initializes or refreshes BM25 index from all documents stored in Chroma."""
        try:
            results = self.collection.get()
            if results and results["ids"]:
                ids = results["ids"]
                documents = results["documents"]
                metadatas = results["metadatas"]
                
                self.bm25_corpus = []
                tokenized_corpus = []
                
                for i in range(len(ids)):
                    chunk_id = ids[i]
                    content = documents[i]
                    meta = metadatas[i] if metadatas else {}
                    self.bm25_corpus.append({
                        "chunk_id": chunk_id,
                        "content": content,
                        "metadata": meta
                    })
                    tokenized_corpus.append(self._tokenize(content))
                    
                if tokenized_corpus:
                    self.bm25_index = BM25Okapi(tokenized_corpus)
                else:
                    self.bm25_index = None
            else:
                self.bm25_corpus = []
                self.bm25_index = None
        except Exception as e:
            print(f"[Error] Failed to rebuild BM25 index: {e}")
            self.bm25_corpus = []
            self.bm25_index = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Adds a list of document chunks to Chroma and updates the BM25 index."""
        if not chunks:
            return
        
        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metas = [c.metadata for c in chunks]
        
        # Dense embeddings
        embeddings = self.embedder.embed_documents(texts)
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas
        )
        
        # Refresh BM25
        self._rebuild_bm25_from_chroma()

    def delete_document(self, doc_id: str):
        """Deletes all chunks associated with a doc_id."""
        self.collection.delete(where={"doc_id": doc_id})
        self._rebuild_bm25_from_chroma()

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns aggregated summary of documents in the store."""
        results = self.collection.get()
        if not results or not results["ids"]:
            return []
        
        docs_map = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            source = meta.get("source", "unknown")
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_count": 0,
                    "pages": set()
                }
            docs_map[doc_id]["chunk_count"] += 1
            if "page" in meta and meta["page"]:
                docs_map[doc_id]["pages"].add(meta["page"])
                
        # Format pages
        output = []
        for d in docs_map.values():
            output.append({
                "doc_id": d["doc_id"],
                "source": d["source"],
                "chunk_count": d["chunk_count"],
                "total_pages": len(d["pages"]) if d["pages"] else 1
            })
        return output

    def dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Executes dense vector search in Chroma."""
        if self.count() == 0:
            return []
        
        query_vector = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        ranked_list = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            
            for i in range(len(ids)):
                # Chroma cosine distance = 1 - cosine_similarity. So similarity = 1 - distance
                sim = max(0.0, 1.0 - distances[i]) if distances else 0.5
                ranked_list.append({
                    "chunk_id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "dense_score": sim,
                    "rank": i + 1
                })
        return ranked_list

    def sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Executes BM25 sparse keyword search."""
        if not self.bm25_index or not self.bm25_corpus:
            return []
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25_index.get_scores(tokenized_query)
        # Pair with corpus indices
        scored_corpus = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_corpus.sort(key=lambda x: x[1], reverse=True)
        
        top_candidates = scored_corpus[:top_k]
        ranked_list = []
        for rank, (idx, score) in enumerate(top_candidates, start=1):
            item = self.bm25_corpus[idx]
            ranked_list.append({
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "bm25_score": float(score),
                "rank": rank
            })
        return ranked_list

    def hybrid_search(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL, rrf_k: int = settings.RRF_K) -> List[Dict[str, Any]]:
        """
        Executes Dense + Sparse Hybrid Search and blends results using Reciprocal Rank Fusion (RRF).
        RRF Score = sum( 1 / (rrf_k + rank) )
        """
        total_items = self.count()
        if total_items == 0:
            return []
        
        fetch_k = min(top_k * 2, max(total_items, 1))
        dense_results = self.dense_search(query, top_k=fetch_k)
        sparse_results = self.sparse_search(query, top_k=fetch_k)
        
        # Merge by chunk_id using RRF
        merged_scores: Dict[str, Dict[str, Any]] = {}
        
        # 1. Process Dense
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_val = 1.0 / (rrf_k + rank)
            merged_scores[cid] = {
                "chunk_id": cid,
                "content": item["content"],
                "metadata": item["metadata"],
                "rrf_score": rrf_val,
                "dense_score": item.get("dense_score", 0.0),
                "bm25_score": 0.0,
                "dense_rank": rank,
                "bm25_rank": None
            }
            
        # 2. Process Sparse BM25
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            rrf_val = 1.0 / (rrf_k + rank)
            if cid in merged_scores:
                merged_scores[cid]["rrf_score"] += rrf_val
                merged_scores[cid]["bm25_score"] = item.get("bm25_score", 0.0)
                merged_scores[cid]["bm25_rank"] = rank
            else:
                merged_scores[cid] = {
                    "chunk_id": cid,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": rrf_val,
                    "dense_score": 0.0,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "dense_rank": None,
                    "bm25_rank": rank
                }
                
        # Sort by RRF score descending
        hybrid_candidates = list(merged_scores.values())
        hybrid_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        return hybrid_candidates[:top_k]
