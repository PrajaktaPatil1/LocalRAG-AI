from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .config import settings
from .vector_store import HybridVectorStore
from .reranker import Reranker

@dataclass
class RoutingDecision:
    mode: str                     # "rag_grounded" or "direct_llm"
    relevance_score: float        # Highest confidence score
    threshold_used: float         # Threshold used for decision
    decision_reason: str          # Explanation of routing choice
    chunks: List[Dict[str, Any]]  # Chunks to be used as context (if any)
    hybrid_candidates: List[Dict[str, Any]]  # Raw candidates retrieved


class QueryRouter:
    """
    Adaptive Relevance Gate & Router.
    Decides whether to use Grounded Document RAG or direct LLM general knowledge.
    """
    def __init__(self, vector_store: HybridVectorStore, reranker: Reranker):
        self.vector_store = vector_store
        self.reranker = reranker

    def route(
        self,
        query: str,
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        top_k_retrieval: int = settings.TOP_K_RETRIEVAL,
        top_k_rerank: int = settings.TOP_K_RERANK,
        force_mode: str = "auto"   # "auto", "rag_only", "direct_only"
    ) -> RoutingDecision:
        
        # Explicit overrides
        if force_mode == "direct_only":
            return RoutingDecision(
                mode="direct_llm",
                relevance_score=0.0,
                threshold_used=relevance_threshold,
                decision_reason="User manually selected Direct LLM mode.",
                chunks=[],
                hybrid_candidates=[]
            )
            
        total_docs = self.vector_store.count()
        if total_docs == 0:
            return RoutingDecision(
                mode="direct_llm",
                relevance_score=0.0,
                threshold_used=relevance_threshold,
                decision_reason="No documents have been uploaded to knowledge base. Answering from general LLM knowledge.",
                chunks=[],
                hybrid_candidates=[]
            )

        # 1. Hybrid Search (Dense + BM25 via RRF)
        hybrid_candidates = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k_retrieval,
            rrf_k=settings.RRF_K
        )

        if not hybrid_candidates:
            return RoutingDecision(
                mode="direct_llm",
                relevance_score=0.0,
                threshold_used=relevance_threshold,
                decision_reason="No matching candidate chunks found in index. Answering from general LLM knowledge.",
                chunks=[],
                hybrid_candidates=[]
            )

        # 2. Cross-Encoder Reranker
        reranked_chunks = self.reranker.rerank(
            query=query,
            candidates=hybrid_candidates,
            top_k=top_k_rerank
        )

        top_score = reranked_chunks[0]["rerank_score"] if reranked_chunks else 0.0

        if force_mode == "rag_only":
            return RoutingDecision(
                mode="rag_grounded",
                relevance_score=top_score,
                threshold_used=relevance_threshold,
                decision_reason="User forced RAG mode.",
                chunks=reranked_chunks,
                hybrid_candidates=hybrid_candidates
            )

        # 3. Relevance Gate check
        if top_score >= relevance_threshold:
            # Filter only chunks that have sufficient relevance
            qualified_chunks = [c for c in reranked_chunks if c["rerank_score"] >= (relevance_threshold * 0.7)]
            if not qualified_chunks:
                qualified_chunks = reranked_chunks[:1]
                
            return RoutingDecision(
                mode="rag_grounded",
                relevance_score=top_score,
                threshold_used=relevance_threshold,
                decision_reason=f"Found high-confidence document context (Top Rerank Score: {top_score:.2f} >= Threshold {relevance_threshold:.2f}). Grounding response on documents.",
                chunks=qualified_chunks,
                hybrid_candidates=hybrid_candidates
            )
        else:
            return RoutingDecision(
                mode="direct_llm",
                relevance_score=top_score,
                threshold_used=relevance_threshold,
                decision_reason=f"Query is out of document context (Top Rerank Score: {top_score:.2f} < Threshold {relevance_threshold:.2f}). Answering smoothly from general LLM knowledge.",
                chunks=[],
                hybrid_candidates=hybrid_candidates
            )
