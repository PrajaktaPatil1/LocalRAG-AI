import math
from typing import List, Dict, Any

class Reranker:
    """Reranks candidate chunks against the user query using a Cross-Encoder or semantic scorer."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._cross_encoder = None
        self._loaded = False

    def _ensure_model(self):
        if not self._loaded:
            self._loaded = True
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
                print(f"[Info] CrossEncoder loaded: {self.model_name}")
            except Exception as e:
                print(f"[Warning] CrossEncoder load failed or disabled ({e}). Using semantic fallback ranker.")
                self._cross_encoder = None

    def _sigmoid(self, x: float) -> float:
        """Converts raw logits to a 0.0 - 1.0 probability score."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks.
        Returns candidates with 'rerank_score' attached, sorted in descending order.
        """
        if not candidates:
            return []
            
        self._ensure_model()
        if self._cross_encoder is not None:
            pairs = [[query, c["content"]] for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            
            reranked = []
            for item, raw_score in zip(candidates, scores):
                # Standardize to 0.0-1.0
                norm_score = self._sigmoid(float(raw_score))
                updated_item = dict(item)
                updated_item["rerank_score"] = round(norm_score, 4)
                updated_item["raw_rerank_score"] = round(float(raw_score), 4)
                reranked.append(updated_item)
                
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
            
        else:
            # Fallback when CrossEncoder is not available: use a composite of dense and RRF scores
            reranked = []
            for item in candidates:
                dense = item.get("dense_score", 0.0)
                rrf = item.get("rrf_score", 0.0)
                # Composite score between 0 and 1
                fallback_score = min(1.0, max(0.0, (dense * 0.7) + (rrf * 15.0 * 0.3)))
                updated_item = dict(item)
                updated_item["rerank_score"] = round(fallback_score, 4)
                reranked.append(updated_item)
                
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
