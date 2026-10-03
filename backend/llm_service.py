import json
import httpx
from typing import List, Dict, Any, AsyncGenerator, Generator
from .config import settings

class OllamaService:
    """Manages Ollama LLM requests, streaming, and prompt grounding."""

    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> List[str]:
        """Fetches the list of installed Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"[Warning] Could not reach Ollama at {self.base_url}: {e}")
        return []

    def format_grounded_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Constructs prompt for Grounded Document RAG with inline citation instructions."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get("metadata", {}).get("source", "Document")
            pg = chunk.get("metadata", {}).get("page", 1)
            content = chunk.get("content", "").strip()
            score = chunk.get("rerank_score", 0.0)
            context_parts.append(
                f"--- [Doc #{i}] (Source: {src}, Page: {pg}, Relevance: {score:.2f}) ---\n{content}"
            )
            
        context_text = "\n\n".join(context_parts)
        
        system_instruction = (
            "You are a helpful, precise AI assistant equipped with user-provided document context.\n"
            "Your task is to answer the user's question accurately using ONLY the provided document context below.\n\n"
            "GUIDELINES:\n"
            "1. Base your answer strictly on the facts present in the document context.\n"
            "2. Whenever you state a claim or fact from a document, include a clear inline citation like `[Source: filename, Page: X]`.\n"
            "3. If the context does not contain enough information to fully answer, state what is known from the context and what is missing.\n"
            "4. Be structured, concise, and professional."
        )
        
        user_content = f"CONTEXT DOCUMENTS:\n{context_text}\n\nUSER QUESTION:\n{query}"
        
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def format_direct_prompt(self, query: str, reason: str = "") -> List[Dict[str, str]]:
        """Constructs prompt for Direct LLM answering (General Knowledge)."""
        system_instruction = (
            "You are a knowledgeable and helpful AI assistant.\n"
            "Answer the user's question directly, clearly, and concisely using your internal knowledge.\n"
            "If code or technical explanation is requested, provide high-quality, executable examples."
        )
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": query}
        ]

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts structured citations metadata from used chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            citations.append({
                "citation_num": i,
                "source": meta.get("source", "Document"),
                "page": meta.get("page", 1),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("rerank_score", 0.0),
                "snippet": chunk.get("content", "")[:280] + ("..." if len(chunk.get("content", "")) > 280 else "")
            })
        return citations

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """Streams tokens from Ollama chat API asynchronously."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        yield f"[Error] Ollama returned status {response.status_code}: {await response.aread()}"
                        return
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                token = chunk_data["message"]["content"]
                                yield token
                            if chunk_data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                yield f"\n[Connection Error with Ollama: {e}]"
