import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Hybrid RAG + Ollama Engine"
    DEBUG: bool = True
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "llama3.2")
    DEFAULT_EMBED_MODEL: str = os.getenv("DEFAULT_EMBED_MODEL", "all-MiniLM-L6-v2")
    
    # Storage Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    CHROMA_PERSIST_DIR: str = os.path.join(DATA_DIR, "chroma_db")
    
    # Chunking Configuration
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 8       # Initial candidates from Hybrid Search
    TOP_K_RERANK: int = 4          # Candidates after Reranking
    RRF_K: int = 60                # Reciprocal Rank Fusion constant
    RELEVANCE_THRESHOLD: float = 0.30  # Relevance gate cutoff (0.0 to 1.0)
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
