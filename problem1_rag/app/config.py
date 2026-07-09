"""
Central config. Everything is read from environment variables (.env is
loaded via python-dotenv). Nothing secret is hardcoded here.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


class Settings:
    # --- LLM (answer generation) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings ---
    # "sentence-transformers" (real, needs internet to download model once) or
    # "hash" (deterministic offline fallback used for CI / no-internet testing)
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = _int("EMBEDDING_DIM", 384)  # matches all-MiniLM-L6-v2

    # --- Chunking ---
    CHUNK_SIZE: int = _int("CHUNK_SIZE", 800)       # characters
    CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 120)  # characters

    # --- Vector store (Chroma, embedded/local — no always-on server) ---
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_corpus")

    # --- Retrieval ---
    DEFAULT_TOP_K: int = _int("DEFAULT_TOP_K", 5)

    # --- Corpus ---
    CORPUS_DIR: str = os.getenv("CORPUS_DIR", "./data/corpus")


settings = Settings()
