"""Application configuration from environment variables."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from .env file and environment variables."""

    # OpenAI-compatible API (LLM generation, reranking, validation)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"

    # VLM API (vision-language model for PDF page → Markdown parsing)
    # Falls back to openai_* if not set
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""

    # PDF parser priority: comma-separated list of "paddleocr", "vlm_api", "mineru"
    # e.g. "paddleocr,vlm_api,mineru" (default) or "vlm_api" (skip others)
    parser_priority: str = "paddleocr,vlm_api,mineru"

    # Local embedding model (FlagEmbedding / BGE)
    knowledge_embedding_model: str = "BAAI/bge-m3"
    enable_faiss: bool = True  # Set to false to skip FAISS vector index (BM25-only)

    # Retrieval
    max_chunk_size: int = 1000
    retrieval_top_k: int = 5
    retrieval_wide_top_k: int = 20  # Wide recall before reranking
    bm25_weight: float = 0.6
    vector_weight: float = 0.4
    clause_boost: float = 1.5
    table_boost: float = 1.3

    # OCR/Image
    image_dpi: int = 200

    # Generation
    temperature: float = 0.1
    max_tokens: int = 2048

    # Grounding thresholds
    grounding_threshold: float = 0.5
    relevance_threshold: int = 3  # 1-5 scale for evidence sufficiency

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
