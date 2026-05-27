"""Index builder: BM25 (TF-IDF) + FAISS vector dual index.

Embeddings are generated locally via FlagEmbedding / BAAI/bge-m3.
No external embedding API calls needed.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.models import Chunk

logger = logging.getLogger(__name__)

# bge-m3 outputs 1024-dimensional dense vectors
BGE_M3_DIM = 1024


class DocumentIndex:
    """Dual index for document chunks: BM25 (char n-gram TF-IDF) + FAISS vectors.

    FAISS vector search is optional. On macOS, FlagEmbedding (torch) can
    segfault during BGE-M3 model loading. Set ENABLE_FAISS=false in .env
    to skip FAISS entirely and use BM25-only retrieval. The LLM Reranker
    partially compensates for the loss of semantic matching.
    """


    def __init__(self):
        self.chunks: list[Chunk] = []
        self.chunk_texts: list[str] = []

        # BM25-like TF-IDF
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None

        # FAISS
        self.embedding_dim: int = BGE_M3_DIM
        self.faiss_index = None

        # Lazy-loaded embedding model
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load the FlagEmbedding BGE-M3 model.

        Returns the model or None if FlagEmbedding is not installed / fails to load.
        """
        if not settings.enable_faiss:
            return None
        if self._embedding_model is None:
            try:
                logger.info("Importing FlagEmbedding...")
                from FlagEmbedding import BGEM3FlagModel
                logger.info("FlagEmbedding imported OK")
            except ImportError:
                logger.warning("FlagEmbedding not installed, FAISS vector search disabled")
                return None

            model_name = settings.knowledge_embedding_model
            logger.info(f"Loading embedding model: {model_name}")
            try:
                logger.info("Creating BGEM3FlagModel instance (use_fp16=True)...")
                import sys
                print("[indexer] About to instantiate BGEM3FlagModel...", flush=True)
                self._embedding_model = BGEM3FlagModel(
                    model_name,
                    use_fp16=True,
                )
                logger.info("BGEM3FlagModel loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = None
                return None
        return self._embedding_model

    def add_chunks(self, chunks: list[Chunk]):
        """Add chunks and build both indices."""
        self.chunks = chunks
        self.chunk_texts = [c.text for c in chunks]

        # Build BM25 index
        self._build_bm25_index()

        # Build FAISS index
        self._build_faiss_index()

        logger.info(f"Indexed {len(chunks)} chunks (BM25 + FAISS/bge-m3)")

    def _build_bm25_index(self):
        """Build char-level n-gram TF-IDF matrix for Chinese text."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def _build_faiss_index(self):
        """Generate embeddings with BGE-M3 and build FAISS index."""
        if not settings.enable_faiss:
            logger.info("FAISS disabled by config (enable_faiss=false), skipping vector index")
            return
        model = self._get_embedding_model()
        if model is None:
            logger.warning("No embedding model available, FAISS vector index disabled")
            return

        try:
            # BGE-M3 encode returns dict with 'dense_vecs'
            output = model.encode(
                self.chunk_texts,
                batch_size=8,
                max_length=8192,
            )
            embeddings = output['dense_vecs']
        except Exception as e:
            logger.warning(f"Embedding generation failed, skipping FAISS: {e}")
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.embedding_dim = embeddings_array.shape[1]

        # Build FAISS index (inner product with normalized vectors = cosine)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)

    def get_query_embedding(self, query: str) -> np.ndarray | None:
        """Generate embedding vector for a query string using BGE-M3."""
        if self.faiss_index is None:
            return None
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            output = model.encode(
                [query],
                batch_size=1,
                max_length=8192,
            )
            vec = np.array(output['dense_vecs'], dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            return vec
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return None

    def bm25_search(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """BM25-style search via TF-IDF cosine similarity.

        Returns (indices, scores) where indices are sorted by descending score
        and scores[i] is the raw (unsorted) score for chunk i.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Sort by score descending
        indices = np.argsort(scores)[::-1]
        return indices, scores  # unsorted: scores[i] = score of chunk i

    def vector_search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """FAISS vector similarity search.

        Returns (indices, scores).
        """
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return indices[0], scores[0]

    def save(self, dir_path: str | Path):
        """Persist indices and chunks to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save chunks as JSON
        with open(dir_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f, ensure_ascii=False, indent=2)

        # Save FAISS index if exists
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(dir_path / "faiss.index"))

        logger.info(f"Index saved to {dir_path}")

    def load(self, dir_path: str | Path):
        """Load indices and chunks from disk."""
        dir_path = Path(dir_path)

        # Load chunks
        with open(dir_path / "chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = [Chunk(**d) for d in data]
            self.chunk_texts = [c.text for c in self.chunks]

        # Rebuild BM25
        self._build_bm25_index()

        # Load FAISS
        faiss_path = dir_path / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            self.embedding_dim = self.faiss_index.d

        logger.info(f"Index loaded from {dir_path} ({len(self.chunks)} chunks)")
