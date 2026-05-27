"""Hybrid retriever: BM25 + Vector search with clause & table boosting."""

import logging
import re

import numpy as np

from src.config import settings
from src.ingestion.indexer import DocumentIndex
from src.models import Chunk, ChunkType, QueryType, RetrieveResult
from src.retrieval.query_analyzer import (
    classify_query,
    extract_clause_numbers,
    extract_section_numbers,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 keyword scores and FAISS vector scores.

    DESIGN NOTE: The table_boost mechanism (1.3x for TABLE-type chunks) only
    activates when chunks have chunk_type=TABLE. With PaddleOCR (which outputs
    flat text), no chunks are classified as TABLE — the table data is embedded
    in CLAUSE chunks as plain text. When switching to VLM or MinerU (which
    produce Markdown tables), this boost automatically becomes active.

    BM25 uses char-level n-grams (1-3) via TfidfVectorizer, which works
    for Chinese without a separate tokenizer — no jieba dependency needed.
    """

    def __init__(self, index: DocumentIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrieveResult]:
        """Retrieve the most relevant chunks for a query.

        Returns top_k results sorted by combined score.
        """
        top_k = top_k or settings.retrieval_top_k
        query_type = classify_query(query)
        clause_nums = extract_clause_numbers(query) + extract_section_numbers(query)
        logger.info(
            f"Retrieval start: query_type={query_type.value}, "
            f"clause_nums={clause_nums}, top_k={top_k}"
            f"chunks_total={len(self.index.chunks)}"
        )

        # Get BM25 scores for all chunks
        logger.info("Step 1/3: BM25 search...")
        bm25_indices, bm25_scores = self.index.bm25_search(query)
        top5_indices = bm25_indices[:5].tolist()
        top5_scores = [round(float(bm25_scores[i]), 4) for i in top5_indices]
        logger.info(
            f"  BM25 done: top5_indices={top5_indices}, top5_scores={top5_scores}"
        )

        # Get vector scores if available
        has_vector = False
        vector_scores_dict: dict[int, float] = {}
        logger.info("Step 2/3: Vector search (if FAISS available)...")
        query_vec = self.index.get_query_embedding(query)
        if query_vec is not None and self.index.faiss_index is not None:
            has_vector = True
            logger.info("  FAISS index available, running vector search...")
            vec_indices, vec_scores = self.index.vector_search(query_vec, top_k * 2)
            for idx, score in zip(vec_indices, vec_scores):
                vector_scores_dict[int(idx)] = float(score)
            logger.info(
                f"  Vector done: top5_indices={vec_indices[:5].tolist()}, "
                f"top5_scores={[round(float(s), 4) for s in vec_scores[:5]]}"
            )
        else:
            logger.info("  FAISS not available, using BM25 only")

        # Combine scores
        logger.info("Step 3/3: Combining scores and applying boosts...")
        results: list[RetrieveResult] = []
        # Score all chunks
        n = len(self.index.chunks)
        for i in range(n):
            bm25 = float(bm25_scores[i])
            vec = vector_scores_dict.get(i, 0.0)

            # Hybrid score: weighted combination
            if has_vector:
                combined = settings.bm25_weight * bm25 + settings.vector_weight * vec
            else:
                combined = bm25  # BM25 only

            # Apply boosts
            chunk = self.index.chunks[i]
            boost = self._compute_boost(chunk, query_type, clause_nums)
            combined *= boost

            if combined > 0:
                results.append(RetrieveResult(
                    chunk=chunk,
                    score=round(combined, 4),
                    bm25_score=round(bm25, 4),
                    vector_score=round(vec, 4),
                ))

        # Sort by score descending, take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        logger.info(f"Retrieval done: {len(results)} results, returning top {min(top_k, len(results))}")
        return results[:top_k]

    def _compute_boost(
        self, chunk: Chunk, query_type: QueryType, clause_nums: list[str]
    ) -> float:
        """Compute relevance boost for a chunk based on query analysis."""
        boost = 1.0

        # Clause number exact match boost
        if clause_nums and chunk.heading_path:
            for cn in clause_nums:
                if chunk.heading_path == cn or chunk.heading_path.startswith(cn + '.'):
                    boost *= settings.clause_boost
                    break
                # Partial path match (parent clause)
                if cn.startswith(chunk.heading_path):
                    boost *= 1.2

        # Table query → boost table chunks
        if query_type == QueryType.TABLE_QUERY and chunk.chunk_type == ChunkType.TABLE:
            boost *= settings.table_boost

        return boost
