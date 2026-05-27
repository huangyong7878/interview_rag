"""Reranker: LLM-based pointwise relevance scoring for retrieved chunks.

After wide recall (top_k=20), the reranker scores each (query, chunk) pair
on a 1-5 relevance scale using a single batched LLM call. Results are then
sorted and trimmed back to top_k for answer generation.

This fixes the "blind scoring" problem where BM25 + vector weights
can't tell if a chunk actually answers the question vs just matching keywords.
"""

import logging
import re

from openai import OpenAI

from src.config import settings
from src.models import RetrieveResult

logger = logging.getLogger(__name__)

BATCH_RERANK_PROMPT = """你是一个检索质量评估员。评估以下每个文档片段与用户问题的相关性。

评分标准：
1 = 完全不相关  2 = 稍微相关  3 = 部分相关  4 = 相关  5 = 高度相关

用户问题：{question}

{chunks_text}

请按以下格式逐条给出评分（只输出数字和编号，不要解释）：
<0>: <score>
<1>: <score>
...
"""


class LLMReranker:
    """LLM-based batch reranker.

    Scores all chunks in a single LLM call by listing them together and
    asking for per-chunk scores in a structured format.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )

    def rerank(
        self, question: str, results: list[RetrieveResult], top_k: int = 5
    ) -> list[RetrieveResult]:
        """Rerank retrieved results by LLM relevance score.

        Args:
            question: User's question.
            results: Wide recall results (up to 20).
            top_k: Final number of results after reranking.

        Returns:
            Top-k results sorted by LLM relevance score.
        """
        if len(results) <= top_k:
            return results

        # Build batch scoring prompt
        chunks_parts = []
        for i, rr in enumerate(results):
            text = rr.chunk.text[:300]  # Truncate per chunk — just need enough for relevance judge
            heading = rr.chunk.heading_title or rr.chunk.heading_path or ""
            label = f"[{heading}] " if heading else ""
            chunks_parts.append(f"<{i}> {label}{text}")

        prompt = BATCH_RERANK_PROMPT.format(
            question=question,
            chunks_text='\n\n'.join(chunks_parts),
        )

        scores: dict[int, int] = {}
        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content or ""
            logger.info(f"Reranker raw response ({len(content)} chars): {content[:200]}")
            scores = self._parse_batch_scores(content, len(results))
        except Exception as e:
            logger.warning(f"Batch rerank failed: {e}")

        # Score each result, fallback to 3 if parsing failed
        scored = []
        for i, rr in enumerate(results):
            llm_score = scores.get(i, 3)
            scored.append((rr, llm_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rr, llm_score in scored[:top_k]:
            normalized_llm = llm_score / 5.0
            combined = 0.4 * rr.score + 0.6 * normalized_llm
            reranked.append(RetrieveResult(
                chunk=rr.chunk,
                score=round(combined, 4),
                bm25_score=rr.bm25_score,
                vector_score=rr.vector_score,
            ))

        return reranked

    def _parse_batch_scores(self, content: str, expected_count: int) -> dict[int, int]:
        """Parse batch rerank output like '<0>: 4', '<1>: 2', etc."""
        scores = {}
        for line in content.strip().split('\n'):
            m = re.search(r'<\s*(\d+)\s*>\s*[:：]?\s*([1-5])', line)
            if m:
                idx = int(m.group(1))
                score = int(m.group(2))
                if 0 <= idx < expected_count:
                    scores[idx] = score

        if not scores:
            # Fallback: try to find any 1-5 digits
            digits = re.findall(r'\b([1-5])\b', content)
            for i, d in enumerate(digits[:expected_count]):
                scores[i] = int(d)

        logger.info(f"Parsed {len(scores)}/{expected_count} rerank scores")
        return scores
