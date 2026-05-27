"""Answer generator using LLM with retrieved evidence."""

import logging

from openai import OpenAI

from src.agent.prompts import GENERATION_SYSTEM_PROMPT, RAG_ANSWER_PROMPT
from src.config import settings
from src.models import RetrieveResult

logger = logging.getLogger(__name__)


class Generator:
    """LLM-based answer generator with evidence-grounded prompting."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )

    def generate(
        self, question: str, retrieved: list[RetrieveResult]
    ) -> str:
        """Generate an answer grounded in the retrieved evidence.

        Args:
            question: The user's question.
            retrieved: Top-k retrieved chunks with scores.

        Returns:
            Generated answer text with source citations.
        """
        # Build evidence block
        evidence_parts = []
        for i, rr in enumerate(retrieved):
            chunk = rr.chunk
            source_info = []
            if chunk.heading_path:
                source_info.append(f"条款 {chunk.heading_path}")
            source_info.append(f"第{chunk.source_page}页")
            source_label = "，".join(source_info)

            evidence_parts.append(
                f"[文档片段 {i + 1}]（{source_label}）\n{chunk.text}"
            )

        evidence_text = "\n\n---\n\n".join(evidence_parts)

        # Build the user prompt
        user_prompt = RAG_ANSWER_PROMPT.format(
            evidence=evidence_text,
            question=question,
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"答案生成失败：{e}"
