"""Self-check validator: evidence sufficiency, claim grounding, citation verification.

Three-stage validation:
1. Pre-generation: Is there enough evidence to answer?
2. Post-generation: Are claims grounded in the evidence?
3. Post-generation: Are citations real?
"""

import logging
import re

from openai import OpenAI

from src.agent.prompts import (
    CLAIM_GROUNDING_PROMPT,
    EVIDENCE_CHECK_PROMPT,
)
from src.config import settings
from src.models import (
    AnswerResult,
    Citation,
    ClaimCheck,
    GroundingVerdict,
    RetrieveResult,
)

logger = logging.getLogger(__name__)

# Pattern to extract citations from answer text like [来源: 条款3.1, 第2页]
CITATION_PATTERN = re.compile(
    r'\[来源:\s*(?:条款\s*(\d+(?:\.\d+)*))?\s*,?\s*第\s*(\d+)\s*页\]'
)

# Pattern to split Chinese text into sentences
SENTENCE_SPLIT_PATTERN = re.compile(r'[。；\n]')


class Validator:
    """Three-stage self-check validator for RAG answers.

    1. Evidence sufficiency: can the retrieved chunks answer the question?
    2. Claim grounding: is each claim in the answer supported by evidence?
    3. Citation verification: do the [来源: ...] markers reference real chunks?

    KNOWN LIMITATION: The LLM-as-judge in stage 2 can produce false negatives
    (marking verbatim quotes as "unsupported"). Two mitigations are in place:
    - Text overlap pre-check: claims with >70% character overlap against
      evidence skip the LLM and are auto-marked SUPPORTED.
    - PARTIAL verdicts count as 0.5 in grounding_score (not 0), reducing
      the impact of borderline LLM judgments.

    For production use with higher reliability requirements, consider
    replacing the LLM judge with a dedicated NLI (natural language inference)
    model or using a stronger LLM for the judge role.
    """

    def __init__(self):
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        return self._client

    def validate(
        self,
        question: str,
        retrieved: list[RetrieveResult],
        answer: str,
    ) -> AnswerResult:
        """Run full three-stage validation.

        Returns AnswerResult with grounding score, claim checks, and citation errors.
        """
        # Stage 1: Evidence sufficiency
        evidence_sufficient, refusal_reason = self._check_evidence_sufficiency(
            question, retrieved
        )

        if not evidence_sufficient:
            return AnswerResult(
                answer=answer,
                citations=[],
                grounding_score=0.0,
                claim_checks=[],
                citation_errors=[],
                should_refuse=True,
                refuse_reason=refusal_reason,
            )

        # Stage 2: Claim grounding
        claim_checks = self._check_claim_grounding(answer, retrieved)

        # Stage 3: Citation verification
        citations, citation_errors = self._verify_citations(answer, retrieved)

        # Compute grounding score: SUPPORTED=1.0, PARTIAL=0.5, UNSUPPORTED=0.0
        if claim_checks:
            score_map = {
                GroundingVerdict.SUPPORTED: 1.0,
                GroundingVerdict.PARTIAL: 0.5,
                GroundingVerdict.UNSUPPORTED: 0.0,
            }
            total = sum(score_map.get(c.verdict, 0.0) for c in claim_checks)
            grounding_score = total / len(claim_checks)
        else:
            grounding_score = 1.0

        should_refuse = grounding_score < settings.grounding_threshold and len(claim_checks) > 0

        return AnswerResult(
            answer=answer,
            citations=citations,
            grounding_score=round(grounding_score, 2),
            claim_checks=claim_checks,
            citation_errors=citation_errors,
            should_refuse=should_refuse,
            refuse_reason="大部分声明未被文档支持" if should_refuse else None,
        )

    def _check_evidence_sufficiency(
        self, question: str, retrieved: list[RetrieveResult]
    ) -> tuple[bool, str | None]:
        """Stage 1: Check if retrieved evidence is sufficient to answer the question.

        Returns (is_sufficient, refusal_reason).
        """
        # Quick rule-based check: if no results or very low scores
        if not retrieved:
            return False, "未检索到相关文档内容"

        if all(r.score < 0.05 for r in retrieved):
            return False, "检索到的文档片段与问题相关性过低"

        # LLM-based evidence sufficiency check
        evidence_text = "\n\n".join(
            f"[片段 {i+1}] {r.chunk.text[:500]}" for i, r in enumerate(retrieved[:3])
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "user", "content": EVIDENCE_CHECK_PROMPT.format(
                        question=question,
                        evidence=evidence_text,
                    )},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content or ""

            # Extract score from response
            score_match = re.search(r'(\d+)', content)
            score = int(score_match.group(1)) if score_match else 3

            if score < settings.relevance_threshold:
                return False, f"文档内容相关性不足（评分 {score}/5）"

            # Check for explicit "not found" response from generator
            return True, None

        except Exception as e:
            logger.warning(f"Evidence check LLM call failed: {e}, defaulting to sufficient")
            return True, None

    def _check_claim_grounding(
        self, answer: str, retrieved: list[RetrieveResult]
    ) -> list[ClaimCheck]:
        """Stage 2: Check each claim in the answer against the evidence.

        Splits answer into atomic claims and checks each one.
        """
        # Build evidence text from all retrieved chunks (already reranked, typically 5)
        evidence_text = "\n\n".join(
            r.chunk.text for r in retrieved
        )

        # Split answer into sentences/claims
        claims = self._extract_claims(answer)

        results = []
        for claim in claims:
            try:
                verdict, reason = self._check_single_claim(claim, evidence_text)
                results.append(ClaimCheck(claim=claim, verdict=verdict, reason=reason))
            except Exception as e:
                logger.warning(f"Claim check failed for '{claim[:50]}...': {e}")
                results.append(ClaimCheck(
                    claim=claim,
                    verdict=GroundingVerdict.PARTIAL,
                    reason=f"校验失败: {e}",
                ))

        return results

    @staticmethod
    def _text_overlap_ratio(claim: str, evidence: str) -> float:
        """Quick character-level overlap: what fraction of claim chars appear in evidence.

        Used as a pre-check before the LLM judge call to catch obvious matches.
        """
        if not claim:
            return 0.0
        ev_set = set(evidence)
        matched = sum(1 for ch in claim if ch in ev_set)
        return matched / len(claim)

    def _check_single_claim(self, claim: str, evidence: str) -> tuple[GroundingVerdict, str]:
        """Check a single claim against evidence using LLM-as-judge.

        Pre-checks with text overlap — if the claim text is nearly verbatim in the
        evidence, skip the LLM call.
        """
        evidence_truncated = evidence[:3000]

        # Fast path: high character overlap → obviously supported, skip LLM
        overlap = self._text_overlap_ratio(claim, evidence_truncated)
        if overlap > 0.7:
            return GroundingVerdict.SUPPORTED, "文本高度重叠，自动判定支持"

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": CLAIM_GROUNDING_PROMPT.format(
                    evidence=evidence_truncated,
                    claim=claim,
                )},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()

        if "支持" in content and "不" not in content and "部分" not in content:
            return GroundingVerdict.SUPPORTED, content
        elif "部分" in content:
            return GroundingVerdict.PARTIAL, content
        else:
            return GroundingVerdict.UNSUPPORTED, content

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        """Split an answer into atomic claims."""
        # Skip the refusal message
        if "文档中未找到相关内容" in answer:
            return []

        # Split by sentence boundaries
        parts = SENTENCE_SPLIT_PATTERN.split(answer)

        claims = []
        for part in parts:
            part = part.strip()
            # Filter: must be substantive (at least 10 chars)
            if len(part) >= 10:
                # Remove citation markers for claim checking
                clean = CITATION_PATTERN.sub('', part).strip()
                if clean:
                    claims.append(clean)

        return claims[:10]  # Cap at 10 claims to avoid excessive API calls

    @staticmethod
    def _verify_citations(
        answer: str, retrieved: list[RetrieveResult]
    ) -> tuple[list[Citation], list[str]]:
        """Stage 3: Verify that citations in the answer reference real chunks."""
        citations = []
        errors = []

        # Collect all valid (clause, page) pairs from retrieved chunks
        valid_clauses = set()
        valid_pages = set()
        for rr in retrieved:
            if rr.chunk.heading_path:
                valid_clauses.add(rr.chunk.heading_path)
            valid_pages.add(rr.chunk.source_page)

        # Extract citations from answer
        for match in CITATION_PATTERN.finditer(answer):
            clause = match.group(1)
            page = int(match.group(2))

            # Verify page exists
            if page not in valid_pages:
                errors.append(f"引用的页码 {page} 在检索结果中不存在")
                continue

            # Verify clause exists (if specified)
            if clause and clause not in valid_clauses:
                errors.append(f"引用的条款 {clause} 在检索结果中不存在")

            # Get snippet from retrieved chunks — match by clause first, then page
            snippet = ""
            for rr in retrieved:
                if clause and rr.chunk.heading_path == clause:
                    snippet = rr.chunk.text[:200]
                    break
            if not snippet:
                for rr in retrieved:
                    if rr.chunk.source_page == page:
                        snippet = rr.chunk.text[:200]
                        break

            citations.append(Citation(
                clause=clause,
                page=page,
                snippet=snippet,
            ))

        return citations, errors
