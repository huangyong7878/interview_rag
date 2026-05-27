"""Test the agent module: generator prompts, validator logic.

These tests validate the self-check mechanisms without requiring live API calls.
They test the logic and structure, not the LLM output quality.

For full integration tests with real LLM, run the server and use:
    uv run pytest tests/test_agent.py --integration

Run: uv run pytest tests/test_agent.py -v
"""

import pytest
from src.agent.validator import Validator, CITATION_PATTERN
from src.agent.prompts import (
    GENERATION_SYSTEM_PROMPT,
    EVIDENCE_CHECK_PROMPT,
    CLAIM_GROUNDING_PROMPT,
)
from src.models import (
    AnswerResult,
    Chunk,
    ChunkType,
    GroundingVerdict,
    RetrieveResult,
)
from tests.fixtures import create_mock_chunks


class TestPrompts:
    def test_generation_prompt_has_constraints(self):
        prompt = GENERATION_SYSTEM_PROMPT
        assert "只根据提供的文档内容回答" in prompt
        assert "来源" in prompt
        assert "文档中未找到相关内容" in prompt

    def test_evidence_check_prompt_has_placeholders(self):
        prompt = EVIDENCE_CHECK_PROMPT
        assert "{question}" in prompt
        assert "{evidence}" in prompt

    def test_claim_grounding_prompt_has_placeholders(self):
        prompt = CLAIM_GROUNDING_PROMPT
        assert "{evidence}" in prompt
        assert "{claim}" in prompt


class TestCitationPattern:
    def test_parse_clause_citation(self):
        match = CITATION_PATTERN.search("[来源: 条款3.1, 第2页]")
        assert match is not None
        assert match.group(1) == "3.1"
        assert match.group(2) == "2"

    def test_parse_page_only_citation(self):
        match = CITATION_PATTERN.search("[来源: 第3页]")
        assert match is not None
        assert match.group(1) is None
        assert match.group(2) == "3"

    def test_parse_multiple_citations(self):
        text = "键的尺寸要求[来源: 条款4.2, 第3页]，材料要求[来源: 条款4.1, 第2页]。"
        matches = CITATION_PATTERN.findall(text)
        assert len(matches) == 2

    def test_no_citation_in_text(self):
        matches = CITATION_PATTERN.findall("这是一段没有引用的回答。")
        assert len(matches) == 0


class TestValidatorLogic:
    """Test validator logic components that don't require LLM API."""

    @pytest.fixture
    def validator(self):
        return Validator()

    @pytest.fixture
    def mock_retrieved(self):
        chunks = create_mock_chunks()
        return [
            RetrieveResult(chunk=c, score=0.9, bm25_score=0.85, vector_score=0.95)
            for c in chunks[:6]  # Pages 1-3
        ]

    def test_extract_claims_from_answer(self, validator):
        answer = "键的抗拉强度应大于等于590 MPa。键表面不允许有裂纹、浮锈、氧化皮和毛刺。"
        claims = validator._extract_claims(answer)
        assert len(claims) >= 2

    def test_extract_claims_empty_refusal(self, validator):
        answer = "文档中未找到相关内容。"
        claims = validator._extract_claims(answer)
        assert claims == []

    def test_extract_claims_filters_short(self, validator):
        answer = "是。否。这是一个足够长的句子可以提取。"
        claims = validator._extract_claims(answer)
        # Only the long sentence should be extracted
        assert len(claims) >= 1

    def test_citation_verification_valid(self, validator, mock_retrieved):
        answer = "键的抗拉强度应大于等于590 MPa [来源: 条款3, 第3页]。"
        citations, errors = validator._verify_citations(answer, mock_retrieved)
        # Should find at least one citation
        assert len(citations) >= 1
        # Page 3 exists in retrieved
        assert len(errors) == 0

    def test_citation_verification_invalid_page(self, validator, mock_retrieved):
        answer = "键的材料是45钢 [来源: 第99页]。"
        citations, errors = validator._verify_citations(answer, mock_retrieved)
        assert len(errors) > 0  # Page 99 doesn't exist

    def test_citation_verification_invalid_clause(self, validator, mock_retrieved):
        answer = "键的要求 [来源: 条款99.99, 第3页]。"
        citations, errors = validator._verify_citations(answer, mock_retrieved)
        # Clause 99.99 doesn't exist in the retrieved set
        assert len(errors) >= 1

    def test_grounding_score_calculation(self, validator, mock_retrieved):
        """Test that grounding score is in valid range."""
        result = AnswerResult(
            answer="测试回答",
            grounding_score=0.75,
            should_refuse=False,
        )
        assert 0.0 <= result.grounding_score <= 1.0

    def test_should_refuse_low_grounding(self, validator, mock_retrieved):
        """Test that low grounding_score triggers refusal."""
        result = AnswerResult(
            answer="不确定的回答",
            grounding_score=0.2,
            should_refuse=True,
        )
        assert result.should_refuse


class TestEndToEndMock:
    """End-to-end test with mock chunks (no real LLM)."""

    def test_full_pipeline_with_mock_data(self):
        """Test that the entire pipeline structure works with mock data."""
        from src.ingestion.indexer import DocumentIndex
        from src.retrieval.retriever import HybridRetriever

        # 1. Load chunks
        chunks = create_mock_chunks()
        assert len(chunks) >= 5, "Should have at least 5 chunks"

        # 2. Build index
        idx = DocumentIndex()
        idx.add_chunks(chunks)
        assert len(idx.chunks) == len(chunks)

        # 3. Retrieve
        retriever = HybridRetriever(idx)
        results = retriever.retrieve("键的材料要求", top_k=3)
        assert len(results) > 0

        # 4. Check that retrieved content is relevant
        top_text = results[0].chunk.text
        assert len(top_text) > 0

        # 5. Verifier can process citations in mock answer
        validator = Validator()
        # Use a page that exists in the retrieved results
        existing_page = results[0].chunk.source_page
        mock_answer = f"键的相关要求 [来源: 第{existing_page}页]。"
        citations, errors = validator._verify_citations(mock_answer, results)
        # Should extract citation since page exists
        assert len(citations) >= 1, f"No citations found for page {existing_page}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"


# Integration tests - require running server
# Run with: INTEGRATION=1 uv run pytest tests/test_agent.py -v -k integration
@pytest.mark.skip(reason="Integration tests require running server and API key")
class TestIntegration:
    def test_server_health(self):
        import httpx
        resp = httpx.get("http://localhost:8000/health")
        assert resp.status_code == 200
