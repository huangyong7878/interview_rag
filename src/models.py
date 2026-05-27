"""Data models for the RAG system."""

from enum import Enum
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    CLAUSE = "clause"
    TABLE = "table"
    PARAGRAPH = "paragraph"
    HEADER = "header"


class QueryType(str, Enum):
    CLAUSE_LOOKUP = "clause_lookup"
    TABLE_QUERY = "table_query"
    FACTUAL = "factual"
    UNRELATED = "unrelated"


class Chunk(BaseModel):
    """A document chunk with metadata."""

    id: str
    text: str
    source_page: int
    heading_path: str | None = None  # e.g. "3.1"
    heading_level: int | None = None  # 1, 2, 3, ...
    heading_title: str | None = None  # e.g. "键的尺寸"
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    table_data: dict | None = None  # {headers: [...], rows: [[...], ...]}


class RetrieveResult(BaseModel):
    """A single retrieval result."""

    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0


class Citation(BaseModel):
    """A citation reference in an answer."""

    clause: str | None = None
    page: int
    snippet: str


class GroundingVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ClaimCheck(BaseModel):
    """Result of checking a single claim against evidence."""

    claim: str
    verdict: GroundingVerdict
    reason: str = ""


class AnswerResult(BaseModel):
    """Complete answer with self-check results."""

    answer: str
    citations: list[Citation] = []
    grounding_score: float = 0.0  # fraction of claims SUPPORTED
    claim_checks: list[ClaimCheck] = []
    citation_errors: list[str] = []  # citations referencing non-existent clauses
    should_refuse: bool = False
    refuse_reason: str | None = None


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    status: str
    page_count: int
    chunk_count: int
    ocr_method: str  # "mineru" or "vlm_api"
    markdown_preview: str = ""


class QueryRequest(BaseModel):
    """Request to ask a question."""

    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    """Response with answer and metadata."""

    question: str
    result: AnswerResult
    retrieved_chunks: list[dict] = []  # summary of retrieved chunks for debugging
