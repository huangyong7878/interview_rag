"""FastAPI application entry point.

Endpoints:
- POST /ingest: Upload PDF, run ingestion pipeline, build index.
- POST /query: Ask a question, retrieve evidence, generate answer, self-check.
- GET  /health: Health check.
"""

import hashlib
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.agent.generator import Generator
from src.agent.validator import Validator
from src.config import settings
from src.ingestion.indexer import DocumentIndex
from src.ingestion.loader import load_pdf
from src.ingestion.parser import parse_pdf
from src.ingestion.splitter import split_by_headings
from src.models import (
    AnswerResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.retrieval.reranker import LLMReranker
from src.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
MARKDOWN_CACHE = OUTPUT_DIR / "parsed.md"
META_CACHE = OUTPUT_DIR / "meta.json"


def _hash_file(file_path: str | Path) -> str:
    """SHA256 hash of file content for cache key."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Global state
_index: DocumentIndex | None = None
_retriever: HybridRetriever | None = None
_reranker: LLMReranker | None = None
_generator: Generator | None = None
_validator: Validator | None = None


def get_index() -> DocumentIndex:
    if _index is None:
        raise HTTPException(status_code=400, detail="请先上传并解析文档 (POST /ingest)")
    return _index


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(get_index())
    return _retriever


def get_reranker() -> LLMReranker:
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker()
    return _reranker


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def get_validator() -> Validator:
    global _validator
    if _validator is None:
        _validator = Validator()
    return _validator


def _auto_load_cache():
    """On startup, load cached index if available."""
    global _index, _retriever
    if (OUTPUT_DIR / "chunks.json").exists():
        try:
            _index = DocumentIndex()
            _index.load(OUTPUT_DIR)
            _retriever = None
            logger.info(f"Auto-loaded cached index: {len(_index.chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to auto-load index: {e}")


app = FastAPI(
    title="智能文档问答 Agent",
    description="RAG-based Q&A system for scanned PDF documents",
    version="0.1.0",
    on_startup=[_auto_load_cache],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "chunk_count": len(_index.chunks) if _index else 0,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-parse even if cached"),
):
    """Upload a PDF, parse it, and build the search index.

    Caches the parsed Markdown so restarts or re-ingests of the same
    file skip the expensive MinerU/VLM step.
    """
    global _index, _retriever

    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只接受 PDF 文件")

    # Save uploaded file to temp location
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        file_hash = _hash_file(tmp_path)

        # Check cache
        ocr_method = "vlm_api"
        if not force and MARKDOWN_CACHE.exists() and META_CACHE.exists():
            meta = json.loads(META_CACHE.read_text())
            if meta.get("file_hash") == file_hash:
                logger.info("Using cached Markdown, skipping MinerU/VLM")
                full_markdown = MARKDOWN_CACHE.read_text(encoding="utf-8")
                ocr_method = meta.get("ocr_method", "cached")
            else:
                full_markdown, ocr_method = await _do_parse(tmp_path)
        else:
            full_markdown, ocr_method = await _do_parse(tmp_path)

        # Cache the result
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MARKDOWN_CACHE.write_text(full_markdown, encoding="utf-8")
        META_CACHE.write_text(json.dumps({
            "file_hash": file_hash,
            "ocr_method": ocr_method,
            "filename": file.filename,
        }))

        # Step 3: Split Markdown → chunks (clause-aware)
        logger.info("Splitting into chunks...")
        chunks = split_by_headings(full_markdown)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 4: Build index (BM25 + FAISS)
        logger.info("Building search index...")
        idx = DocumentIndex()
        idx.add_chunks(chunks)
        _index = idx
        _retriever = None

        # Save index to output directory
        idx.save(OUTPUT_DIR)

        # Report page count
        pdf_info = load_pdf(tmp_path)  # Quick call for metadata
        page_count = pdf_info.page_count

        return IngestResponse(
            status="ok" if not force else "ok (forced)",
            page_count=page_count,
            chunk_count=len(chunks),
            ocr_method=ocr_method,
            markdown_preview=full_markdown[:1000],
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档解析失败: {e}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _do_parse(tmp_path: str) -> tuple[str, str]:
    """Run the full parse pipeline (MinerU or VLM fallback)."""
    logger.info("Parsing PDF to Markdown...")
    pdf_info = load_pdf(tmp_path)
    logger.info(f"Loaded {pdf_info.page_count} pages (scanned={pdf_info.is_scanned})")
    full_markdown, ocr_method = await parse_pdf(Path(tmp_path), pdf_info.pages)
    logger.info(f"Parsed {pdf_info.page_count} pages using: {ocr_method}")
    return full_markdown, ocr_method


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question about the ingested document."""
    index = get_index()
    retriever = get_retriever()
    generator = get_generator()
    validator = get_validator()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"[query] 收到问题: {question}")

    # Step 1: Wide recall
    wide_top_k = settings.retrieval_wide_top_k
    logger.info(f"[query] Step 1/3: 宽召回 ({wide_top_k})...")
    wide_results = retriever.retrieve(question, top_k=wide_top_k)
    logger.info(f"[query] → 宽召回完成, 获得 {len(wide_results)} 条结果")

    if not wide_results:
        logger.info("[query] → 无检索结果，拒答")
        return QueryResponse(
            question=question,
            result=AnswerResult(
                answer="文档中未找到相关内容。",
                grounding_score=0.0,
                should_refuse=True,
                refuse_reason="未检索到相关文档片段",
            ),
            retrieved_chunks=[],
        )

    # Step 2: Rerank — LLM pointwise relevance scoring
    final_top_k = req.top_k or settings.retrieval_top_k
    reranker = get_reranker()
    logger.info(f"[query] Step 2/3: Reranker 精排, {len(wide_results)} → {final_top_k}...")
    retrieved = reranker.rerank(question, wide_results, top_k=final_top_k)
    logger.info(f"[query] → 精排完成, {len(retrieved)} 条结果")

    # Step 3: Generate + Validate
    logger.info("[query] Step 3/3: LLM 生成 + 三阶段自检...")
    answer = generator.generate(question, retrieved)
    validation_result = validator.validate(question, retrieved, answer)

    # Build debug info (show both pre and post rerank)
    retrieved_summary = [
        {
            "chunk_id": r.chunk.id,
            "heading": r.chunk.heading_path,
            "page": r.chunk.source_page,
            "type": r.chunk.chunk_type.value,
            "score": r.score,
            "text_preview": r.chunk.text[:200],
        }
        for r in retrieved
    ]

    return QueryResponse(
        question=question,
        result=validation_result,
        retrieved_chunks=retrieved_summary,
    )
