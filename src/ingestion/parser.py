"""Parser: Convert scanned PDF to structured Markdown.

Strategy: PaddleOCR → VLM API → MinerU (fastest to highest quality).
- PaddleOCR: Fast local OCR, Chinese-optimized, plain text output.
- VLM API: Per-page multimodal LLM, structured Markdown with tables.
- MinerU: Highest quality, best for complex layouts/tables/formulas.
"""

import asyncio
import base64
import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)

# Prompt for VLM fallback: instruct the vision model to produce structured Markdown
VLM_PARSE_PROMPT = """You are a document digitization assistant. Convert this scanned page image from a Chinese national standard (GB/T) document into clean, structured Markdown.

Rules:
1. Preserve the heading hierarchy using Markdown headings (#, ##, ###, ####).
   - Top-level clause numbers like "1", "2", "3" → use ##
   - Sub-clauses like "3.1", "3.2" → use ###
   - Sub-sub-clauses like "3.1.1" → use ####
2. Convert ALL tables to Markdown table format (| col1 | col2 | ... |).
3. Preserve technical symbols, units (MPa, mm, etc.), and numbers exactly.
4. Preserve clause numbers (like "3.1", "4.1.1") as they appear.
5. Do NOT add any content that is not on the page.
6. Do NOT include any explanations or meta-commentary — output ONLY the Markdown.
7. If the page has a footer/header with page number or document number, include it in a blockquote.
8. Subscripts and superscripts: use HTML tags <sub> and <sup> if needed.

Output ONLY valid Markdown, nothing else."""


def _find_mineru_cli() -> Optional[str]:
    """Find the MinerU CLI executable.

    Checks for 'mineru' (v3.x) and 'magic-pdf' (legacy).
    """
    for name in ["mineru", "magic-pdf"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _run_mineru(pdf_path: Path, output_dir: Path) -> str:
    """Run MinerU CLI on a PDF and return the combined Markdown output.

    MinerU produces output in: {output_dir}/{pdf_name}/{method}/
    The markdown is typically in an 'auto' or 'ocr' subdirectory.

    stderr is streamed to Python logger in real-time so progress is visible.
    """
    cli = _find_mineru_cli()
    if not cli:
        raise RuntimeError("MinerU CLI not found")

    cmd = [cli, "-p", str(pdf_path), "-o", str(output_dir), "-b", "pipeline"]

    logger.info(f"Running MinerU: {' '.join(cmd)}")
    logger.info("MinerU first run may take 3-10 minutes (loading models + OCR + layout analysis)...")

    # Use Popen to stream stderr in real-time
    import sys
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,                 # Line-buffered
    )

    stdout_lines = []
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"[MinerU] {line}")
            stdout_lines.append(line)

    process.wait(timeout=900)

    if process.returncode != 0:
        last_lines = "\n".join(stdout_lines[-20:]) if stdout_lines else "(no output)"
        raise RuntimeError(f"MinerU failed (code {process.returncode}):\n{last_lines}")

    # MinerU writes output to: output_dir/{pdf_stem}/{method}/*.md
    pdf_stem = pdf_path.stem
    base = output_dir / pdf_stem

    # Try 'auto' first (v3.x), then 'ocr', then scan for any .md files
    markdown_parts = []
    for method_dir in ["auto", "ocr"]:
        md_dir = base / method_dir
        if md_dir.exists():
            for md_file in sorted(md_dir.glob("*.md")):
                markdown_parts.append(md_file.read_text(encoding="utf-8"))
            if markdown_parts:
                break

    # Fallback: find any .md files under the output
    if not markdown_parts:
        for md_file in sorted(base.rglob("*.md")):
            markdown_parts.append(md_file.read_text(encoding="utf-8"))

    if not markdown_parts:
        raise RuntimeError(f"MinerU produced no .md files in {base}")

    return "\n\n".join(markdown_parts)


class MinerUParser:
    """Parse PDF to Markdown using local MinerU CLI."""

    def is_available(self) -> bool:
        return _find_mineru_cli() is not None

    async def parse(self, pdf_path: Path) -> str:
        """Run MinerU on the original PDF and return combined Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "mineru_output"
            output_dir.mkdir(parents=True)

            loop = asyncio.get_running_loop()
            markdown = await loop.run_in_executor(
                None, _run_mineru, pdf_path, output_dir
            )
            return markdown


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class PaddleOCRParser:
    """Fast local OCR using PaddleOCR. Chinese-optimized, no API cost.

    Output is plain text with basic page markers — no table structure
    or heading hierarchy.

    KNOWN LIMITATION: The flat-text output means:
    - Tables are NOT rendered as Markdown tables. Table data appears as
      inline text (e.g. "键宽b1.01.01.5"), which mistletoe cannot detect
      as Table tokens. This makes table_boost in the retriever dead code
      and table queries rely solely on BM25 keyword matching.
    - Headings are NOT marked with ##/###. The splitter's _normalize_headings()
      workaround recovers clause numbers from plain text like "1 范围",
      but this is a heuristic patch — VLM or MinerU should be preferred
      when structured output is needed.
    - For text-based PDFs (is_scanned=False), this parser performs
      redundant OCR on page images. A TextExtractor parser using
      page.get_text("markdown") would be faster and preserve structure.
    """


    def __init__(self):
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            logger.info("Loading PaddleOCR...")
            self._ocr = PaddleOCR(lang='ch')
        return self._ocr

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False

    async def parse_pages(self, pages: list) -> str:
        """Extract text from all pages using PaddleOCR.

        Returns plain text with page markers, sorted top-to-bottom.
        """
        import numpy as np

        loop = asyncio.get_running_loop()
        results = []

        for page in pages:
            # PaddleOCR 3.x accepts numpy array or file path, not PIL Image
            img_array = np.array(page.image)

            # Run OCR in thread pool (PaddleOCR is synchronous)
            ocr_result = await loop.run_in_executor(
                None, lambda arr=img_array: self.ocr.predict(arr)
            )

            # PaddleOCR 3.x returns list of dicts, each with "rec_texts" array
            lines = []
            if ocr_result:
                for item in ocr_result:
                    if isinstance(item, dict):
                        for text in item.get("rec_texts", []):
                            if text.strip():
                                lines.append(text)

            page_text = "\n".join(lines) if lines else "(PaddleOCR: no text detected)"
            results.append(f"<!-- page {page.page_num} -->\n\n{page_text}")

        return "\n\n".join(results)


class VLMParser:
    """Fallback parser using a multimodal LLM (OpenAI-compatible vision API).

    Uses VLM-specific config (vlm_base_url, vlm_api_key, vlm_model) when set,
    otherwise falls back to the main LLM config (openai_base_url, etc.).
    """

    def __init__(self):
        base_url = settings.vlm_base_url or settings.openai_base_url
        api_key = settings.vlm_api_key or settings.openai_api_key
        self.model = settings.vlm_model or settings.llm_model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    async def parse_page_image(self, image: Image.Image, page_num: int) -> str:
        """Send a single page image to vision LLM and get Markdown back."""
        data_url = f"data:image/png;base64,{image_to_base64(image)}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PARSE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


async def parse_pdf(pdf_path: Path, pages: list) -> tuple[str, str]:
    """Parse a PDF to Markdown, returning (markdown, method_used).

    Parser priority is configurable via PARSER_PRIORITY in .env.
    Each method is tried in order; on failure the next one is used.

    Available methods:
    - paddleocr: Fast local Chinese OCR, plain text output.
    - vlm_api: Multimodal LLM per page, structured Markdown, API cost.
    - mineru: Highest quality, local, slow.

    Args:
        pdf_path: Path to the original PDF file (for MinerU).
        pages: List of PageImage objects (for PaddleOCR/VLM).

    Returns:
        (full_markdown, method) where method is one of the configured priorities.
    """
    methods = [m.strip() for m in settings.parser_priority.split(",") if m.strip()]
    if not methods:
        raise RuntimeError(
            "No parser configured. Set PARSER_PRIORITY in .env, e.g.:\n"
            "  PARSER_PRIORITY=vlm_api\n"
            "Available: paddleocr, vlm_api, mineru"
        )

    skip_reasons: list[str] = []

    for method in methods:
        try:
            if method == "paddleocr":
                parser = PaddleOCRParser()
                if not parser.is_available():
                    skip_reasons.append("paddleocr: 未安装 (uv sync --extra paddleocr)")
                    continue
                logger.info(f"Using PaddleOCR for {len(pages)} pages")
                markdown = await parser.parse_pages(pages)
                if markdown.strip():
                    return markdown, "paddleocr"
                skip_reasons.append("paddleocr: 未识别到文字")

            elif method == "vlm_api":
                logger.info(f"Using VLM API for {len(pages)} pages")
                vlm = VLMParser()
                results = []
                for page in pages:
                    md = await vlm.parse_page_image(page.image, page.page_num)
                    results.append(f"<!-- page {page.page_num} -->\n\n{md}")
                markdown = "\n\n".join(results)
                if markdown.strip():
                    return markdown, "vlm_api"
                skip_reasons.append("vlm_api: 返回空结果，请检查 OPENAI_API_KEY 和 LLM_MODEL")

            elif method == "mineru":
                mineru = MinerUParser()
                if not mineru.is_available():
                    skip_reasons.append("mineru: 未安装 (uv sync --extra mineru && mineru-models-download)")
                    continue
                logger.info("Using MinerU for PDF parsing")
                markdown = await mineru.parse(pdf_path)
                if markdown.strip():
                    return markdown, "mineru"
                skip_reasons.append("mineru: 返回空结果")

            else:
                skip_reasons.append(f"{method}: 未知方法，可用: paddleocr, vlm_api, mineru")

        except Exception as e:
            skip_reasons.append(f"{method}: {e}")

    msg = "所有解析方法均失败：\n" + "\n".join(f"  - {r}" for r in skip_reasons)
    raise RuntimeError(msg)
