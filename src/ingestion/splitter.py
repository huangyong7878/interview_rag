"""Markdown AST-based clause-aware chunking.

Uses mistletoe to parse the Markdown AST and split into chunks
based on heading hierarchy. Handles tables as separate chunk types.
"""

import re
import uuid
from typing import Optional

import mistletoe
from mistletoe.block_token import (
    BlockToken,
    Heading,
    Paragraph,
    Table,
    CodeFence,
    List as MDList,
    Quote,
    HTMLBlock,
)
BlockQuote = Quote  # Alias for compatibility

from src.config import settings
from src.models import Chunk, ChunkType


def extract_clause_number(heading_text: str) -> tuple[Optional[str], int]:
    """Extract clause number and level from heading text.

    Examples:
        "1 范围" -> ("1", 1)
        "3.1 键的尺寸" -> ("3.1", 2)
        "4.1.1 检验方法" -> ("4.1.1", 3)
        "技术要求" -> (None, 0)
    """
    match = re.match(r'^(\d+(?:\.\d+)*)\s', heading_text.strip())
    if match:
        clause_num = match.group(1)
        level = clause_num.count('.') + 1
        return clause_num, level
    return None, 0


def render_children(token: BlockToken) -> str:
    """Render a token and its children back to plain Markdown text.

    This preserves the original formatting from the parser output.
    """
    if hasattr(token, 'children'):
        result = []
        for child in token.children:
            if isinstance(child, str):
                result.append(child)
            elif hasattr(child, 'content'):
                result.append(child.content)
            else:
                result.append(render_children(child))
        return ''.join(result)
    return getattr(token, 'content', str(token))


def render_block_to_text(token: BlockToken) -> str:
    """Render a block-level token to plain text, preserving structure."""
    parts = []

    if isinstance(token, Heading):
        if hasattr(token, 'children'):
            text = render_children(token)
            parts.append(f"{'#' * token.level} {text}")
    elif isinstance(token, Table):
        if hasattr(token, 'children'):
            parts.append(render_table(token))
    elif isinstance(token, Paragraph):
        if hasattr(token, 'children'):
            parts.append(render_children(token))
    elif isinstance(token, MDList):
        if hasattr(token, 'children'):
            parts.append(render_list(token))
    elif isinstance(token, CodeFence):
        if hasattr(token, 'children'):
            parts.append(render_children(token))
    elif isinstance(token, BlockQuote):
        if hasattr(token, 'children'):
            parts.append(render_children(token))
    elif isinstance(token, HTMLBlock):
        if hasattr(token, 'children'):
            parts.append(render_children(token))
    else:
        if hasattr(token, 'content'):
            parts.append(token.content)
        elif hasattr(token, 'children'):
            parts.append(render_children(token))

    return '\n'.join(parts)


def render_table(token: Table) -> str:
    """Render a table token as Markdown table string."""
    if not hasattr(token, 'children') or not token.children:
        return ""

    lines = []
    for i, row in enumerate(token.children):
        cells = []
        for cell in row.children if hasattr(row, 'children') else []:
            cells.append(render_children(cell).strip())
        lines.append('| ' + ' | '.join(cells) + ' |')
        # Add header separator after first row
        if i == 0:
            lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

    return '\n'.join(lines)


def render_list(token: MDList) -> str:
    """Render a list token as Markdown list."""
    lines = []
    for item in token.children if hasattr(token, 'children') else []:
        lines.append(f"- {render_children(item)}")
    return '\n'.join(lines)


def parse_page_from_markdown(markdown_text: str) -> int:
    """Extract page number from page separator comment."""
    match = re.search(r'<!-- page (\d+) -->', markdown_text)
    return int(match.group(1)) if match else 0


def _normalize_headings(markdown_text: str) -> str:
    """Preprocess plain-text section numbers into Markdown headings.

    PaddleOCR and other plain-text parsers output numbered sections like
    '1 范围' or '3.1 技术要求' without '##' prefixes. This converts them
    so the Markdown AST splitter can detect heading boundaries and
    build proper clause paths.

    NOTE: This is a workaround for parsers that don't output structured
    Markdown. It recovers clause structure but cannot fix table formatting.
    When using VLM API or MinerU, this step is a no-op (headings already exist).

    Only converts lines where the number is followed by text that starts
    with a Chinese character or ASCII letter (avoiding dates, table data, etc.).
    Also tolerates leading noise characters (e.g. OCR artifacts like '.5.1').
    """
    lines = markdown_text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # Match "1 范围", "3.1xxx", "4.1.1每个键..." — space is optional
        # Allow leading noise like OCR dots: ".5.1", "一5.2" etc.
        m = re.match(r'^\W*(\d+(?:\.\d+)*)\s{0,3}(\S.*)', stripped)
        if m:
            num = m.group(1)
            rest = m.group(2)
            if rest and (rest[0].isascii() and rest[0].isalpha()
                         or '一' <= rest[0] <= '鿿'
                         or '　' <= rest[0] <= '〿'):
                depth = num.count('.')
                result.append(f"{'#' * (depth + 1)} {num} {rest}")
                continue
        result.append(line)
    return '\n'.join(result)


def split_by_headings(markdown_text: str) -> list[Chunk]:
    """Parse Markdown AST and split into clause-aware chunks.

    Strategy:
    - Each heading starts a new chunk boundary
    - Content between headings is grouped under the preceding heading
    - Tables are extracted as independent chunks
    - Page separators (HTML comments) track page numbers
    """
    # Normalize plain-text section numbers to Markdown headings (for PaddleOCR etc.)
    markdown_text = _normalize_headings(markdown_text)

    try:
        doc = mistletoe.Document(markdown_text)
    except Exception:
        # If mistletoe can't parse, fall back to simple paragraph splitting
        return _fallback_split(markdown_text)

    chunks: list[Chunk] = []
    current_page = 0
    heading_stack: list[tuple[str, int]] = []  # [(clause_num, level), ...]

    # Buffer for content under the current heading
    current_heading: Optional[str] = None
    current_level: Optional[int] = None
    current_heading_title: Optional[str] = None
    content_buffer: list[str] = []

    def flush_buffer():
        nonlocal content_buffer
        if not content_buffer:
            return

        text = '\n\n'.join(content_buffer).strip()
        if not text:
            content_buffer = []
            return

        # heading_path = the deepest (most specific) clause number.
        # Clause numbers like "3.1" already encode parent hierarchy,
        # so there's no need to join stack entries with dots.
        heading_path = heading_stack[-1][0] if heading_stack else None

        chunk = Chunk(
            id=f"chunk_{uuid.uuid4().hex[:8]}",
            text=text,
            source_page=current_page or 1,
            heading_path=heading_path,
            heading_level=current_level,
            heading_title=current_heading_title,
            chunk_type=ChunkType.CLAUSE if heading_path else ChunkType.PARAGRAPH,
        )
        chunks.append(chunk)
        content_buffer = []

    # Walk the AST
    if not hasattr(doc, 'children'):
        return _fallback_split(markdown_text)

    for token in doc.children:
        if not isinstance(token, BlockToken):
            continue

        # Track page from HTML comments (both HTMLBlock and Paragraph with <!-- -->)
        page_marker_text = ""
        if isinstance(token, HTMLBlock):
            page_marker_text = getattr(token, 'content', '')
        elif isinstance(token, Paragraph):
            # mistletoe 1.5+ may wrap HTML comments in Paragraph > RawText
            raw = render_children(token).strip()
            if raw.startswith('<!--') and raw.endswith('-->'):
                page_marker_text = raw

        if page_marker_text:
            page = parse_page_from_markdown(page_marker_text)
            if page:
                current_page = page
            continue

        # Heading: new chunk boundary
        if isinstance(token, Heading):
            flush_buffer()

            heading_text = render_children(token)
            clause_num, level = extract_clause_number(heading_text)

            if clause_num:
                # Pop heading_stack to match current level
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()
                heading_stack.append((clause_num, level))

            current_heading = clause_num
            current_level = level
            current_heading_title = heading_text.strip()

            # Include the heading text in the chunk content
            content_buffer.append(f"{'#' * token.level} {heading_text}")
            continue

        # Table: extract as independent chunk
        if isinstance(token, Table):
            flush_buffer()
            table_md = render_table(token)
            if table_md.strip():
                heading_path = heading_stack[-1][0] if heading_stack else None

                chunks.append(Chunk(
                    id=f"chunk_{uuid.uuid4().hex[:8]}",
                    text=table_md,
                    source_page=current_page or 1,
                    heading_path=heading_path,
                    heading_level=current_level,
                    heading_title=current_heading_title,
                    chunk_type=ChunkType.TABLE,
                ))
            continue

        # Regular content: add to buffer
        text = render_block_to_text(token)
        if text.strip():
            content_buffer.append(text)

    # Flush remaining content
    flush_buffer()

    # If no chunks were created (empty doc), fall back
    if not chunks:
        return _fallback_split(markdown_text)

    # Split oversized chunks
    return _split_large_chunks(chunks)


def _split_large_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Split chunks exceeding max_chunk_size by paragraph boundaries."""
    max_size = settings.max_chunk_size
    result = []

    for chunk in chunks:
        if len(chunk.text) <= max_size:
            result.append(chunk)
            continue

        # Split by double newline (paragraph boundary)
        paragraphs = chunk.text.split('\n\n')
        current_parts = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_size and current_parts:
                result.append(chunk.model_copy(update={
                    "text": '\n\n'.join(current_parts),
                    "id": f"chunk_{uuid.uuid4().hex[:8]}",
                }))
                current_parts = [para]
                current_len = len(para)
            else:
                current_parts.append(para)
                current_len += len(para)

        if current_parts:
            result.append(chunk.model_copy(update={
                "text": '\n\n'.join(current_parts),
                "id": f"chunk_{uuid.uuid4().hex[:8]}",
            }))

    return result


def _fallback_split(markdown_text: str) -> list[Chunk]:
    """Fallback: split by double newlines when AST parsing fails."""
    chunks = []
    paragraphs = markdown_text.split('\n\n')
    page = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check for page marker
        p = parse_page_from_markdown(para)
        if p:
            page = p
            continue

        chunks.append(Chunk(
            id=f"chunk_{uuid.uuid4().hex[:8]}",
            text=para,
            source_page=page or 1,
            chunk_type=ChunkType.PARAGRAPH,
        ))

    return _split_large_chunks(chunks)
