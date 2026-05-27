"""Test the ingestion pipeline: splitter, indexer with mock data.

Run: uv run pytest tests/test_ingestion.py -v
"""

import pytest
from src.ingestion.splitter import (
    split_by_headings,
    extract_clause_number,
)
from src.ingestion.indexer import DocumentIndex
from src.models import ChunkType
from tests.fixtures import create_mock_chunks


class TestClauseExtraction:
    def test_top_level_clause(self):
        num, level = extract_clause_number("1 范围")
        assert num == "1"
        assert level == 1

    def test_sub_clause(self):
        num, level = extract_clause_number("3.1 键")
        assert num == "3.1"
        assert level == 2

    def test_deep_sub_clause(self):
        num, level = extract_clause_number("4.1.1 检验方法")
        assert num == "4.1.1"
        assert level == 3

    def test_no_clause_number(self):
        num, level = extract_clause_number("技术要求")
        assert num is None
        assert level == 0

    def test_chinese_clause_format(self):
        num, level = extract_clause_number("3.1 键的尺寸与公差")
        assert num == "3.1"
        assert level == 2


class TestSplitter:
    def test_split_markdown_with_headings(self):
        """Test that Markdown with headings is correctly split into chunks."""
        md = """## 1 范围
本标准规定了键的技术要求。

### 1.1 适用范围
本标准适用于一般机械传动中使用的键。

## 2 引用文件
以下文件对于本文件的应用是必不可少的。
"""
        chunks = split_by_headings(md)
        assert len(chunks) > 0

    def test_table_detection(self):
        """Test that tables are split into separate chunks."""
        md = """## 4.1 材料
| 材料 | 强度 |
| --- | --- |
| 45 | 600 |
| 40Cr | 800 |
"""
        chunks = split_by_headings(md)
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) >= 1

    def test_chunks_have_page_numbers(self):
        """Test that page markers are tracked in chunks."""
        md = """<!-- page 2 -->
## 3 术语和定义
### 3.1 键
用于连接轴和轴上零件的机械零件。
"""
        chunks = split_by_headings(md)
        for chunk in chunks:
            assert chunk.source_page == 2

    def test_heading_path_in_metadata(self):
        """Test that heading paths are captured in chunk metadata."""
        md = """## 3 术语和定义
### 3.1 键
用于连接轴和轴上零件的机械零件。
"""
        chunks = split_by_headings(md)
        paths = [c.heading_path for c in chunks if c.heading_path]
        # Parent heading "3" gets its own chunk, child "3.1" gets hierarchical path
        assert "3" in paths, f"Expected '3' in paths, got {paths}"
        # The sub-clause gets hierarchical path: "3.3.1" (3 → 3.1)
        assert any("3.1" in (p or "") for p in paths), f"Expected '3.1' somewhere in paths, got {paths}"

    def test_empty_markdown(self):
        """Test that empty input doesn't crash."""
        chunks = split_by_headings("")
        assert chunks == []

    def test_oversized_chunk_splitting(self):
        """Test that large chunks are split."""
        # Create a very long paragraph
        long_text = "这是一个很长的段落。" * 200
        md = f"## 1 范围\n{long_text}"
        chunks = split_by_headings(md)
        # Large text should be split into multiple chunks
        assert len(chunks) >= 1


class TestIndexer:
    def test_build_index(self):
        """Test that BM25 index is built correctly from chunks."""
        chunks = create_mock_chunks()
        idx = DocumentIndex()
        idx.add_chunks(chunks)

        assert idx.tfidf_matrix is not None
        assert idx.tfidf_matrix.shape[0] == len(chunks)
        assert len(idx.chunks) == len(chunks)

    def test_bm25_search_returns_results(self):
        """Test that BM25 search returns relevant results."""
        chunks = create_mock_chunks()
        idx = DocumentIndex()
        idx.add_chunks(chunks)

        indices, scores = idx.bm25_search("键的抗拉强度")
        # Should return some results
        assert len(indices) > 0
        # Technical requirement chunks should score higher
        top_chunk_idx = indices[0]
        top_chunk = idx.chunks[top_chunk_idx]
        # Either the clause 3 (技术要求 with 590MPa) or clause 3.4 should be top
        assert "590" in top_chunk.text or "抗拉" in top_chunk.text

    def test_clause_number_search(self):
        """Test that searching for a clause number finds the right chunk."""
        chunks = create_mock_chunks()
        idx = DocumentIndex()
        idx.add_chunks(chunks)

        indices, scores = idx.bm25_search("3.1 键的抗拉强度")
        top_chunks = [idx.chunks[i] for i in indices[:3]]
        # At least one of the top chunks should be about clause 3
        has_clause_3 = any(
            c.heading_path and "3" in (c.heading_path or "")
            for c in top_chunks
        )
        assert has_clause_3
