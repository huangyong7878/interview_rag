"""Query analyzer: intent classification and keyword extraction."""

import re

from src.models import QueryType


# Keywords suggesting a table query
TABLE_KEYWORDS = [
    "表", "表格", "列出", "尺寸", "规格", "参数", "数值",
    "多少", "宽度", "高度", "长度", "直径", "厚度",
    "list", "table", "dimension", "size", "spec",
]

# Keywords suggesting an unrelated/out-of-scope question
UNRELATED_KEYWORDS = [
    "天气", "新闻", "股票", "游戏", "音乐",
]

# Patterns that clearly indicate questions outside the document scope
OUT_OF_SCOPE_PATTERNS = [
    r'(起草|编写|作者)(人|者)?(是|为)谁',
    r'谁(起草|编写|制定)',
    r'(哪年|何时|什么时候)(发布|制定|实施)',
]


def extract_clause_numbers(query: str) -> list[str]:
    """Extract clause numbers from query text.

    Examples:
        "条款3.1的内容" -> ["3.1"]
        "3.1和4.1.1" -> ["3.1", "4.1.1"]
    """
    # Match patterns like "3.1", "4.1.1", but not dates like "2008"
    matches = re.findall(r'(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)', query)
    return matches


def extract_section_numbers(query: str) -> list[str]:
    """Extract top-level section numbers (e.g., "第3条" -> "3")."""
    matches = re.findall(r'第\s*(\d+)\s*[条款章节]', query)
    return matches


def classify_query(query: str) -> QueryType:
    """Classify the query type based on keywords and patterns."""
    query_lower = query.lower()

    # Check unrelated first
    for kw in UNRELATED_KEYWORDS:
        if kw in query_lower:
            return QueryType.UNRELATED

    # Check table-related
    table_signals = sum(1 for kw in TABLE_KEYWORDS if kw in query_lower)
    if table_signals >= 2:
        return QueryType.TABLE_QUERY

    # Check clause lookup
    if extract_clause_numbers(query) or extract_section_numbers(query):
        return QueryType.CLAUSE_LOOKUP

    # Default: factual query
    return QueryType.FACTUAL


def is_likely_out_of_scope(query: str, known_standard: str | None = None) -> bool:
    """Check if the question is likely outside the document's scope.

    NOTE: This function is currently NOT used in the query pipeline (the
    pre-check was removed from main.py because regex-based pre-filtering
    caused too many false positives). The pipeline now relies on the
    retrieval + generation + validation stages to handle out-of-scope
    detection naturally. This function is retained as a utility for
    future use or external callers.

    Args:
        query: The user's question.
        known_standard: The standard number of the loaded document (e.g. "GB/T 1568-2008").
                        References to this standard are not treated as cross-standard queries.
    """
    query_lower = query.lower()

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    # Check for cross-standard references
    cross_ref = re.search(
        r'(ISO|GB/T?\s*\d+(?:\.\d+)?(?:-\d+)?|ASTM|DIN|JIS)\s*\d*',
        query, re.IGNORECASE,
    )
    if cross_ref:
        matched = cross_ref.group(0).strip()
        # If the matched reference is the loaded document itself, it's not cross-standard
        if known_standard and known_standard.replace(" ", "") in matched.replace(" ", ""):
            return False
        return True

    return False
