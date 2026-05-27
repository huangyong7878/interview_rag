"""Test fixtures and utility functions for the RAG system."""

import json
import os
from pathlib import Path
from typing import Optional

# Ensure we can import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_questions() -> list[dict]:
    """Load test questions from JSON file."""
    data_path = Path(__file__).parent.parent / "src" / "data" / "test_questions.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_mock_chunks() -> list:
    """Create mock document chunks for testing retrieval and generation.

    These simulate parsed Markdown from GB/T 1568-2008 键 技术条件.
    """
    from src.models import Chunk, ChunkType

    return [
        Chunk(
            id="chunk_001",
            text="## 1 范围\n\n本标准规定了除花键外的各种键的技术要求、验收检查、标志与包装。",
            source_page=3,
            heading_path="1",
            heading_level=1,
            heading_title="1 范围",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_002",
            text="## 2 规范性引用文件\n\nGB/T 1184 形状和位置公差未注公差值\nGB/T 2828.1-2003 计数抽样检验程序第1部分：按接收质量限(AQL)检索的逐批检验抽样计划\nGB/T 11334-2005 产品几何量技术规范(GPS)圆锥公差",
            source_page=3,
            heading_path="2",
            heading_level=1,
            heading_title="2 规范性引用文件",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_003",
            text="## 3 技术要求\n\n### 3.1\n键的抗拉强度应大于等于590 MPa。\n\n### 3.2\n键表面不允许有裂纹、浮锈、氧化皮和毛刺。\n\n### 3.3\nA、C型平键、楔键的圆弧部分不应有偏斜。",
            source_page=3,
            heading_path="3",
            heading_level=1,
            heading_title="3 技术要求",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_004",
            text="### 3.4\n半圆键的键长L两端允许倒成圆角，圆角半径r=0.5mm~1.5mm（小键取小值，大键取大值）。\n\n### 3.5\n普通平键、导向键和薄型平键，当键长L与键宽b之比大于等于8时，键宽b面在长度方向的平行度应按GB/T 1184的规定。\n\n### 3.6\n楔键斜度1：100的角度公差按GB/T 11334—2005中的AT8级选取。\n\n### 3.7\n在供、需双方同意的情况下，平键、楔键的半圆部分和半圆键的圆弧部分允许不倒角或倒圆，但需去毛刺。",
            source_page=3,
            heading_path="3.4",
            heading_level=2,
            heading_title="3.4 半圆键圆角",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_005",
            text="## 4 验收检查\n\n### 4.1 基本规则\n\n#### 4.1.1\n每个键都应当符合相应标准的全部规定，但在大批量生产中并不总是可能的。\n\n#### 4.1.2\n供方在生产过程中有权采用任何检查程序控制质量。\n\n#### 4.1.3\n需方认为必要或经济合理时，可根据供方的质量信誉对提交验收的产品免除检查。",
            source_page=3,
            heading_path="4",
            heading_level=1,
            heading_title="4 验收检查",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_006",
            text="### 4.2\n根据使用要求，由供、需双方协议，可对键进行抗拉强度试验。其抽样方案为：合格质量水平AQL为1.5，样本n为8，合格判定数为Ac。\n\n### 4.3 尺寸检查\n键的检查项目和合格质量水平见表1，样本大小按GB/T 2828.1—2003中一般检查水平Ⅱ，正常检查一次抽样方案抽取。\n\n### 4.4\n按GB/T 2828.1的规定，可根据供、需双方协议，对键的产品质量进行抽检。\n\n### 4.5\n从检查批中随机抽取样本，逐项进行检查并分项记录缺陷数量，如每项缺陷数均等于或小于相应的合格判定数(Ac)，则接收该批产品，否则拒收。",
            source_page=4,
            heading_path="4.2",
            heading_level=2,
            heading_title="4.2 抗拉强度试验",
            chunk_type=ChunkType.CLAUSE,
        ),
        Chunk(
            id="chunk_007",
            text="| 检查项目 | 普通平键 | 导向平键 | 薄型平键 | 半圆键 | 钩头楔键 |\n| --- | --- | --- | --- | --- | --- |\n| 键宽b | 1.0 | 1.0 | 1.5 | — | — |\n| 键高h | 2.5 | 2.5 | 2.5 | 2.5 | — |\n| 键长L | 4 | — | 4 | — | — |\n| 直径d | — | — | — | 2.5 | — |\n| 键宽平行度 | — | 1.5 | — | — | — |\n| 1:100斜度 | — | — | — | — | 1.5 |",
            source_page=4,
            heading_path="4.3",
            heading_level=2,
            heading_title="表1 合格质量水平AQL",
            chunk_type=ChunkType.TABLE,
        ),
        Chunk(
            id="chunk_008",
            text="## 5 标志与包装\n\n### 5.1\n包装箱、盒等外表面应有以下标志或标签：a)制造厂名；b)产品名称；c)产品标准规定的标记；d)产品数量或净重；e)制造或出厂日期。\n\n### 5.2\n产品在包装前应涂有防锈剂，以防止在运输和贮藏中受到腐蚀。在正常的运输和保管条件下，应保证自出厂之日起一年内不生锈。\n\n### 5.3\n产品的包装应保证产品不受损坏和便于使用，包装形式和方法由制造厂确定。",
            source_page=4,
            heading_path="5",
            heading_level=1,
            heading_title="5 标志与包装",
            chunk_type=ChunkType.CLAUSE,
        ),
    ]


def get_integration_test_url() -> str:
    """Get the base URL for integration tests."""
    return os.environ.get("TEST_BASE_URL", "http://localhost:8000")
