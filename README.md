# 智能文档问答 Agent — RAG 原型系统

## 概述

一个基于 RAG（检索增强生成）的智能文档问答系统，专门处理**扫描版 PDF**，支持中文标准文档的理解、检索、问答与自检。

**核心能力：**
- 扫描 PDF 自动识别 → PaddleOCR/VLM/MinerU 生成结构化 Markdown
- 条款感知分块（基于 Markdown AST Heading 层级，支持纯文本自动补全标题标记）
- BM25 + FAISS **宽召回（top_k=20）**，条款编号精确匹配加权
- **LLM Reranker** 批量精排，一次 LLM 调用完成全部打分
- LLM 生成答案 + 来源引用 + **三阶段自检**

## 架构设计

```
扫描 PDF
   │
   ▼
Loader (PyMuPDF) ── 提取页面图像，判断 PDF 类型（扫描/文字型）
   │
   ▼
Parser (PaddleOCR→VLM→MinerU) ── 三级策略链，按优先级回退
   │
   ▼
Splitter (Markdown AST) ── 条款感知分块（自动补全纯文本标题标记）
   │
   ▼
Indexer (BM25 + FAISS) ── 双索引建立（FAISS 可选，通过 ENABLE_FAISS 控制）
   │
   ▼
[Query] → Retriever (宽召回 top_k=20) → Reranker (LLM 批量精排) → Generator (LLM 生成) → Validator (三阶段自检)
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| PDF→Markdown | PaddleOCR 优先（免费），VLM/MinerU 高质回退 | 日常低延迟零成本，必要时切换高质量解析 |
| 分块策略 | Markdown AST Heading + 纯文本自动补全 | PaddleOCR 不输出 `##` 标记时自动将"1 范围"转为标题，保证条款结构 |
| 召回策略 | 宽召回(top_k=20) + LLM 批量精排 | 粗筛不遗漏，一次 LLM 调用完成精排，避免盲打分 |
| 检索方式 | BM25 字符级 n-gram + FAISS 向量（可选） | 字符级规避中文分词依赖，FAISS 可选配，关闭时靠 Reranker 弥补语义 |
| 自检机制 | 三阶段自检 + 文本重叠预检 | LLM-as-judge 存在随机误判，文本重叠预检可跳过 LLM 减少假阴性 |

### Parser 策略链

系统采用三级 Parser 链，按 `.env` 中 `PARSER_PRIORITY` 配置的顺序尝试，任一成功即返回：

| Parser | 速度 | 成本 | 表格/结构 | 适用场景 |
|--------|------|------|-----------|---------|
| **PaddleOCR** | 快 | 免费 | 无（纯文本） | 简单文本提取，日常开发测试 |
| **VLM API** | 中 | API 按量付费 | 结构化 Markdown+表格 | 需要表格和标题结构时 |
| **MinerU** | 慢(3-10min) | 免费（本地） | 最高质量 | 复杂版面/公式，生产交付 |

**当前默认**：`PARSER_PRIORITY=paddleocr`（快速迭代）。需要表格/结构时改为 `vlm_api`。

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) 包管理器
- [LibreOffice](https://www.libreoffice.org/)（MinerU 依赖）

### 安装

```bash
# 1. 克隆代码
cd interview_rag

# 2. 安装依赖（包含 PaddleOCR + FlagEmbedding）
uv sync

# 3. MinerU（可选，更高质量的 PDF 解析）
brew install libreoffice
uv sync --extra mineru
mineru-models-download

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API 配置：
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   OPENAI_API_KEY=sk-xxx
#   LLM_MODEL=gpt-4o
#   ENABLE_FAISS=false   # false=BM25 only, true=BM25+向量（需 FlagEmbedding）
```

### 启动服务

```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger API 文档。

### API 使用

**1. 上传并解析文档**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@GBT 1568-2008 键 技术条件.pdf"
```

返回：
```json
{
  "status": "ok",
  "page_count": 4,
  "chunk_count": 24,
  "ocr_method": "paddleocr",
  "markdown_preview": "## 1 范围\n本标准规定了..."
}
```

**2. 提问**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "GB/T 1568-2008的适用范围是什么？"}'
```

返回：
```json
{
  "question": "GB/T 1568-2008的适用范围是什么？",
  "result": {
    "answer": "本标准规定了除花键外的各种键的技术要求、验收检查、标志与包装。[来源: 条款1, 第3页]",
    "citations": [{"clause": "1", "page": 3, "snippet": "# 1 范围\n\n本标准规定了..."}],
    "grounding_score": 0.8,
    "claim_checks": [...],
    "citation_errors": [],
    "should_refuse": false
  },
  "retrieved_chunks": [...]
}
```

### Result 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | `str` | LLM 生成的答案，含 `[来源: 条款X, 第Y页]` 引用标记 |
| `citations` | `list` | 从答案提取的引用列表，每条含 `clause`、`page`、`snippet` |
| `grounding_score` | `float(0–1)` | 答案接地分数，SUPPORTED=1.0、PARTIAL=0.5、UNSUPPORTED=0.0，≥0.5 为通过 |
| `claim_checks` | `list` | 每项 `{claim, verdict(supported/partial/unsupported), reason}` |
| `citation_errors` | `list[str]` | 引用校验问题，如"引用的条款 X 不存在" |
| `should_refuse` | `bool` | 是否建议拒答，当 grounding_score<0.5 时为 true |
| `refuse_reason` | `str\|null` | 拒答原因，should_refuse=false 时为 null |
| `retrieved_chunks` | `list` | 检索到的文档片段（调试用），含 chunk_id、条款号、页码、分数、文本预览 |

## 测试

### 单元测试（无需 API/LLM）

```bash
uv run pytest tests/test_ingestion.py tests/test_retrieval.py tests/test_agent.py -v
```

测试覆盖：
- 条款编号提取（1, 3.1, 4.1.1 等格式）
- Markdown AST 分块（Heading 层级、表格检测、页码追踪）
- 查询意图分类（条款查询/表格查询/事实查询/无关问题）
- 引用格式解析与校验逻辑
- 声明提取与接地评分计算
- BM25 检索准确性（给定已知文本，验证召回是否命中目标条款）

### 演示录屏

`interview.mp4` 是从服务启动、文档上传、到逐条提问的完整操作录屏。
### 端到端测试

启动服务后运行：

```bash
# 先 ingest 文档
curl -X POST http://localhost:8000/ingest \
  -F "file=@GBT 1568-2008 键 技术条件.pdf"

# 批量测试（逐条提问，中间间隔 1 秒避免服务器过载）
uv run python -c "
from tests.fixtures import load_test_questions
import httpx, json, time

for tc in load_test_questions():
    resp = httpx.post('http://localhost:8000/query', json={'question': tc['question']}, timeout=180)
    result = resp.json()['result']
    passed = (
        (tc['expected_type'] == 'refusal' and result['should_refuse']) or
        (tc['expected_type'] == 'answer' and not result['should_refuse'] and
         result['grounding_score'] >= tc.get('min_grounding_score', 0.4))
    )
    print(f'[{\"PASS\" if passed else \"FAIL\"}] {tc[\"id\"]}: gs={result[\"grounding_score\"]:.2f} | {tc[\"question\"][:40]}')
    time.sleep(1)
"
```

### 测试用例设计

| # | 类型 | 问题 | 验证点 |
|---|------|------|--------|
| 1 | 事实查询 | GB/T 1568-2008适用范围？ | 返回范围条款 + 来源 |
| 2 | 条款查询 | 半圆键圆角半径范围？ | 精确返回3.4的0.5mm~1.5mm |
| 3 | **表格查询** | 表1普通平键键宽b的AQL？ | 从Table 1提取1.0 |
| 4 | **无答案** | 键的疲劳寿命计算公式？ | 正确拒答（超出范围）|
| 5 | 跨条款 | 包装运输和防锈要求？ | 跨5.1–5.4合成 |

## Reranker 精排

检索管线采用"宽召回 + 精排"两阶段设计，解决混合打分（0.6×BM25 + 0.4×Vector）盲打分的问题：

```
Retriever 宽召回 (top_k=20)
        │
        ▼
Reranker (LLM 批量评分)
  将所有 chunk 合并为一次 LLM 调用，请求逐条评分 1-5
  1=无关  2=略相关  3=部分相关  4=相关  5=高度相关
        │
        ▼
按 LLM 评分排序，取 top_k=5
        │
        ▼
Generator 生成答案
```

**Reranker vs 无 Reranker 对比：**

| | 无 Reranker | 有 Reranker |
|---|---|---|
| 召回 | top_k=5，盲打分截断 | top_k=20，宽召回不遗漏 |
| 排序 | BM25+向量加权（无法理解 query-chunk 语义关系）| LLM 一次性读全部 chunk，批量判断实质相关性 |
| API 调用 | 0 | +1 次（批量子） |
| 多条款问题 | 只能命中一个强信号 chunk | 宽召回覆盖多个条款，精排挑出最相关的 |

## 自检机制说明

系统在三个环节进行质量把关：

1. **证据充分性（生成前）**：检索到的文档片段是否与问题相关？
   - 规则检查：所有 chunk 得分过低 → 直接拒答
   - LLM 评分：1-5 分评估，< 阈值 → 拒答

2. **声明接地（生成后）**：答案中的每个关键陈述能否被检索原文支持？
   - 拆分答案为原子声明
   - 先做文本重叠预检（>70% 字符重叠直接判支持，跳过 LLM）
   - LLM-as-judge 逐条判断：支持/部分支持/不支持
   - 计算 grounding_score = (支持数×1 + 部分支持数×0.5) / 总声明数

3. **引用校验（生成后）**：答案中 `[来源: 条款X, 第Y页]` 是否真实存在？
   - 提取所有引用标记
   - 与检索结果元数据逐一比对
   - 标记不存在的引用（幻觉引用）

## 业务场景扩展

系统核心模块均采用策略模式设计，可替换实现适配不同场景：

```python
# 金融报告场景
pipeline = RAGPipeline(
    parser=MultiFormatParser([PDFParser(), ExcelParser()]),  # 增加 Excel
    splitter=FinancialSectionChunker(),   # 按 BS/IS/CF 科目分段
    retriever=HybridRetriever(enable_range_search=True),  # 金额范围检索
    validator=FinancialValidator(cross_check_rules={...}), # 数字交叉校验
)
```

| 场景 | Parser 变化 | Splitter 变化 | Retriever 增强 | Validator 增强 |
|------|------------|-------------|---------------|---------------|
| **标准文档**（当前）| MinerU/VLM | Heading AST | 条款号加权 | 声明接地 |
| **金融报告** | +Excel/CSV Parser | 报表科目分段 | 数值范围检索 | 数字交叉校验 (BS=Assets-Liab) |
| **合同审查** | PDF Parser | 合同条款分段 | 当事人过滤 | 义务识别 (应/可/禁止) |
| **合规文件** | +Docx Parser | 法规条款分段 | 时效过滤 (生效日期) | 合规条款比对 |
| **产品手册** | 图像 Parser | 章节+表格 | 属性-值检索 | 参数一致性校验 |

## OCR 错误的处理策略

即使使用 MinerU/VLM，OCR 级别的字符错误仍可能发生。系统的分层容错设计：

1. **Parser 层**：VLM 的上下文理解能力天然纠正部分识别错误
2. **Splitter 层**：只要 Heading 标记（`##`, `###`）正确，条款结构就不会被破坏
3. **Retriever 层**：混合检索（BM25 + 向量）提供互补匹配，即使关键词有错别字，向量检索仍能保证召回
4. **Generator 层**：LLM 在生成时结合上下文推断正确语义
5. **Validator 层**：声明接地检查会标记出与原文不一致的声明

## 已知局限

1. **嵌入模型可能崩溃**：FlagEmbedding (torch) 在 macOS 上可能 segfault，设置 `ENABLE_FAISS=false` 可退化为纯 BM25 检索（Reranker 可部分弥补语义损失）
2. **表格重建精度**：PaddleOCR 输出扁平文本而非结构化表格，复杂表格可能丢失结构
3. **单文档设计**：当前设计未针对多文档/海量文档做优化（如分片索引、增量更新）
4. **Reranker 依赖 LLM**：批量评分增加约 1 次 LLM 调用（约 1-3 秒），对 4 页文档性价比有限
5. **自检依赖 LLM**：LLM-as-judge 本身可能有偏差，已加文本重叠预检缓解
6. **VLM 和 MinerU 未测试**：因时间限制，当前仅测试了 PaddleOCR 解析链路。VLM API 和 MinerU 的代码路径已实现但未经过完整的端到端验证

## 技术栈

| 层 | 技术 |
|---|---|
| 包管理 | uv |
| Web 框架 | FastAPI |
| PDF 处理 | PyMuPDF (fitz) |
| PDF→Markdown | MinerU (Docker) / VLM API |
| Markdown 解析 | mistletoe |
| 向量库 | FAISS |
| BM25 | scikit-learn TfidfVectorizer |
| LLM | OpenAI 兼容 API (可配置) |
