---
title: RAG 技术详解：让 AI Agent 拥有无限知识
slug: rag-retrieval-augmented-generation
authors: [default-author]
tags: [大语言模型, 自然语言处理]
date: 2026-03-05
---

检索增强生成（RAG）是解决 LLM 知识局限性的关键技术。通过在生成前检索相关文档，RAG 让 Agent 能够基于最新、最准确的信息回答问题。本文将深入介绍 RAG 的架构、实现和优化策略。

<!-- truncate -->

## 为什么需要 RAG？

LLM 存在几个固有的局限性：

- **知识截止**：训练数据有截止日期，无法获取最新信息
- **幻觉问题**：可能生成看似合理但实际错误的内容
- **领域知识不足**：对特定领域的专业知识覆盖有限
- **无法访问私有数据**：企业内部文档、个人笔记等

RAG 通过"先检索，再生成"的方式解决这些问题。

## RAG 的基本架构

```
用户提问
    │
    ▼
┌──────────┐     ┌──────────────┐
│ 查询编码  │────▶│  向量数据库   │
│ Encoder  │     │ Vector Store │
└──────────┘     └──────┬───────┘
                        │ 检索 Top-K
                        ▼
                ┌──────────────┐
                │  相关文档片段  │
                └──────┬───────┘
                       │
    用户提问 ──────────▶│
                       ▼
                ┌──────────────┐
                │    LLM 生成   │
                └──────┬───────┘
                       │
                       ▼
                   最终回答
```

## 文档处理与分块

RAG 的第一步是将文档处理成适合检索的片段：

```python
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """文档片段"""
    content: str
    metadata: dict[str, str]
    chunk_id: str


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """将文本按固定大小分块，支持重叠"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句号处断开，避免截断句子
        if end < len(text):
            last_period = chunk.rfind('。')
            if last_period > chunk_size * 0.5:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


# 示例
text = "人工智能是计算机科学的一个分支。" * 20
chunks = split_text(text, chunk_size=100, overlap=20)
print(f"原文长度: {len(text)}, 分块数: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"  块 {i}: {len(chunk)} 字符")
```

### 分块策略对比

| 策略 | 优点 | 缺点 |
|------|------|------|
| 固定大小 | 简单高效 | 可能截断语义 |
| 按句子分割 | 保持语义完整 | 块大小不均匀 |
| 按段落分割 | 语义连贯 | 段落可能过长 |
| 递归分割 | 灵活适应 | 实现复杂 |
| 语义分割 | 语义最优 | 计算成本高 |

## 向量嵌入与检索

将文本转换为向量，通过相似度搜索找到最相关的片段：

```python
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimpleVectorStore:
    """简单的向量存储（演示用）"""
    documents: list[str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)

    def add(self, text: str, vector: np.ndarray) -> None:
        """添加文档和对应向量"""
        self.documents.append(text)
        self.vectors.append(vector / np.linalg.norm(vector))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """余弦相似度搜索"""
        query_norm = query_vector / np.linalg.norm(query_vector)
        scores = [
            np.dot(query_norm, vec) for vec in self.vectors
        ]
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.documents[i], float(scores[i]))
            for i in indices
        ]


# 模拟使用
store = SimpleVectorStore()
dim = 384  # 嵌入维度

# 添加文档（实际中使用嵌入模型生成向量）
docs = [
    "Transformer 是一种基于注意力机制的神经网络架构",
    "RAG 通过检索增强来提升 LLM 的回答质量",
    "向量数据库用于存储和检索高维向量",
]
for doc in docs:
    vec = np.random.randn(dim)  # 实际中用嵌入模型
    store.add(doc, vec)

# 搜索
query_vec = np.random.randn(dim)
results = store.search(query_vec, top_k=2)
for doc, score in results:
    print(f"[{score:.3f}] {doc}")
```

## 构建 RAG 提示

将检索到的文档片段与用户问题组合成提示：

```python
def build_rag_prompt(
    question: str,
    contexts: list[str],
    max_context_length: int = 3000,
) -> str:
    """构建 RAG 提示"""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        addition = f"\n[文档 {i}]\n{ctx}\n"
        if len(context_text) + len(addition) > max_context_length:
            break
        context_text += addition

    return f"""基于以下参考文档回答用户的问题。如果文档中没有相关信息，请明确说明。

参考文档：
{context_text}

用户问题：{question}

回答："""


# 示例
prompt = build_rag_prompt(
    question="什么是 Transformer？",
    contexts=[
        "Transformer 是 Google 在 2017 年提出的神经网络架构...",
        "Transformer 使用自注意力机制处理序列数据...",
    ],
)
print(prompt)
```

## 高级 RAG 技术

### 查询改写（Query Rewriting）

用户的原始查询可能不够精确，通过 LLM 改写查询可以提升检索效果：

```python
def rewrite_queries(original_query: str) -> list[str]:
    """将一个查询改写为多个角度的查询（模拟）"""
    # 实际中由 LLM 生成
    return [
        original_query,
        f"{original_query} 的定义和概念",
        f"{original_query} 的应用场景和实例",
        f"{original_query} 的技术原理",
    ]


queries = rewrite_queries("RAG 技术")
for q in queries:
    print(f"  - {q}")
```

### 重排序（Reranking）

初步检索后，使用更精确的模型对结果重新排序：

```python
def rerank(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """重排序（简化版）"""
    # 实际中使用 Cross-Encoder 模型
    scored = []
    for doc in documents:
        # 简单的关键词匹配评分（演示用）
        score = sum(
            1 for word in query
            if word in doc
        ) / len(query)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

## 常用向量数据库

| 数据库 | 类型 | 特点 |
|--------|------|------|
| Chroma | 嵌入式 | 轻量、易上手 |
| Milvus | 分布式 | 高性能、可扩展 |
| Pinecone | 云服务 | 全托管、免运维 |
| Weaviate | 混合搜索 | 支持向量+关键词 |
| pgvector | PostgreSQL 扩展 | 与现有数据库集成 |

## 评估 RAG 系统

RAG 系统的评估需要关注两个维度：

- **检索质量**：召回率、精确率、MRR（平均倒数排名）
- **生成质量**：忠实度（是否基于检索内容）、相关性、完整性

## 总结

RAG 是让 AI Agent 突破知识边界的关键技术。通过合理的文档分块、高效的向量检索和精心设计的提示模板，RAG 系统能够显著提升 LLM 回答的准确性和时效性。下一篇我们将探讨多 Agent 协作系统的设计。
