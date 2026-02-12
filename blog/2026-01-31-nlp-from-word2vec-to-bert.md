---
title: 自然语言处理：从 Word2Vec 到 BERT
slug: nlp-from-word2vec-to-bert
authors: [default-author]
tags: [自然语言处理, 深度学习, 机器学习]
date: 2026-01-31
---

自然语言处理（NLP）是 AI 中最活跃的研究领域之一。本文将梳理 NLP 的发展脉络，从词向量到预训练语言模型，带你了解这个领域的核心技术演进。

<!-- truncate -->

## 词向量：让计算机理解语义

在深度学习时代之前，NLP 主要依赖手工特征和统计方法。词向量（Word Embedding）的出现改变了这一局面，它将每个词映射为一个稠密的低维向量，使得语义相近的词在向量空间中距离更近。

### Word2Vec

Word2Vec 由 Google 在 2013 年提出，包含两种训练方式：

- **CBOW（Continuous Bag of Words）**：根据上下文预测中心词
- **Skip-gram**：根据中心词预测上下文

```python
from gensim.models import Word2Vec

# 示例语料
sentences = [
    ["深度", "学习", "是", "机器", "学习", "的", "子领域"],
    ["神经", "网络", "是", "深度", "学习", "的", "基础"],
    ["自然", "语言", "处理", "使用", "深度", "学习", "技术"],
    ["Transformer", "改变", "了", "自然", "语言", "处理"],
]

# 训练 Word2Vec 模型
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
)

# 查看词向量
vector = model.wv["深度"]
print(f"'深度' 的词向量维度: {vector.shape}")
```

### GloVe

GloVe（Global Vectors）由斯坦福大学提出，结合了全局统计信息和局部上下文窗口的优点，通过词共现矩阵的分解来学习词向量。

## 序列模型：RNN 与 LSTM

词向量解决了词的表示问题，但 NLP 任务通常需要理解整个句子或文档。循环神经网络（RNN）通过隐藏状态传递序列信息。

```python
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """基于 LSTM 的文本分类器"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        # 拼接前向和后向的最终隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)


model = TextClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=5,
)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 注意力机制的引入

RNN 在处理长序列时存在信息衰减问题。注意力机制允许模型在生成每个输出时，动态地关注输入序列的不同部分，大幅提升了机器翻译等任务的性能。

## BERT：预训练语言模型的里程碑

BERT（Bidirectional Encoder Representations from Transformers）由 Google 在 2018 年提出，开创了"预训练 + 微调"的范式。

BERT 的核心创新：

- **双向上下文**：同时利用左侧和右侧的上下文信息
- **掩码语言模型（MLM）**：随机遮盖输入中的部分 token，让模型预测被遮盖的词
- **下一句预测（NSP）**：判断两个句子是否连续

```python
from transformers import BertTokenizer, BertModel

# 加载预训练的 BERT 模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 编码文本
text = "深度学习改变了自然语言处理"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取句子表示
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"句子向量维度: {cls_embedding.shape}")
```

## NLP 技术发展时间线

| 年份 | 里程碑 | 意义 |
|------|--------|------|
| 2013 | Word2Vec | 高效词向量训练 |
| 2014 | GloVe | 全局词向量 |
| 2015 | Attention | 注意力机制用于机器翻译 |
| 2017 | Transformer | 全注意力架构 |
| 2018 | BERT | 预训练语言模型 |
| 2019 | GPT-2 | 大规模语言生成 |
| 2022 | ChatGPT | 对话式 AI |
| 2024 | GPT-4o | 多模态大模型 |

## 总结

NLP 从早期的规则方法，经历了统计方法、词向量、序列模型，到如今的预训练大模型，每一步都带来了质的飞跃。理解这个演进过程，有助于我们更好地把握当前大语言模型技术的本质。
