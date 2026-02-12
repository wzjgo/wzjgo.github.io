---
title: Transformer 架构详解：从注意力机制到大语言模型
slug: intro-to-transformer
authors: [default-author]
tags: [深度学习, 自然语言处理, 大语言模型, 机器学习]
date: 2026-01-10
---

Transformer 架构自 2017 年被提出以来，彻底改变了自然语言处理领域的格局。从 BERT 到 GPT，再到如今的各类大语言模型，Transformer 已经成为现代 AI 系统的核心基础架构。本文将带你深入理解 Transformer 的核心原理与实现方式。

<!-- truncate -->

## 什么是 Transformer？

Transformer 是 Google 在论文《Attention Is All You Need》中提出的一种全新的序列到序列模型架构。与传统的 RNN 和 LSTM 不同，Transformer 完全基于**注意力机制**（Attention Mechanism），摒弃了循环结构，从而实现了高效的并行计算。

![Transformer 架构示意图](/img/transformer-architecture.svg)

Transformer 的核心优势包括：

- **并行计算能力**：不依赖序列顺序，可以同时处理所有位置的输入
- **长距离依赖建模**：通过自注意力机制直接建立任意两个位置之间的关系
- **可扩展性强**：通过增加层数和参数量，模型性能可以持续提升

## 自注意力机制

自注意力（Self-Attention）是 Transformer 的核心组件。它通过计算输入序列中每个位置与其他所有位置的相关性，来生成上下文感知的表示。

具体来说，对于输入序列中的每个 token，自注意力机制会计算三个向量：**Query（查询）**、**Key（键）** 和 **Value（值）**。注意力权重通过 Query 和 Key 的点积计算得到，然后用这些权重对 Value 进行加权求和。

下面是一个使用 PyTorch 实现自注意力机制的简单示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """简单的自注意力机制实现"""

    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 计算 Q、K、V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算注意力分数
        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 应用 softmax 获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, v)
        return self.out(output)


# 使用示例
embed_dim = 512
seq_len = 10
batch_size = 2

model = SelfAttention(embed_dim=embed_dim)
x = torch.randn(batch_size, seq_len, embed_dim)
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
```

## 多头注意力

在实际应用中，Transformer 使用**多头注意力**（Multi-Head Attention）来让模型同时关注不同子空间的信息。每个"头"独立地执行注意力计算，最后将结果拼接起来。

```python
# 多头注意力的核心思想
num_heads = 8
head_dim = embed_dim // num_heads  # 每个头的维度

# 将 Q、K、V 拆分为多个头
q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
# 形状变为: (batch_size, num_heads, seq_len, head_dim)
```

## 位置编码

由于 Transformer 没有循环结构，它无法感知输入序列中 token 的位置信息。因此需要引入**位置编码**（Positional Encoding）来注入位置信息。

原始论文使用正弦和余弦函数生成位置编码：

```python
import numpy as np


def positional_encoding(seq_len: int, embed_dim: int) -> np.ndarray:
    """生成正弦位置编码"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
    )

    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# 生成位置编码
pe = positional_encoding(seq_len=50, embed_dim=512)
print(f"位置编码形状: {pe.shape}")
```

## Transformer 的应用场景

Transformer 架构已经在多个 AI 领域取得了突破性进展：

| 领域 | 代表模型 | 应用场景 |
|------|----------|----------|
| 自然语言处理 | GPT、BERT | 文本生成、情感分析、机器翻译 |
| 计算机视觉 | ViT、DETR | 图像分类、目标检测 |
| 多模态学习 | CLIP、DALL-E | 图文匹配、图像生成 |
| 语音处理 | Whisper | 语音识别、语音翻译 |

## 注意力公式

Transformer 中最核心的计算可以用一个简洁的数学公式来表达。对于给定的查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$，注意力的计算方式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $d_k$ 是键向量的维度，除以 $\sqrt{d_k}$ 是为了防止点积值过大导致 softmax 梯度消失。

多头注意力则将输入投影到 $h$ 个不同的子空间中分别计算注意力，然后拼接结果：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

其中每个头 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

## 总结

Transformer 架构通过自注意力机制实现了对序列数据的高效建模，其优秀的并行计算能力和可扩展性使其成为当前深度学习领域最重要的基础架构之一。理解 Transformer 的工作原理，对于深入学习大语言模型和其他前沿 AI 技术至关重要。

在后续的文章中，我们将进一步探讨基于 Transformer 的各类预训练模型，以及如何在实际项目中应用这些技术。
