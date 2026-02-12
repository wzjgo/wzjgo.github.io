---
title: 深度学习基础：神经网络的原理与实践
slug: deep-learning-neural-networks
authors: [default-author]
tags: [深度学习, 机器学习]
date: 2026-01-24
---

深度学习是机器学习的一个子领域，通过多层神经网络来学习数据的层次化表示。本文将从感知机出发，逐步介绍神经网络的核心概念和训练方法。

<!-- truncate -->

## 从感知机到神经网络

感知机（Perceptron）是最简单的神经网络模型，由 Frank Rosenblatt 在 1957 年提出。它接收多个输入，通过加权求和和激活函数产生输出。

```python
import numpy as np


class Perceptron:
    """简单感知机实现"""

    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x: np.ndarray) -> int:
        linear_output = np.dot(x, self.weights) + self.bias
        return 1 if linear_output >= 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                update = self.lr * (yi - prediction)
                self.weights += update * xi
                self.bias += update
```

## 多层感知机（MLP）

单层感知机无法解决非线性问题（如 XOR），多层感知机通过引入隐藏层解决了这个限制。

```python
import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 创建模型
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 激活函数

激活函数为神经网络引入非线性，使其能够学习复杂的模式。

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出 (0,1)，容易梯度消失 |
| Tanh | $\tanh(x)$ | 输出 (-1,1)，零中心化 |
| ReLU | $\max(0, x)$ | 计算高效，可能出现死神经元 |
| GELU | $x \cdot \Phi(x)$ | Transformer 中常用 |

## 反向传播与梯度下降

神经网络通过反向传播算法计算损失函数对每个参数的梯度，然后使用梯度下降更新参数。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

# 定义模型、损失函数和优化器
model = MLP(input_dim=10, hidden_dim=64, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()       # 反向传播计算梯度
    optimizer.step()      # 更新参数

    if (epoch + 1) % 10 == 0:
        accuracy = (output.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
```

## 常见的深度学习架构

- **卷积神经网络（CNN）**：擅长处理图像数据，通过卷积操作提取空间特征
- **循环神经网络（RNN）**：处理序列数据，具有记忆能力
- **Transformer**：基于注意力机制，已成为 NLP 和 CV 的主流架构
- **生成对抗网络（GAN）**：通过生成器和判别器的对抗训练生成数据

## 防止过拟合

深度学习模型参数量大，容易过拟合。常用的正则化技术：

- **Dropout**：训练时随机丢弃部分神经元
- **权重衰减（L2 正则化）**：限制权重大小
- **数据增强**：扩充训练数据
- **早停（Early Stopping）**：验证集性能不再提升时停止训练
- **Batch Normalization**：标准化每层的输入

## 总结

深度学习通过多层神经网络实现了对复杂数据模式的自动学习。掌握神经网络的基本原理、训练方法和正则化技术，是进一步学习 CNN、RNN、Transformer 等高级架构的基础。
