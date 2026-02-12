---
title: 机器学习入门：从监督学习到无监督学习
slug: machine-learning-basics
authors: [default-author]
tags: [机器学习]
date: 2026-01-17
---

机器学习是人工智能的核心分支，它让计算机能够从数据中自动学习规律，而无需显式编程。本文将介绍机器学习的基本概念、主要类型和常见算法。

<!-- truncate -->

## 什么是机器学习？

机器学习（Machine Learning）是一种通过数据驱动的方法，让计算机自动发现数据中的模式和规律。与传统编程不同，机器学习不需要人工编写具体的规则，而是通过算法从大量数据中"学习"出这些规则。

## 机器学习的三大类型

### 1. 监督学习（Supervised Learning）

监督学习是最常见的机器学习类型。训练数据包含输入特征和对应的标签（正确答案），模型的目标是学习从输入到输出的映射关系。

常见算法：
- **线性回归**：用于连续值预测
- **逻辑回归**：用于二分类问题
- **决策树**：基于特征条件进行分裂
- **支持向量机（SVM）**：寻找最优分类超平面
- **随机森林**：多棵决策树的集成

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 生成示例数据
np.random.seed(42)
X = np.random.randn(1000, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### 2. 无监督学习（Unsupervised Learning）

无监督学习的训练数据没有标签，模型需要自行发现数据中的结构和模式。

常见算法：
- **K-Means 聚类**：将数据分成 K 个簇
- **DBSCAN**：基于密度的聚类
- **主成分分析（PCA）**：降维技术
- **自编码器**：神经网络降维

```python
from sklearn.cluster import KMeans
import numpy as np

# 生成聚类数据
np.random.seed(42)
cluster1 = np.random.randn(100, 2) + [2, 2]
cluster2 = np.random.randn(100, 2) + [-2, -2]
cluster3 = np.random.randn(100, 2) + [2, -2]
X = np.vstack([cluster1, cluster2, cluster3])

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"聚类中心:\n{kmeans.cluster_centers_}")
print(f"各簇样本数: {np.bincount(labels)}")
```

### 3. 强化学习（Reinforcement Learning）

强化学习通过智能体与环境的交互来学习最优策略。智能体根据当前状态选择动作，环境返回奖励信号，目标是最大化累积奖励。

核心概念：
- **状态（State）**：环境的当前情况
- **动作（Action）**：智能体可以执行的操作
- **奖励（Reward）**：环境对动作的反馈
- **策略（Policy）**：从状态到动作的映射

## 模型评估

选择合适的评估指标对于衡量模型性能至关重要：

| 任务类型 | 常用指标 | 说明 |
|----------|----------|------|
| 分类 | 准确率、精确率、召回率、F1 | 衡量分类正确性 |
| 回归 | MSE、MAE、R² | 衡量预测误差 |
| 聚类 | 轮廓系数、Calinski-Harabasz | 衡量聚类质量 |

## 总结

机器学习为我们提供了从数据中自动提取知识的强大工具。理解监督学习、无监督学习和强化学习的区别与适用场景，是深入学习 AI 技术的第一步。在后续文章中，我们将深入探讨深度学习的核心概念。
