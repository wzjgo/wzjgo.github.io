---
title: 计算机视觉：从 CNN 到 Vision Transformer
slug: computer-vision-cnn-to-vit
authors: [default-author]
tags: [计算机视觉, 深度学习]
date: 2026-02-05
---

计算机视觉是让机器"看懂"图像和视频的技术。从经典的卷积神经网络到最新的 Vision Transformer，本文将带你了解计算机视觉领域的核心技术和发展趋势。

<!-- truncate -->

## 卷积神经网络（CNN）

CNN 是计算机视觉的基石。它通过卷积层自动提取图像的层次化特征：浅层提取边缘和纹理，深层提取更抽象的语义特征。

### 卷积操作

卷积核（Filter）在输入图像上滑动，计算局部区域的加权和，生成特征图（Feature Map）。

```python
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """简单的 CNN 图像分类器"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = SimpleCNN(num_classes=10)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")
```

## 经典 CNN 架构演进

| 模型 | 年份 | 层数 | 关键创新 |
|------|------|------|----------|
| AlexNet | 2012 | 8 | ReLU、Dropout、GPU 训练 |
| VGGNet | 2014 | 16/19 | 小卷积核堆叠 |
| GoogLeNet | 2014 | 22 | Inception 模块 |
| ResNet | 2015 | 152 | 残差连接 |
| EfficientNet | 2019 | - | 复合缩放策略 |

### ResNet 的残差连接

ResNet 引入了跳跃连接（Skip Connection），解决了深层网络的梯度消失问题：

```python
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        return self.relu(out)
```

## Vision Transformer（ViT）

2020 年，Google 提出了 Vision Transformer，将 Transformer 架构直接应用于图像分类，证明了纯注意力模型在视觉任务上也能取得优异性能。

ViT 的核心思想：
1. 将图像分割为固定大小的 patch（如 16×16）
2. 将每个 patch 线性投影为一个向量（类似 NLP 中的 token）
3. 加上位置编码后送入标准 Transformer Encoder

```python
class PatchEmbedding(nn.Module):
    """将图像分割为 patch 并嵌入"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # 展平为序列: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


patch_embed = PatchEmbedding()
img = torch.randn(1, 3, 224, 224)
patches = patch_embed(img)
print(f"图像 {img.shape} -> {patches.shape[1]} 个 patch, 维度 {patches.shape[2]}")
```

## 目标检测与分割

计算机视觉不仅仅是分类，还包括：

- **目标检测**：定位图像中的物体并分类（YOLO、DETR）
- **语义分割**：对每个像素进行分类（U-Net、SegFormer）
- **实例分割**：区分同类的不同实例（Mask R-CNN）

## 总结

计算机视觉从 CNN 时代走向了 Transformer 时代，但 CNN 并未被完全取代。在实际应用中，CNN 和 Transformer 各有优势，混合架构也在不断涌现。理解这些基础架构，对于解决实际的视觉问题至关重要。
