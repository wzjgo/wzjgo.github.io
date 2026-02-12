---
title: Agent 记忆与状态管理：让 AI 拥有持久记忆
slug: agent-memory-and-state
authors: [default-author]
tags: [大语言模型, 机器学习]
date: 2026-03-19
---

记忆是智能的基础。AI Agent 需要有效的记忆系统来积累经验、维护上下文和个性化服务。本文将深入探讨 Agent 记忆系统的设计与实现。

<!-- truncate -->

## 记忆的类型

借鉴认知科学，Agent 的记忆可以分为以下几类：

| 记忆类型 | 说明 | 实现方式 |
|----------|------|----------|
| 感知记忆 | 当前输入的原始信息 | 当前 prompt |
| 工作记忆 | 当前任务的上下文 | 对话历史（上下文窗口） |
| 短期记忆 | 近期的交互记录 | 内存缓存 |
| 长期记忆 | 持久化的知识和经验 | 向量数据库 |
| 情景记忆 | 具体事件的记录 | 结构化日志 |
| 语义记忆 | 抽象的知识和概念 | 知识图谱 |

## 工作记忆：上下文窗口管理

LLM 的上下文窗口是最基本的"记忆"，但容量有限。需要策略来管理：

```python
from dataclasses import dataclass, field


@dataclass
class Message:
    """对话消息"""
    role: str
    content: str
    token_count: int = 0

    def __post_init__(self):
        # 简单估算 token 数（中文约 1 字 = 1.5 token）
        self.token_count = int(len(self.content) * 1.5)


class WorkingMemory:
    """工作记忆管理器"""

    def __init__(self, max_tokens: int = 4000):
        self.messages: list[Message] = []
        self.max_tokens = max_tokens
        self.system_message: Message | None = None

    def set_system(self, content: str) -> None:
        """设置系统消息（始终保留）"""
        self.system_message = Message("system", content)

    def add(self, role: str, content: str) -> None:
        """添加消息"""
        self.messages.append(Message(role, content))
        self._trim()

    def _trim(self) -> None:
        """裁剪超出容量的旧消息"""
        total = sum(m.token_count for m in self.messages)
        if self.system_message:
            total += self.system_message.token_count

        while total > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total -= removed.token_count

    def get_messages(self) -> list[dict[str, str]]:
        """获取当前上下文"""
        result = []
        if self.system_message:
            result.append({
                "role": self.system_message.role,
                "content": self.system_message.content,
            })
        for msg in self.messages:
            result.append({
                "role": msg.role,
                "content": msg.content,
            })
        return result

    @property
    def total_tokens(self) -> int:
        total = sum(m.token_count for m in self.messages)
        if self.system_message:
            total += self.system_message.token_count
        return total


# 使用示例
memory = WorkingMemory(max_tokens=200)
memory.set_system("你是一个 AI 助手")
memory.add("user", "什么是机器学习？")
memory.add("assistant", "机器学习是人工智能的一个分支...")
memory.add("user", "能举个例子吗？")
print(f"当前 token 数: {memory.total_tokens}")
print(f"消息数: {len(memory.get_messages())}")
```

## 对话摘要：压缩历史信息

当对话过长时，可以用 LLM 对历史对话进行摘要，保留关键信息：

```python
from dataclasses import dataclass, field


@dataclass
class SummarizedMemory:
    """带摘要的记忆系统"""
    summary: str = ""
    recent_messages: list[dict[str, str]] = field(
        default_factory=list,
    )
    max_recent: int = 10

    def add_message(self, role: str, content: str) -> None:
        self.recent_messages.append({
            "role": role,
            "content": content,
        })

        if len(self.recent_messages) > self.max_recent:
            self._compress()

    def _compress(self) -> None:
        """压缩旧消息为摘要"""
        # 取出前半部分消息进行摘要
        half = len(self.recent_messages) // 2
        old_messages = self.recent_messages[:half]
        self.recent_messages = self.recent_messages[half:]

        # 实际中调用 LLM 生成摘要
        old_text = " | ".join(
            f"{m['role']}: {m['content'][:50]}"
            for m in old_messages
        )
        if self.summary:
            self.summary += f"\n之前的对话: {old_text}"
        else:
            self.summary = f"对话摘要: {old_text}"

    def get_context(self) -> str:
        """获取完整上下文"""
        parts = []
        if self.summary:
            parts.append(f"[历史摘要]\n{self.summary}")
        parts.append("[近期对话]")
        for msg in self.recent_messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(parts)
```

## 长期记忆：向量存储

长期记忆通过向量数据库实现持久化存储和语义检索：

```python
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """长期记忆条目"""
    content: str
    timestamp: str
    importance: float = 0.5
    access_count: int = 0
    tags: list[str] = field(default_factory=list)


class LongTermMemory:
    """长期记忆系统"""

    def __init__(self, embedding_dim: int = 384):
        self.entries: list[MemoryEntry] = []
        self.vectors: list[np.ndarray] = []
        self.dim = embedding_dim

    def store(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> None:
        """存储记忆"""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now().isoformat(),
            importance=importance,
            tags=tags or [],
        )
        self.entries.append(entry)
        # 实际中使用嵌入模型生成向量
        vector = np.random.randn(self.dim)
        self.vectors.append(vector / np.linalg.norm(vector))

    def recall(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """检索相关记忆"""
        if not self.vectors:
            return []

        query_norm = query_vector / np.linalg.norm(query_vector)
        scores = []
        for i, vec in enumerate(self.vectors):
            similarity = float(np.dot(query_norm, vec))
            # 综合相似度和重要性
            combined = (
                0.7 * similarity
                + 0.3 * self.entries[i].importance
            )
            scores.append((i, combined))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, _ in scores[:top_k]:
            self.entries[idx].access_count += 1
            results.append(self.entries[idx])
        return results

    def forget(self, threshold: float = 0.1) -> int:
        """遗忘不重要的记忆"""
        before = len(self.entries)
        keep_indices = [
            i for i, entry in enumerate(self.entries)
            if entry.importance >= threshold
            or entry.access_count > 0
        ]
        self.entries = [self.entries[i] for i in keep_indices]
        self.vectors = [self.vectors[i] for i in keep_indices]
        return before - len(self.entries)


# 使用示例
ltm = LongTermMemory()
ltm.store("用户喜欢 Python 编程", importance=0.8, tags=["偏好"])
ltm.store("用户问过 Transformer 架构", importance=0.6, tags=["历史"])
ltm.store("用户的项目使用 PyTorch", importance=0.7, tags=["技术栈"])
print(f"存储了 {len(ltm.entries)} 条记忆")
```

## 反思记忆：从经验中学习

Agent 可以通过反思过去的行为来改进未来的决策：

```python
from dataclasses import dataclass


@dataclass
class Reflection:
    """反思记录"""
    task: str
    outcome: str
    is_success: bool
    lesson: str


class ReflectionMemory:
    """反思记忆"""

    def __init__(self):
        self.reflections: list[Reflection] = []

    def add_reflection(
        self,
        task: str,
        outcome: str,
        is_success: bool,
        lesson: str,
    ) -> None:
        self.reflections.append(Reflection(
            task=task,
            outcome=outcome,
            is_success=is_success,
            lesson=lesson,
        ))

    def get_lessons(
        self,
        only_failures: bool = False,
    ) -> list[str]:
        """获取经验教训"""
        filtered = self.reflections
        if only_failures:
            filtered = [
                r for r in filtered if not r.is_success
            ]
        return [r.lesson for r in filtered]

    def get_success_rate(self) -> float:
        """计算成功率"""
        if not self.reflections:
            return 0.0
        successes = sum(
            1 for r in self.reflections if r.is_success
        )
        return successes / len(self.reflections)


# 使用示例
rm = ReflectionMemory()
rm.add_reflection(
    task="搜索最新论文",
    outcome="成功找到 5 篇相关论文",
    is_success=True,
    lesson="使用具体的关键词搜索效果更好",
)
rm.add_reflection(
    task="生成代码",
    outcome="代码有语法错误",
    is_success=False,
    lesson="生成代码后应该先验证语法再返回",
)
print(f"成功率: {rm.get_success_rate():.0%}")
print(f"失败教训: {rm.get_lessons(only_failures=True)}")
```

## 记忆架构设计原则

1. **分层存储**：不同类型的信息用不同的存储策略
2. **主动遗忘**：定期清理低价值记忆，避免噪声干扰
3. **关联检索**：支持语义相似度搜索，而非精确匹配
4. **重要性评估**：根据使用频率和相关性动态调整优先级
5. **隐私保护**：敏感信息加密存储，支持用户删除

## 总结

记忆系统是 Agent 从"无状态工具"进化为"有经验的助手"的关键。通过工作记忆、长期记忆和反思记忆的协同工作，Agent 能够积累经验、个性化服务，并持续改进自身的表现。下一篇我们将探讨 Agent 安全与对齐问题。
