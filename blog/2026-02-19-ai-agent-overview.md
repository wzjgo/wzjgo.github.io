---
title: AI Agent 全景：从概念到架构
slug: ai-agent-overview
authors: [default-author]
tags: [大语言模型, 机器学习]
date: 2026-02-19
---

AI Agent 是当前大语言模型领域最令人兴奋的方向之一。它让 LLM 不再只是"聊天机器人"，而是能够自主规划、使用工具、完成复杂任务的智能体。本文将全面介绍 AI Agent 的核心概念和架构设计。

<!-- truncate -->

## 什么是 AI Agent？

AI Agent（智能体）是一个能够感知环境、自主决策并采取行动来实现目标的系统。在 LLM 时代，Agent 通常以大语言模型为"大脑"，结合工具调用、记忆系统和规划能力，形成一个完整的自主系统。

与传统的 LLM 对话不同，Agent 具备以下核心能力：

- **自主规划**：将复杂任务分解为可执行的步骤
- **工具使用**：调用外部 API、执行代码、搜索信息
- **记忆管理**：维护短期和长期记忆，积累经验
- **自我反思**：评估执行结果，调整策略

## Agent 的核心架构

一个典型的 AI Agent 系统包含四个核心模块：

```
┌─────────────────────────────────────┐
│            AI Agent                  │
│                                      │
│  ┌──────────┐    ┌──────────────┐   │
│  │  规划器   │    │   记忆系统    │   │
│  │ Planner  │    │   Memory     │   │
│  └────┬─────┘    └──────┬───────┘   │
│       │                  │           │
│  ┌────▼──────────────────▼───────┐  │
│  │        LLM（大脑）             │  │
│  └────┬──────────────────┬───────┘  │
│       │                  │           │
│  ┌────▼─────┐    ┌──────▼───────┐   │
│  │  工具集   │    │   执行器     │   │
│  │  Tools   │    │  Executor   │   │
│  └──────────┘    └──────────────┘   │
└─────────────────────────────────────┘
```

## 规划：任务分解与推理

规划是 Agent 最核心的能力。常见的规划策略包括：

### ReAct（Reasoning + Acting）

ReAct 将推理和行动交替进行，每一步都先思考再行动：

```python
from dataclasses import dataclass


@dataclass
class AgentStep:
    """Agent 执行的一个步骤"""
    thought: str      # 思考过程
    action: str       # 要执行的动作
    action_input: str # 动作的输入
    observation: str   # 执行结果


def react_prompt(question: str, steps: list[AgentStep]) -> str:
    """构建 ReAct 提示"""
    prompt = f"问题: {question}\n\n"

    for i, step in enumerate(steps, 1):
        prompt += f"思考 {i}: {step.thought}\n"
        prompt += f"动作 {i}: {step.action}[{step.action_input}]\n"
        prompt += f"观察 {i}: {step.observation}\n\n"

    prompt += f"思考 {len(steps) + 1}: "
    return prompt


# 示例
steps = [
    AgentStep(
        thought="我需要搜索最新的 AI 新闻",
        action="search",
        action_input="2026 AI latest news",
        observation="找到了 10 条相关结果...",
    ),
]
print(react_prompt("最近 AI 领域有什么新进展？", steps))
```

### Plan-and-Execute

先制定完整计划，再逐步执行：

```python
from dataclasses import dataclass, field


@dataclass
class Plan:
    """执行计划"""
    goal: str
    steps: list[str] = field(default_factory=list)
    current_step: int = 0

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    @property
    def next_step(self) -> str | None:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def advance(self) -> None:
        self.current_step += 1


# 示例：Agent 制定的计划
plan = Plan(
    goal="写一篇关于 Transformer 的技术博客",
    steps=[
        "搜索 Transformer 的最新研究进展",
        "整理核心概念和关键创新点",
        "编写代码示例",
        "撰写文章草稿",
        "审校和优化",
    ],
)

while not plan.is_complete:
    print(f"执行: {plan.next_step}")
    plan.advance()
```

## 工具使用（Tool Use）

工具是 Agent 与外部世界交互的桥梁。常见的工具类型：

| 工具类型 | 示例 | 用途 |
|----------|------|------|
| 搜索 | Google、Bing | 获取最新信息 |
| 代码执行 | Python 解释器 | 计算和数据处理 |
| API 调用 | 天气、地图、数据库 | 获取结构化数据 |
| 文件操作 | 读写文件 | 持久化存储 |
| 浏览器 | 网页浏览 | 获取网页内容 |

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """Agent 可用的工具"""
    name: str
    description: str
    func: Callable[..., str]


def search(query: str) -> str:
    """模拟搜索工具"""
    return f"搜索 '{query}' 的结果: ..."


def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression)  # 仅用于演示
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 注册工具
tools = [
    Tool(
        name="search",
        description="搜索互联网获取信息",
        func=search,
    ),
    Tool(
        name="calculator",
        description="执行数学计算",
        func=calculator,
    ),
]

# 工具描述（提供给 LLM）
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
```

## 记忆系统

Agent 的记忆分为短期记忆和长期记忆：

- **短期记忆**：当前对话的上下文窗口，受 token 限制
- **长期记忆**：通过向量数据库持久化存储的历史信息

```python
from dataclasses import dataclass, field


@dataclass
class Memory:
    """简单的 Agent 记忆系统"""
    short_term: list[dict[str, str]] = field(default_factory=list)
    long_term: list[dict[str, str]] = field(default_factory=list)
    max_short_term: int = 20

    def add_message(self, role: str, content: str) -> None:
        """添加到短期记忆"""
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.max_short_term:
            # 将溢出的记忆转移到长期记忆
            overflow = self.short_term.pop(0)
            self.long_term.append(overflow)

    def get_context(self) -> list[dict[str, str]]:
        """获取当前上下文"""
        return self.short_term.copy()

    def search_long_term(self, query: str) -> list[dict[str, str]]:
        """搜索长期记忆（简化版，实际应使用向量检索）"""
        results = []
        for memory in self.long_term:
            if query.lower() in memory["content"].lower():
                results.append(memory)
        return results
```

## Agent 框架生态

当前主流的 Agent 开发框架：

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangChain | 生态丰富，组件多 | 通用 Agent 开发 |
| LlamaIndex | 数据连接能力强 | RAG + Agent |
| AutoGen | 多 Agent 协作 | 复杂多步骤任务 |
| CrewAI | 角色扮演式协作 | 团队模拟 |
| Dify | 低代码平台 | 快速原型开发 |

## 总结

AI Agent 代表了 LLM 应用的下一个阶段。通过规划、工具使用和记忆管理，Agent 能够处理远超简单问答的复杂任务。在接下来的文章中，我们将深入探讨 Function Calling 的实现细节。
