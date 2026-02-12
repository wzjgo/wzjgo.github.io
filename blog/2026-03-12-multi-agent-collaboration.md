---
title: 多 Agent 协作：构建 AI 团队
slug: multi-agent-collaboration
authors: [default-author]
tags: [大语言模型, 机器学习]
date: 2026-03-12
---

单个 Agent 的能力终究有限。多 Agent 协作通过让多个专业化的 Agent 分工合作，能够解决更加复杂的任务。本文将介绍多 Agent 系统的设计模式、通信机制和实际应用。

<!-- truncate -->

## 为什么需要多 Agent？

单个 Agent 面临的挑战：

- **上下文窗口限制**：复杂任务需要的信息量超出单次对话容量
- **专业化需求**：不同子任务需要不同的专业知识和工具
- **可靠性**：单点故障风险，缺乏交叉验证
- **效率**：串行处理无法充分利用并行能力

多 Agent 系统通过分工协作解决这些问题，就像一个高效的开发团队。

## 多 Agent 架构模式

### 1. 主从模式（Orchestrator Pattern）

一个主 Agent 负责任务分配和结果汇总，多个子 Agent 负责具体执行：

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """Agent 间的消息"""
    sender: str
    receiver: str
    content: str
    msg_type: str = "task"  # task, result, feedback


@dataclass
class SubAgent:
    """子 Agent"""
    name: str
    role: str
    skills: list[str] = field(default_factory=list)

    def execute(self, task: str) -> str:
        """执行任务（简化版）"""
        return f"[{self.name}] 完成任务: {task}"


class Orchestrator:
    """主 Agent：任务编排器"""

    def __init__(self):
        self.agents: dict[str, SubAgent] = {}
        self.message_log: list[AgentMessage] = []

    def register_agent(self, agent: SubAgent) -> None:
        """注册子 Agent"""
        self.agents[agent.name] = agent

    def assign_task(
        self,
        task: str,
        agent_name: str,
    ) -> str:
        """分配任务给指定 Agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' 不存在"

        # 记录消息
        self.message_log.append(AgentMessage(
            sender="orchestrator",
            receiver=agent_name,
            content=task,
            msg_type="task",
        ))

        result = agent.execute(task)

        self.message_log.append(AgentMessage(
            sender=agent_name,
            receiver="orchestrator",
            content=result,
            msg_type="result",
        ))

        return result

    def run(self, goal: str) -> list[str]:
        """执行复杂任务"""
        # 简化版：实际中由 LLM 决定任务分配
        results = []
        for name, agent in self.agents.items():
            result = self.assign_task(
                f"作为 {agent.role}，完成: {goal}",
                name,
            )
            results.append(result)
        return results


# 构建团队
orchestrator = Orchestrator()
orchestrator.register_agent(SubAgent(
    name="researcher",
    role="研究员",
    skills=["信息检索", "文献分析"],
))
orchestrator.register_agent(SubAgent(
    name="writer",
    role="技术作者",
    skills=["文章撰写", "内容优化"],
))
orchestrator.register_agent(SubAgent(
    name="reviewer",
    role="审稿人",
    skills=["质量审核", "事实核查"],
))

results = orchestrator.run("撰写一篇关于 AI Agent 的技术文章")
for r in results:
    print(r)
```

### 2. 对等模式（Peer-to-Peer Pattern）

Agent 之间平等协作，通过消息传递进行沟通：

```python
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PeerAgent:
    """对等 Agent"""
    name: str
    role: str
    inbox: deque[AgentMessage] = field(
        default_factory=deque,
    )

    def send(
        self,
        receiver: 'PeerAgent',
        content: str,
    ) -> None:
        """发送消息"""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver.name,
            content=content,
        )
        receiver.inbox.append(msg)

    def process_messages(self) -> list[str]:
        """处理收件箱中的消息"""
        responses = []
        while self.inbox:
            msg = self.inbox.popleft()
            response = f"[{self.name}] 收到来自 {msg.sender} 的消息，已处理"
            responses.append(response)
        return responses


# 对等协作示例
coder = PeerAgent(name="coder", role="开发者")
tester = PeerAgent(name="tester", role="测试员")

coder.send(tester, "我完成了登录功能的代码，请帮忙测试")
responses = tester.process_messages()
for r in responses:
    print(r)
```

### 3. 辩论模式（Debate Pattern）

多个 Agent 对同一问题提出不同观点，通过辩论达成共识：

```python
from dataclasses import dataclass


@dataclass
class DebateRound:
    """辩论回合"""
    agent_name: str
    position: str
    argument: str


def run_debate(
    topic: str,
    agents: list[str],
    rounds: int = 3,
) -> list[DebateRound]:
    """运行辩论（简化版）"""
    debate_log: list[DebateRound] = []

    for round_num in range(rounds):
        for agent in agents:
            # 实际中每个 Agent 会基于之前的论点生成新论点
            debate_log.append(DebateRound(
                agent_name=agent,
                position=f"第 {round_num + 1} 轮观点",
                argument=f"[{agent}] 关于 '{topic}' 的论述...",
            ))

    return debate_log


# 辩论示例
log = run_debate(
    topic="AI Agent 是否应该具有自主决策权？",
    agents=["乐观派", "谨慎派", "中立派"],
    rounds=2,
)
for entry in log:
    print(f"{entry.agent_name}: {entry.argument}")
```

## Agent 间通信协议

设计良好的通信协议是多 Agent 系统的基础：

```python
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    """消息类型"""
    Request = "request"       # 请求
    Response = "response"     # 响应
    Broadcast = "broadcast"   # 广播
    Delegate = "delegate"     # 委托
    Report = "report"         # 汇报


@dataclass
class StructuredMessage:
    """结构化消息"""
    msg_type: MessageType
    sender: str
    receiver: str
    content: str
    context: dict[str, str] = field(default_factory=dict)
    priority: int = 0  # 0=普通, 1=高, 2=紧急

    def to_prompt(self) -> str:
        """转换为 LLM 可理解的提示"""
        return (
            f"[{self.msg_type.value}] "
            f"来自: {self.sender}\n"
            f"内容: {self.content}\n"
            f"上下文: {self.context}"
        )
```

## 实际应用场景

| 场景 | Agent 角色 | 协作方式 |
|------|-----------|----------|
| 软件开发 | 产品经理、开发者、测试员、审核员 | 主从 + 流水线 |
| 研究分析 | 检索员、分析师、写作者 | 主从模式 |
| 内容创作 | 策划、写手、编辑、设计 | 流水线模式 |
| 决策支持 | 多个领域专家 | 辩论模式 |
| 客服系统 | 路由、专业客服、质检 | 主从模式 |

## 挑战与注意事项

1. **通信开销**：Agent 间的消息传递消耗 token 和时间
2. **一致性**：多个 Agent 的输出可能相互矛盾
3. **死循环**：Agent 之间可能陷入无限的来回沟通
4. **调试困难**：多 Agent 系统的行为难以预测和复现
5. **成本控制**：每个 Agent 的 LLM 调用都有成本

## 总结

多 Agent 协作是 AI Agent 技术的高级形态。通过合理的架构设计、清晰的角色分工和高效的通信机制，多 Agent 系统能够解决单个 Agent 难以应对的复杂任务。下一篇我们将探讨 Agent 的记忆与状态管理。
