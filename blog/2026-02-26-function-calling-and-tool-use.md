---
title: Function Calling 深入解析：让 LLM 调用工具
slug: function-calling-and-tool-use
authors: [default-author]
tags: [大语言模型, 机器学习]
date: 2026-02-26
---

Function Calling 是 AI Agent 的核心能力之一，它让大语言模型能够结构化地调用外部函数和 API。本文将深入解析 Function Calling 的工作原理、实现方式和最佳实践。

<!-- truncate -->

## Function Calling 是什么？

Function Calling（函数调用）是 LLM 的一种能力：模型根据用户的自然语言请求，自动决定调用哪个函数，并生成符合函数签名的结构化参数。

传统流程：
```
用户 -> LLM -> 文本回答
```

Function Calling 流程：
```
用户 -> LLM -> 选择函数 + 生成参数 -> 执行函数 -> LLM -> 最终回答
```

## 定义工具（Tool Definition）

工具通过 JSON Schema 描述其名称、功能和参数：

```python
import json
from typing import Any

# 工具定义（OpenAI 格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在知识库中搜索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

print(json.dumps(tools, indent=2, ensure_ascii=False))
```

## 实现 Function Calling 循环

一个完整的 Function Calling 实现需要处理多轮调用：

```python
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class FunctionCall:
    """LLM 生成的函数调用"""
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """工具执行结果"""
    name: str
    result: str
    is_error: bool = False


class FunctionCallingAgent:
    """支持 Function Calling 的 Agent"""

    def __init__(self):
        self.tools: dict[str, Callable[..., str]] = {}
        self.messages: list[dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        func: Callable[..., str],
    ) -> None:
        """注册工具"""
        self.tools[name] = func

    def execute_function(self, call: FunctionCall) -> ToolResult:
        """执行函数调用"""
        if call.name not in self.tools:
            return ToolResult(
                name=call.name,
                result=f"未知工具: {call.name}",
                is_error=True,
            )

        try:
            result = self.tools[call.name](**call.arguments)
            return ToolResult(name=call.name, result=result)
        except Exception as e:
            return ToolResult(
                name=call.name,
                result=f"执行错误: {e}",
                is_error=True,
            )

    def run(self, user_input: str) -> str:
        """运行 Agent（简化版）"""
        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        # 模拟 LLM 决定调用工具
        # 实际中这里会调用 LLM API
        call = self._mock_llm_decision(user_input)

        if call:
            result = self.execute_function(call)
            self.messages.append({
                "role": "tool",
                "name": result.name,
                "content": result.result,
            })
            return f"调用了 {result.name}，结果: {result.result}"

        return "直接回答（无需工具）"

    def _mock_llm_decision(
        self,
        user_input: str,
    ) -> FunctionCall | None:
        """模拟 LLM 的工具选择决策"""
        if "天气" in user_input:
            return FunctionCall(
                name="get_weather",
                arguments={"city": "北京"},
            )
        return None


# 使用示例
agent = FunctionCallingAgent()
agent.register_tool(
    "get_weather",
    lambda city, unit="celsius": f"{city}今天 22°C，晴",
)

result = agent.run("北京今天天气怎么样？")
print(result)
```

## 并行函数调用

现代 LLM 支持在一次响应中生成多个函数调用，提高效率：

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class FunctionCall:
    """函数调用"""
    name: str
    arguments: dict[str, Any]


async def execute_parallel(
    calls: list[FunctionCall],
    tools: dict[str, Callable[..., str]],
) -> list[str]:
    """并行执行多个函数调用"""

    async def execute_one(call: FunctionCall) -> str:
        func = tools.get(call.name)
        if not func:
            return f"未知工具: {call.name}"
        # 模拟异步执行
        await asyncio.sleep(0.1)
        return func(**call.arguments)

    tasks = [execute_one(call) for call in calls]
    return await asyncio.gather(*tasks)


# 示例：同时查询多个城市的天气
async def main():
    tools = {
        "get_weather": lambda city, **_: f"{city}: 晴 22°C",
    }

    calls = [
        FunctionCall("get_weather", {"city": "北京"}),
        FunctionCall("get_weather", {"city": "上海"}),
        FunctionCall("get_weather", {"city": "广州"}),
    ]

    results = await execute_parallel(calls, tools)
    for call, result in zip(calls, results):
        print(f"{call.arguments['city']}: {result}")

# asyncio.run(main())
```

## 错误处理与重试

健壮的 Function Calling 需要完善的错误处理：

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    retry_on_error: bool = True


def execute_with_retry(
    func_name: str,
    arguments: dict[str, Any],
    tools: dict[str, Any],
    config: RetryConfig = RetryConfig(),
) -> str:
    """带重试的函数执行"""
    last_error = None

    for attempt in range(config.max_retries):
        try:
            func = tools[func_name]
            result = func(**arguments)
            return result
        except KeyError:
            return f"工具 '{func_name}' 不存在"
        except TypeError as e:
            # 参数错误，让 LLM 重新生成参数
            last_error = f"参数错误 (尝试 {attempt + 1}): {e}"
            if not config.retry_on_error:
                break
        except Exception as e:
            last_error = f"执行错误 (尝试 {attempt + 1}): {e}"
            if not config.retry_on_error:
                break

    return f"执行失败: {last_error}"
```

## Function Calling 最佳实践

1. **工具描述要清晰**：description 越准确，LLM 选择工具越精确
2. **参数约束要明确**：使用 enum、required 等约束减少错误
3. **控制工具数量**：工具太多会降低选择准确率，建议不超过 20 个
4. **做好错误处理**：工具执行可能失败，需要优雅降级
5. **日志记录**：记录每次工具调用的输入输出，便于调试

## 总结

Function Calling 是连接 LLM 与外部世界的关键桥梁。通过结构化的工具定义和调用机制，LLM 从"只会说"变成了"能做事"。下一篇文章我们将探讨 RAG 技术，让 Agent 拥有更强的知识获取能力。
