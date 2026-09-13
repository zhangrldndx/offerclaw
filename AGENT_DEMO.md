# OfferClaw · Agent Demo

一个命令行可交互的最小 AI Agent 实现，演示「用户提问 → LLM 决定调工具 → 工具执行 → LLM 整合结果 → 回复」的完整闭环。

本 Demo 是 **OfferClaw 项目 V1 代码实现层的核心交付物**，同时作为第一个可写进简历的 AI 应用项目雏形。

---

## 快速开始

### 前置条件

- Python 3.10+
- `requests` 库（`pip install requests`）
- 智谱 AI 账号 + API Key（https://open.bigmodel.cn/ 免费注册）

### 配置

项目根目录创建 `.env.local` 文件（已在 `.gitignore` 中）：

```
ZHIPU_API_KEY=你的智谱密钥
```

### 运行

```bash
python agent_demo.py
```

看到以下提示即启动成功：

```
============================================================
OfferClaw · Agent Demo · 最小可运行版
Model    : glm-4-flash
Tools    : get_current_time, calculator, echo
输入 'quit' 或 Ctrl+C 退出
============================================================

You:
```

---

## 架构

```
用户输入（命令行 input）
   ↓
agent_demo.py（主循环）
   ├─ load_local_env()      读取 .env.local
   ├─ build_zhipu_jwt()     构造智谱 JWT Bearer Token
   ├─ call_llm()            向智谱 API POST（带 tools 参数）
   │
   └─ LLM 响应含 tool_calls？
       │
       ├─ 是 → execute_tool_call()
       │        ↓
       │      路由到 tools.py 里的实际函数
       │        ↓
       │      把结果以 role=tool 追加进 messages
       │        ↓
       │      继续下一轮 call_llm（最多 5 轮迭代）
       │
       └─ 否 → 返回最终文本回复给用户
                ↓
tools.py（工具模块）
   ├─ tool_get_current_time()   返回当前本地时间
   ├─ tool_calculator()         AST 安全求值（不用 eval）
   ├─ tool_echo()               原样回显
   ├─ TOOL_FUNCTIONS            name → function 路由表
   └─ TOOLS_SCHEMA              OpenAI 兼容 JSON schema
```

## 文件清单

| 文件 | 行数 | 作用 |
|---|---|---|
| `agent_demo.py` | ~300 | 主入口：密钥加载 / JWT 签名 / LLM 调用 / Agent 主循环 / 命令行交互 |
| `tools.py` | ~150 | 工具实现：3 个工具函数 + 注册表 + OpenAI 兼容 schema |
| `README.md` | 本文档 | 运行说明 + 架构图 + 简历描述 |
| `.env.local` | 1 行 | 智谱 API Key（gitignored） |

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 单会话多轮对话 | 进程内保留 conversation history，退出即清空 |
| OpenAI 兼容 function calling | 基于智谱 GLM-4-Flash 原生 tools 参数 |
| 工具调用循环 | 单轮对话内最多 5 次 tool call 迭代，防死循环 |
| AST 安全计算器 | 拒绝 `__import__` / `eval` 等代码注入 |
| 手写智谱 JWT 签名 | 纯标准库实现，无 PyJWT 依赖 |
| 零 Agent 框架 | 无 LangChain / LlamaIndex / LangGraph |

## 3 个内置工具

| 工具 | 描述 | 参数 | 示例 |
|---|---|---|---|
| `get_current_time` | 获取当前本地时间（精确到秒） | 无 | `2026-04-16 00:32:51` |
| `calculator` | 安全数学表达式求值（`+ - * / ** %` 和括号） | `expression: str` | `(3+4)*5 → 35` |
| `echo` | 原样回显文本 | `text: str` | `hello → echo: hello` |

---

## 验证（4 级阶梯测试）

按顺序敲下面 4 类问题，每一级验证不同能力。

### 阶梯 1 · 无工具闲聊

```
You: 你是谁？
Agent: 我是 OfferClaw Agent Demo，一个带工具调用能力的最小演示助手。
```

验证：LLM 链路 + system prompt 加载 + **LLM 正确识别"不需要调工具"**。

### 阶梯 2 · 单工具调用

```
You: 现在几点了？
  [tool:get_current_time] 2026-04-16 00:32:51
Agent: 当前时间是 2026-04-16 00:32:51。
```

验证：function calling 基本链路 + 无参数工具调用。

### 阶梯 3 · 带参数工具调用

```
You: 帮我算 1234 * 5678
  [tool:calculator] 7006652
Agent: 1234 * 5678 等于 7006652。
```

验证：LLM 生成结构化参数 + 参数解析 + 计算器正确求值。

### 阶梯 4 · 多工具链式调用

```
You: 现在几点？然后算一下从现在到 2026-05-01 还剩多少天
  [tool:get_current_time] 2026-04-16 00:33:35
  [tool:calculator] 15.0
Agent: 现在是 2026-04-16 00:33:35，还剩 15 天。
```

验证：**单轮对话内的连续多工具决策** + tool loop 迭代 + LLM 基于工具结果做二次推理。

---

## 已知限制 / 明确不做

| 能力 | 状态 | 下一版计划 |
|---|---|---|
| RAG / 向量检索 / 知识库 | 当时未做 | 后续只索引已批准资料；候选 JD 不作为知识库 |
| 跨会话持久化 memory | 不做 | 下一版加 JSON 文件存 conversation history |
| LangGraph / LlamaIndex 框架化 | 不做 | 工具数 > 5 时再考虑 |
| Web UI | 不做 | 命令行够用 |
| 多 Agent 协作 | 不做 | V1 不在范围 |
| 复杂错误恢复 / retry | 不做 | 当前只做基础异常捕获 + 回滚 user 消息 |
| `simple_profile_lookup` 第 4 个工具 | 预留 | 读取 `user_profile.md` 的简单查询工具，下一版加 |

---

## 安全设计

### 计算器为什么不用 `eval()`

Python 的 `eval()` 可以执行任意代码，包括 `__import__("os").system("rm -rf /")` 这种注入。本项目用 `ast.parse` + 白名单遍历的方式求值：

- 只允许 `ast.Constant`（数字）/ `ast.BinOp`（二元运算）/ `ast.UnaryOp`（一元运算）
- 拒绝 `ast.Call`（函数调用）/ `ast.Attribute`（属性访问）/ `ast.Name`（变量引用）
- 运算符白名单：`+ - * / ** %` 和一元 `+/-`

实测验证：

```python
tool_calculator('__import__("os").system("ls")')
# → "计算出错：ValueError: 不支持的表达式节点：Call"
```

### API Key 管理

- `.env.local` 存放密钥，加入 `.gitignore`
- `load_local_env()` 程序启动时读取 → 注入 `os.environ`
- 已存在的环境变量优先级 > `.env.local`（支持临时覆盖）
- 智谱 JWT 只在每次 HTTP 请求时临时生成，过期时间 1 小时

---

## 简历描述草稿

### 完整版（约 180 字，简历项目经历栏）

> **OfferClaw Agent Demo**（Python 个人项目，2026-04）
>
> 从零实现的最小可运行 AI Agent，演示「LLM 调用 + 工具路由 + 多轮对话 + 工具调用循环」完整闭环。
>
> **核心技术**：智谱 GLM-4-Flash API + OpenAI 兼容 function calling + 手写 JWT 签名（纯标准库，无 PyJWT 依赖）。
>
> **功能点**：注册 3 个内置工具（时间查询 / AST 安全计算器 / 文本回显）；工具调用循环最多 5 次迭代防死循环；单会话多轮对话；安全计算器用 Python `ast` 白名单解析替代 `eval()`，可抵御代码注入。
>
> **工程亮点**：纯标准库 + `requests`，无 LangChain / LlamaIndex / LangGraph 等第三方 Agent 框架依赖；代码约 450 行分为 `agent_demo.py`（主循环）和 `tools.py`（工具模块）两个职责清晰的文件；4 级阶梯测试（无工具闲聊 / 单工具 / 参数工具 / 多工具链式）全部通过。

### 一句话版（简历技能栏或自我介绍）

> 独立实现最小 AI Agent，基于智谱 GLM-4-Flash 原生 function calling + 手写 JWT + AST 安全计算器，纯标准库无框架依赖，~450 行代码 / 2 模块 / 3 工具 / 4 级验证通过。

---

## 开发日志

- **2026-04-15**：`day1_api_starter.py` LLM API 底座跑通（智谱 JWT 认证 + `.env.local` 加载机制）
- **2026-04-16**：Agent Demo 最小可运行版完成；`tools.py` 模块化拆分；4 级阶梯测试全通过

---

## 下一步改进方向

按优先级：

1. **加第 4 个工具 `simple_profile_lookup`**：读取 `user_profile.md`，让 Agent 能回答"我的专业是什么"这类关于用户画像的问题
2. **跨会话 memory**：用 JSON 文件存每次对话历史，启动时恢复
3. **最小 RAG 模块**：索引已批准资料；岗位适配通过真实投递绑定 JD 的结构化链路完成
4. **接入 OfferClaw 其他模块**：和 `match_job.py` / `job_match_prompt.md` 打通，Agent 直接调用完整的匹配流程

---

## 许可

本项目是 OfferClaw 求职作战官的一部分，仅供学习和个人求职使用。
