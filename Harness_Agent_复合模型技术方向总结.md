# Harness Agent 的“复合模型”技术方向总结

## 一、结论

如果把 `Harness Agent` 中提到的“复合模型”放到当前 AI 技术语境中理解，它更准确地说并不是某一种新的底层大模型结构，而是一个**复合式智能体系统**。

它对应的核心技术方向是：

- `Agent Engineering`
- `Agent Harness`
- `LLMOps`
- `DevOps Automation`
- `Enterprise AI Runtime`

它的重点不在于“训练一个更大的模型”，而在于把多个能力层组合起来，让大模型真正能够在企业 DevOps 场景中执行任务、调用工具、接入流程，并受到安全治理约束。

可以用一句话概括：

> Harness Agent 的“复合模型”，本质上是“模型能力 + 工具能力 + 上下文感知 + 工作流编排 + 安全治理”的复合式 Agent 架构。

---

## 二、为什么它不是传统意义上的“模型方向”

很多人在看到“复合模型”时，第一反应会想到以下这些偏底层模型算法的方向：

- `MoE`（混合专家模型）
- 多模态模型
- 模型集成（Ensemble）
- 模型蒸馏
- `RAG` 检索增强生成

但从 Harness 官方文档的表述来看，它的重点并不是发明一种新的神经网络结构，而是强调：

> `Agent = Model + Harness`

其中：

- `Model` 负责语言理解、推理、生成
- `Harness` 负责执行环境、上下文管理、工具调用、流程控制、错误恢复、权限约束和审计

因此，Harness Agent 更偏向**系统工程方向**，而不是纯粹的底层模型架构方向。

---

## 三、“复合”到底复合了什么

从系统设计角度看，Harness Agent 的“复合”至少包括以下五层。

### 1. 模型层

Harness 官方文档明确提到可接入多种模型，例如：

- Anthropic
- OpenAI
- Gemini

这说明它不是依赖单一模型，而是采用一种**模型可替换、可路由、可按任务选型**的思路。

例如：

- 代码生成可以选择更擅长编码的模型
- 日志分析可以选择更擅长长上下文理解的模型
- YAML 或配置生成可以选择结构化输出稳定的模型

因此，它更像是一个**模型编排平台**，而不是一个固定模型产品。

### 2. 上下文层

Harness Agent 并不是脱离环境的聊天机器人。它强调接入真实平台上下文，例如：

- pipelines
- services
- environments
- infrastructure configs
- 历史执行记录
- 知识图谱（Knowledge Graph）

这一层的意义是让 Agent 的判断建立在真实上下文上，而不是仅凭通用语言知识进行“猜测式回答”。

这类能力属于典型的：

- `context-grounded agent`
- `environment-aware agent`

也就是上下文锚定、环境感知型智能体。

### 3. 工具层

Harness 文档中提到：

- `MCP`
- Harness APIs
- pipeline-native steps

这意味着 Agent 不只是“会回答”，还可以：

- 读取日志
- 分析失败原因
- 生成或修改 YAML
- 创建 PR
- 触发或参与流水线步骤
- 调用平台 API
- 输出修复建议或直接执行修复流程

这一层属于：

- `Tool-Using Agent`
- `Action Agent`

也就是具备工具调用与任务执行能力的智能体。

### 4. 编排层

Harness Agent 最关键的产品特征之一是：

> Agent 运行在 Pipeline 中。

这和普通聊天式 AI 助手非常不同。

它意味着 Agent 可以成为软件交付流程中的一个节点：

1. 流水线失败
2. Agent 自动读取日志和上下文
3. 分析根因
4. 生成修复方案
5. 修改配置或代码
6. 创建 PR 或继续后续审批流程

这属于：

- `AI + Workflow Orchestration`
- `Pipeline-native Agent`

也就是把 AI 能力嵌入企业工作流和自动化系统中，而不是仅提供一个对话窗口。

### 5. 治理层

Harness 特别强调企业级治理能力，例如：

- `RBAC`
- `OPA policy`
- 审计日志
- allow-listed tools
- approval gates
- 合规追踪

这一层说明 Harness 的目标不是“尽可能自由地自动执行”，而是“在可控、可审计、可授权的边界内自动执行”。

在企业场景里，这一点比单纯提升模型能力更重要，因为真正的风险往往来自：

- 越权访问
- 数据泄露
- 错误修改生产配置
- 无法审计的自动操作

所以它本质上也是一种：

- `AI capability + enterprise governance`

的复合方向。

---

## 四、放到整个 AI 技术图谱里，它属于哪些赛道

如果把 Harness Agent 放到更大的 AI 技术版图中，它大致位于以下几个交叉方向。

### 1. Agent Engineering

目标是把大模型从“回答问题的模型”变成“能够执行任务的智能体”。

典型关键词：

- planning
- tool use
- memory
- reflection
- orchestration
- verification

### 2. LLMOps / DevOps AI

Harness 的落点非常明确，偏向把 AI 深度用于：

- CI
- CD
- Code Review
- Failure RCA
- Security Remediation
- Release Automation

因此它是典型的：

- `LLM + DevOps`
- `AI for Software Delivery`

方向。

### 3. Enterprise AI Runtime

这里的重点不是让模型更聪明，而是让系统更适合企业使用，例如：

- 权限控制
- secrets 管理
- connectors
- 可观测性
- 多租户
- 合规审计

这一层决定 AI 是否能真正进入生产环境。

### 4. Composite AI Systems

如果一定要给“复合模型”找一个更准确的学术或产业表述，那么更接近的是：

`Composite AI System`

也就是多个能力模块共同构成智能行为，例如：

- LLM
- retrieval
- memory
- knowledge graph
- tools
- policy engine
- workflow engine

这比“单一模型”更贴近 Harness Agent 的真实形态。

---

## 五、它和普通 AI 助手的区别

普通 AI 助手通常是：

- 你提问，它回答
- 最多帮助写代码、写文档、解释错误
- 缺少真实执行权限和企业流程接入能力

而 Harness Agent 更像是：

- 在 pipeline 中运行
- 拿到真实运行上下文
- 能做多步任务分解
- 能调用工具和平台 API
- 能接入审批、策略、审计体系

因此它不是简单的 Copilot 类产品，而更接近：

> 面向企业软件交付场景的自治型执行代理

---

## 六、为什么这种方向重要

在企业落地 AI 时，真正的瓶颈通常不是模型本身不够强，而是：

- 缺少上下文
- 缺少工具能力
- 缺少执行链路
- 缺少治理边界
- 缺少可观测性与审计能力

Harness Agent 这条路线解决的是“AI 从演示走向生产”的关键问题：

> 如何把一个会说话的大模型，变成一个能在真实企业流程中稳定、安全、可控地完成任务的 Agent。

这也是为什么越来越多团队认为：

> 模型会越来越成为可替换的引擎，而 Harness / Orchestration / Runtime 才是智能体产品真正的护城河。

---

## 七、最终总结

如果要对 `Harness Agent` 的“复合模型”做一个准确定位，可以这样表述：

> 它不是单一的大模型算法，而是一种面向企业 DevOps 场景的复合式智能体架构。其核心方向是将大模型、上下文增强、工具调用、工作流编排和企业级安全治理整合为统一的 Agent 运行系统。

换句话说，Harness Agent 代表的不是“更强的一个模型”，而是：

**让多个 AI 能力层协同工作，使模型真正具备在企业环境中执行复杂任务的能力。**
