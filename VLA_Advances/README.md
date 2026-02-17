# VLA 技术进展学习文档索引

> 🤖 Vision-Language-Action Models (VLA) 截止 2026 年 2 月的 11 项关键技术演进

[⬅️ 返回主目录](../README.md) | [📖 LLM 基础](../LLM_Advances/README.md) | [🎨 VLM 进展](../VLM_Advances/README.md)

---

## 📑 目录概览

VLA 是 VLM 的自然延伸——**VLM 让模型"看懂世界"，VLA 则让模型"动手改变世界"**。

本文档汇总了视觉-语言-动作模型领域的核心技术演进，分为两大部分：

- **Part 4**: VLA 基础与架构 (文档 17-22)
- **Part 5**: VLA 关键技术 (文档 23-27)

---

## Part 4: 视觉-语言-动作模型 (VLA Foundation)

### 17. [VLA 概述：从感知到行动的范式转变](./17_VLA_Overview.md)
*   **演进点**：机器人控制从手工策略 → 学习策略 → 基座模型策略
*   **核心价值**：理解 VLA 在 LLM → VLM → VLA 技术栈中的位置，以及它为什么是具身智能 (Embodied AI) 的核心
*   **关键概念**：感知-决策-执行闭环、基座模型范式
*   **关联阅读**：先阅读 [VLM 进展](../VLM_Advances/README.md)（VLA 的基础）

### 18. [RT-1 到 RT-2：VLA 的诞生](./18_RT1_RT2_Foundation.md)
*   **演进点**：从专用 Robotics Transformer 到 VLM 驱动的 VLA
*   **核心价值**：RT-2 首次证明 **互联网规模的视觉-语言知识可以直接迁移到机器人控制**，开创了 VLA 这一模型类别
*   **代表模型**：RT-1, RT-2, RT-2-X
*   **关联阅读**：→ [19. VLA 架构](./19_VLA_Architecture.md)（RT-2 的架构分析）

### 19. [VLA 架构深度解析：如何把 VLM 变成机器人控制器](./19_VLA_Architecture.md)
*   **演进点**：VLA 的五大架构范式——自回归、扩散、强化、混合、专用
*   **核心价值**：深入拆解 VLA 的输入-处理-输出流水线，理解视觉编码器、语言模型、动作解码头三大模块的协作
*   **代表模型**：RT-2, OpenVLA, Octo, π0
*   **关联阅读**：→ [20. 动作表示](./20_Action_Tokenization.md)（动作解码头的核心）

### 20. [动作表示：从离散 Token 到扩散生成](./20_Action_Tokenization.md)
*   **演进点**：动作空间的编码方式——离散化 Bin → FAST 频域压缩 → 扩散去噪
*   **核心价值**：动作表示是 VLA 区别于 VLM 的核心差异，不同编码方式直接决定了模型的精度、速度和泛化能力
*   **代表方法**：RT-2 Binning, FAST, Diffusion Policy, Action Chunking
*   **关联阅读**：→ [22. π0 系列](./22_Pi0_Generalist_Policy.md)（FAST 的实际应用）

### 21. [OpenVLA：开源 VLA 的里程碑](./21_OpenVLA.md)
*   **演进点**：从闭源 55B RT-2-X 到开源 7B OpenVLA，性能反超 16.5%
*   **核心价值**：首个可在消费级 GPU 上微调的开源 VLA，极大降低了机器人基座模型的研究门槛
*   **代表模型**：OpenVLA, OpenVLA-OFT
*   **关联阅读**：→ [23. 跨具身数据](./23_Data_Cross_Embodiment.md)（OpenVLA 的训练数据）

### 22. [π0 系列：通用机器人基座模型](./22_Pi0_Generalist_Policy.md)
*   **演进点**：从单任务策略 → 多任务泛化 → 开放世界泛化 (π0 → π0.5)
*   **核心价值**：Physical Intelligence 的 π0 系列是目前最接近"通用机器人大脑"的模型，能在从未见过的家庭环境中完成复杂长程任务
*   **代表模型**：π0, π0-FAST, π0.5
*   **关联阅读**：→ [24. 世界模型](./24_World_Models_Planning.md), → [27. SOTA 对比](./27_VLA_SOTA_Review.md)

## Part 5: VLA 关键技术 (VLA Key Technologies)

### 23. [数据与跨具身泛化：Open X-Embodiment](./23_Data_Cross_Embodiment.md)
*   **演进点**：从单机器人数据集到跨具身统一数据集
*   **核心价值**：Open X-Embodiment 汇聚 22 个机器人平台的百万级演示数据，证明了**不同机器人的数据可以互相受益**，奠定了 VLA 的"ImageNet 时刻"
*   **代表数据集**：Open X-Embodiment, DROID, BridgeData V2
*   **关联阅读**：→ [21. OpenVLA](./21_OpenVLA.md)（使用该数据集训练的模型）

### 24. [世界模型与机器人规划](./24_World_Models_Planning.md)
*   **演进点**：从反应式控制到预测式规划——让机器人"先在脑中预演"
*   **核心价值**：视频世界模型让 VLA 可以预测未来画面和奖励，在虚拟空间中规划最优动作序列
*   **代表方法**：UniPi, iVideoGPT, World-VLA-Loop, ViPRA
*   **关联阅读**：→ [VLM/S3. 视频理解](../VLM_Advances/VLM_S3_Video_Understanding.md), → [VLM/S5. 统一视觉模型](../VLM_Advances/VLM_S5_Unified_Vision.md)

### 25. [Sim-to-Real：从仿真到真实世界](./25_Sim2Real_Transfer.md)
*   **演进点**：解决 VLA 最大瓶颈——真实数据稀缺和安全风险
*   **核心价值**：通过仿真训练 + 域随机化 + 师生蒸馏，实现从虚拟环境到真实机器人的零样本迁移
*   **代表方法**：Sim2Real-VLA, Isaac Lab, Domain Randomization
*   **关联阅读**：→ [23. 跨具身数据](./23_Data_Cross_Embodiment.md)（仿真数据与真实数据的结合）

### 26. [VLA 部署：实时控制与边缘推理](./26_Edge_Deployment.md)
*   **演进点**：7B 参数的模型如何在机器人上实现 30Hz 实时控制？
*   **核心价值**：从量化、异步推理到轻量化模型，解决 VLA 从实验室到真实机器人的"最后一公里"问题
*   **代表方法**：EdgeVLA, LiteVLA, VLASH, BLURR, π0-FAST@30Hz
*   **关联阅读**：→ [LLM/06. GQA](../LLM_Advances/06_MQA_GQA.md), → [LLM/08. FlashAttention](../LLM_Advances/08_FlashAttention.md)（推理优化技术）

### 27. [VLA SOTA 模型全景对比 (截至 2026.02)](./27_VLA_SOTA_Review.md)
*   **演进点**：当前最强 VLA 模型的横向对比与选型指南
*   **核心价值**：从架构、数据、性能、开源状态等维度全面对比 RT-2-X, OpenVLA, π0.5, Octo 等模型
*   **代表模型**：RT-2-X, OpenVLA-OFT, π0.5, Octo, GR-2
*   **关联阅读**：→ [VLM/16. SOTA 模型](../VLM_Advances/16_SOTA_Models_Review.md)（VLM 基座的对比）

---

## 🔗 技术关联图

```
基础架构层 (Part 4):
17. VLA 概述 → 18. RT-1/RT-2 → 19. VLA 架构
                                     ↓
                            20. 动作表示 (关键差异)
                                     ↓
                    21. OpenVLA ←→ 22. π0 系列

关键技术层 (Part 5):
23. 跨具身数据 (训练基础)
   ↓
24. 世界模型 (预测规划) ←→ 25. Sim-to-Real (数据扩充)
   ↓
26. 边缘部署 (实时控制)
   ↓
27. SOTA 对比 (选型指南)
```

---

## 🔄 VLA 与 VLM 的核心差异

| 维度 | VLM | VLA |
|------|-----|-----|
| **输出** | 文本描述 | 连续动作（位置、姿态、抓握力） |
| **评估** | VQA、Caption 准确率 | 任务成功率、安全性 |
| **训练数据** | 图文对 (Web 规模) | 机器人演示 (千-百万级) |
| **推理延迟** | 秒级可接受 | 必须 <50ms (20-30Hz) |
| **失败代价** | 回答错误 | 可能损坏设备或伤人 |
| **关键技术** | Connector, AnyRes | 动作表示, Grounding, 世界模型 |

**核心技术复用**：
- VLA 继承 VLM 的视觉编码器 ([VLM/S1](../VLM_Advances/VLM_S1_Vision_Encoder_Evolution.md))
- VLA 需要 Grounding 能力 ([VLM/S2](../VLM_Advances/VLM_S2_Visual_Grounding.md))
- VLA 的世界模型基于视频生成 ([VLM/S3](../VLM_Advances/VLM_S3_Video_Understanding.md), [VLM/S5](../VLM_Advances/VLM_S5_Unified_Vision.md))

---

## 📚 学习建议

### 🎯 新手路径
1. **前置知识**：先完成 [LLM](../LLM_Advances/README.md) 和 [VLM](../VLM_Advances/README.md) 学习
2. **VLA 基础**：17 → 18 → 19（理解 VLA 是什么）
3. **核心差异**：20（动作表示）+ [VLM/S2](../VLM_Advances/VLM_S2_Visual_Grounding.md)（空间定位）
4. **前沿模型**：21/22 → 27（了解当前能力边界）

### 🚀 进阶路径
1. **架构深度**：19 → 20（五大架构范式 + 动作表示对比）
2. **数据与训练**：23 → 25（跨具身泛化 + Sim-to-Real）
3. **规划与推理**：24（世界模型）+ [VLM/S3](../VLM_Advances/VLM_S3_Video_Understanding.md)（视频预测）

### 💡 实践路径
- **部署 VLA**：26 → 21（边缘推理 + OpenVLA 开源实现）
- **提升性能**：23 → 24 → 25（数据 → 规划 → 泛化）
- **选型决策**：27 → 对比表格（根据场景选择模型）

### 🔬 研究路径
- **开源入门**：21. OpenVLA（代码可复现）
- **前沿探索**：22. π0.5（通用能力）, 24. 世界模型（未来方向）
- **关键挑战**：25. Sim-to-Real（Sim2Real Gap）, 26. 边缘部署（实时性）

---

## 🔄 与其他技术栈的关系

```
LLM (语言理解)
 ↓
VLM (视觉 + 语言理解)
 ↓
VLA (视觉 + 语言 + 动作执行) ← 你在这里
```

- **← LLM 技术栈**：VLA 的语言模型基座
  - [返回 LLM 文档目录](../LLM_Advances/README.md)

- **← VLM 技术栈**：VLA 的视觉理解基础
  - [返回 VLM 文档目录](../VLM_Advances/README.md)

**必读 VLM 专题**（对 VLA 至关重要）：
- [S1. 视觉编码器](../VLM_Advances/VLM_S1_Vision_Encoder_Evolution.md) - VLA 的"眼睛"
- [S2. 视觉接地](../VLM_Advances/VLM_S2_Visual_Grounding.md) - 从理解到定位
- [S3. 视频理解](../VLM_Advances/VLM_S3_Video_Understanding.md) - 时间序列感知
- [S5. 统一视觉模型](../VLM_Advances/VLM_S5_Unified_Vision.md) - 世界模型的基础

---

## 📝 文档说明

- **Part 4（17-22）**：VLA 的基础架构与代表模型
- **Part 5（23-27）**：数据、规划、部署等关键技术
- 所有文档包含：原理解析、数学公式、代码示例、应用案例
- 大量跨文档引用，建议按技术栈顺序学习（LLM → VLM → VLA）

---

## 🎯 VLA 的未来方向

1. **更长的规划视野**：从单步反应到多步规划（世界模型）
2. **更强的泛化能力**：从特定场景到开放世界（π0.5 方向）
3. **更快的推理速度**：从云端到边缘实时控制（<10ms）
4. **更少的数据需求**：从百万演示到少样本迁移（Sim-to-Real）
5. **多模态融合**：视觉 + 触觉 + 听觉的统一表示

---

[⬅️ 返回主目录](../README.md) | [📖 LLM 基础](../LLM_Advances/README.md) | [🎨 VLM 进展](../VLM_Advances/README.md)
