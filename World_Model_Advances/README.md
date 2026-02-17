# World Model 技术进展学习文档索引 (Index of World Model Advances)

本文档汇总了截止 2026 年 2 月 World Model（世界模型）领域的 **10 项**关键技术演进。

世界模型是 AI 从"理解世界"到"预测世界"的关键一步——它让 AI 拥有了"想象力"，能在脑中模拟未来。

---

## Part 6: 世界模型基础 (World Model Foundation)

### WM-01. [世界模型概述：AI 的想象力引擎](./WM_01_Overview.md)
*   **核心概念**：什么是世界模型？它与 LLM、VLM、VLA 的关系是什么？
*   **核心价值**：世界模型是 AI 实现"先想后做"的核心——让 AI 不用真正尝试就能预测行动的后果。
*   **关键概念**：内部模拟器、状态预测、物理推理。

### WM-02. [Dreamer 系列：在想象中学习的 RL 智能体](./WM_02_Dreamer.md)
*   **演进点**：从 Model-Free RL 到 Model-Based RL 世界模型。
*   **核心价值**：Dreamer V3 首次在 150+ 任务上用单一配置超越专用方法，并在 Minecraft 中从零收集钻石。TD-MPC2 将世界模型扩展到 317M 参数跨域智能体。
*   **代表模型**：Dreamer V1/V2/V3, TD-MPC2, MuDreamer.

### WM-03. [JEPA：LeCun 的世界模型愿景](./WM_03_JEPA.md)
*   **演进点**：从像素预测到潜在空间预测——"预测本质而非表象"。
*   **核心价值**：Yann LeCun 提出的 JEPA 架构代表了一种非生成式的世界模型路线，V-JEPA 2 (1.2B) 实现了零样本机器人规划。
*   **代表模型**：I-JEPA, V-JEPA, V-JEPA 2.

## Part 7: 世界模型应用 (World Model Applications)

### WM-04. [视频世界模型：Sora 的启示](./WM_04_Video_World_Models.md)
*   **演进点**：视频生成模型作为物理世界模拟器。
*   **核心价值**：Sora 证明了扩展视频生成模型可以涌现出对物理世界的初步理解。视频世界模型成为连接虚拟与现实的桥梁。
*   **代表模型**：Sora, WorldDreamer, iVideoGPT, Vid2World.

### WM-05. [交互式世界生成：Genie 系列](./WM_05_Genie.md)
*   **演进点**：从被动视频生成 → 可实时交互的虚拟世界。
*   **核心价值**：Google DeepMind 的 Genie 系列首次实现了"从一张图生成可玩的 3D 世界"，为 AI 智能体训练提供了无限的仿真环境。
*   **代表模型**：Genie 1 (11B), Genie 2, Genie 3 (24fps/720p 实时).

### WM-06. [自动驾驶世界模型](./WM_06_Autonomous_Driving.md)
*   **演进点**：从规则驱动的仿真 → 学习型世界模型生成驾驶场景。
*   **核心价值**：GAIA、DriveDreamer 等模型可以生成逼真的多视角驾驶视频，解决自动驾驶中极端场景（corner case）数据稀缺的问题。
*   **代表模型**：GAIA-1/2, DriveDreamer, UniSim, OccWorld.

### WM-07. [机器人世界模型与规划](./WM_07_Robot_World_Models.md)
*   **演进点**：从反应式 VLA → 基于世界模型的规划式 VLA。
*   **核心价值**：世界模型让机器人能够"先在脑中预演，再在现实中执行"，实现更高效的长程任务规划。
*   **代表方法**：UniPi, World-VLA-Loop, iVideoGPT, ViPRA.

## Part 8: 世界模型工程与评测 (Engineering & Evaluation)

### WM-08. [NVIDIA Cosmos：世界基座模型平台](./WM_08_Cosmos.md)
*   **演进点**：从学术研究到工业级世界模型平台。
*   **核心价值**：NVIDIA Cosmos 提供了完整的"预测-转换-推理"三合一平台，是首个面向物理 AI 的开源世界基座模型。
*   **平台组件**：Cosmos Predict, Cosmos Transfer, Cosmos Reason.

### WM-09. [世界模型评测：物理推理基准](./WM_09_Evaluation.md)
*   **演进点**：从"好不好看"到"懂不懂物理"——评测标准的根本转变。
*   **核心价值**：WorldModelBench、PhysBench 等基准首次系统评测世界模型的物理一致性，而非仅评估视频质量。
*   **代表基准**：WorldModelBench, PhysBench, PhysReason, SmallWorld.

### WM-10. [SOTA 世界模型全景对比（截至 2026.02）](./WM_10_SOTA_Review.md)
*   **演进点**：当前最强世界模型的横向对比与未来展望。
*   **核心价值**：从架构、应用领域、开源状态、物理理解等维度全面对比各类世界模型。
*   **代表模型**：Dreamer V3, V-JEPA 2, Genie 3, GAIA-2, Cosmos, World-VLA-Loop.
