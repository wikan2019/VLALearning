# AI 技术进展学习文档库

> 全面系统地学习大语言模型（LLM）、视觉语言模型（VLM）、视觉-语言-动作模型（VLA）的技术演进

## 📚 项目简介

本项目汇总了截至 2026 年 2 月，人工智能领域三大关键技术栈的核心进展：

- **LLM (Large Language Models)** - 大语言模型的基础架构与效率优化
- **VLM (Vision-Language Models)** - 多模态理解与视觉推理
- **VLA (Vision-Language-Action Models)** - 具身智能与机器人控制

每个技术栈都包含详细的技术文档，从基础原理到前沿应用，适合研究者、工程师和学生系统学习。

---

## 🗂️ 文档结构

```
.
├── LLM_Advances/          # Part 1-2: 大语言模型技术进展 (10篇)
├── VLM_Advances/          # Part 3 + 补充: 视觉语言模型技术进展 (16篇)
├── VLA_Advances/          # Part 4-5: 视觉-语言-动作模型技术进展 (11篇)
├── Prompt_log.md          # 文档生成历史与迭代记录
└── README.md              # 本文件
```

---

## 🚀 快速导航

### Part 1-2: LLM 技术进展 (10篇)

👉 [**进入 LLM 技术文档目录**](./LLM_Advances/README.md)

#### 基础架构演进
1. [LayerNorm → RMSNorm](./LLM_Advances/01_LayerNorm_to_RMSNorm.md) - 归一化层的效率革命
2. [APE → RoPE](./LLM_Advances/02_APE_to_RoPE.md) - 位置编码的长度外推
3. [FFN → Gated FFN](./LLM_Advances/03_FFN_to_GatedFFN.md) - 门控机制与 SwiGLU
4. [Attention 稳定性设计](./LLM_Advances/04_Attention_Stability.md) - QK-Norm 与训练稳定性
5. [Register/Memory Token](./LLM_Advances/05_Register_Memory_Token.md) - StreamLLM 的无限上下文

#### 效率与 Scaling
6. [MHA → MQA/GQA](./LLM_Advances/06_MQA_GQA.md) - KV Cache 压缩与推理加速
7. [RoPE → ALiBi](./LLM_Advances/07_ALiBi.md) - 线性偏置的外推能力
8. [FlashAttention](./LLM_Advances/08_FlashAttention.md) - IO 感知的精确注意力算法
9. [Sparse MoE](./LLM_Advances/09_MoE.md) - 专家混合与稀疏激活
10. [DPO](./LLM_Advances/10_DPO.md) - 无需强化学习的偏好对齐

---

### Part 3 + 补充: VLM 技术进展 (16篇)

👉 [**进入 VLM 技术文档目录**](./VLM_Advances/README.md)

#### 核心架构与训练
11. [Connector 连接器](./VLM_Advances/11_Connector.md) - Q-Former → MLP 的简化之路
12. [AnyRes 动态分辨率](./VLM_Advances/12_Resolution_AnyRes.md) - 高清图像的细粒度理解
13. [纯解码器架构](./VLM_Advances/13_Decoder_Only_VLM.md) - 移除 Vision Encoder 的极简设计
14. [视觉指令微调](./VLM_Advances/14_Visual_Instruction_Tuning.md) - 从对齐到指令理解
15. [视觉思维链](./VLM_Advances/15_Visual_CoT.md) - 多步推理与逻辑分析
16. [SOTA 模型对比](./VLM_Advances/16_SOTA_Models_Review.md) - Qwen3 vs Gemini 3 vs GPT-5.2

#### 补充专题
- [S1. 视觉编码器演进](./VLM_Advances/VLM_S1_Vision_Encoder_Evolution.md) - CLIP → SigLIP 2
- [S2. 视觉接地](./VLM_Advances/VLM_S2_Visual_Grounding.md) - 从"看图说话"到"指哪说哪"
- [S3. 视频理解](./VLM_Advances/VLM_S3_Video_Understanding.md) - 从单帧到时间流
- [S4. 多模态幻觉](./VLM_Advances/VLM_S4_Hallucination.md) - VLM 最致命的缺陷
- [S5. 统一视觉模型](./VLM_Advances/VLM_S5_Unified_Vision.md) - 理解+生成的统一架构

---

### Part 4-5: VLA 技术进展 (11篇)

👉 [**进入 VLA 技术文档目录**](./VLA_Advances/README.md)

#### 基础与架构
17. [VLA 概述](./VLA_Advances/17_VLA_Overview.md) - 从感知到行动的范式转变
18. [RT-1 到 RT-2](./VLA_Advances/18_RT1_RT2_Foundation.md) - VLA 的诞生
19. [VLA 架构深度解析](./VLA_Advances/19_VLA_Architecture.md) - 五大架构范式
20. [动作表示](./VLA_Advances/20_Action_Tokenization.md) - Token 化到扩散生成
21. [OpenVLA](./VLA_Advances/21_OpenVLA.md) - 开源 VLA 的里程碑
22. [π0 系列](./VLA_Advances/22_Pi0_Generalist_Policy.md) - 通用机器人基座模型

#### 关键技术
23. [跨具身泛化](./VLA_Advances/23_Data_Cross_Embodiment.md) - Open X-Embodiment 数据集
24. [世界模型与规划](./VLA_Advances/24_World_Models_Planning.md) - 预测式控制
25. [Sim-to-Real](./VLA_Advances/25_Sim2Real_Transfer.md) - 从仿真到真实世界
26. [边缘部署](./VLA_Advances/26_Edge_Deployment.md) - 实时控制与推理优化
27. [SOTA 模型对比](./VLA_Advances/27_VLA_SOTA_Review.md) - RT-2-X vs OpenVLA vs π0.5

---

## 🔗 技术栈关系图

```
LLM (纯文本理解)
 ↓
VLM (视觉 + 文本理解)
 ↓
VLA (视觉 + 文本 + 动作执行)
```

- **LLM** 奠定了语言理解与生成的基础
- **VLM** 将视觉感知与语言理解连接
- **VLA** 在 VLM 基础上增加了动作输出，实现具身智能

文档间有大量技术复用和引用关系，建议按顺序学习。

---

## 📖 使用建议

### 适用人群
- 🎓 **AI 研究者**：了解前沿技术演进脉络
- 💻 **工程师**：掌握关键技术的实现原理
- 📚 **学生**：系统学习多模态 AI 知识体系

### 学习路径

#### 初学者路径
1. 从 LLM 基础架构开始（Part 1: 01-05）
2. 学习 VLM 核心技术（11-16）
3. 了解 VLA 基础概念（17-22）

#### 进阶路径
1. 深入效率优化技术（06-10）
2. 研究 VLM 补充专题（S1-S5）
3. 掌握 VLA 关键技术（23-27）

#### 实践路径
- 关注带有"代表模型"的章节，了解工业应用
- 重点学习 FlashAttention、MoE、DPO 等可直接应用的技术
- 研究 OpenVLA、π0 等开源模型的实现

---

## 🔄 更新日志

查看 [Prompt_log.md](./Prompt_log.md) 了解文档的生成历史和迭代过程。

---

## 📝 文档特点

- ✅ **系统性**：覆盖从基础到前沿的完整技术栈
- ✅ **深度性**：每篇文档都包含原理、公式、代码和案例
- ✅ **时效性**：截至 2026 年 2 月的最新技术进展
- ✅ **关联性**：文档间有清晰的技术传承关系
- ✅ **实用性**：标注了代表模型和工业应用案例

---

## 🤝 贡献

本文档库持续更新中。如发现错误或希望补充内容，欢迎提交 Issue 或 Pull Request。

---

## 📄 许可

本项目仅供学习和研究使用。

---

**Happy Learning! 🚀**
