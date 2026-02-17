# LLM 技术进展学习文档索引

> 📖 Large Language Models (LLM) 过去 5 年的 10 项关键技术演进

[⬅️ 返回主目录](../README.md)

---

## 📑 目录概览

本文档汇总了大语言模型领域的核心技术演进，分为两大部分：

- **Part 1**: 基础架构演进 (文档 01-05)
- **Part 2**: 效率与 Scaling 进阶 (文档 06-10)

---

## Part 1: 基础架构演进 (Fundamental Advances)

### 1. [LayerNorm → RMSNorm](./01_LayerNorm_to_RMSNorm.md)
*   **演进点**：从 Layer Normalization 到 Root Mean Square Normalization
*   **核心价值**：去除了 Mean 统计量，仅保留缩放不变性。**计算效率更高**，且在深层网络训练中具有**更好的梯度稳定性**
*   **代表模型**：LLaMA, Gopher, Chinchilla
*   **关联阅读**：→ [04. Attention 稳定性设计](./04_Attention_Stability.md)

### 2. [APE → RoPE](./02_APE_to_RoPE.md)
*   **演进点**：从绝对位置嵌入 (Absolute Positional Embedding) 到旋转位置编码 (Rotary Positional Embedding)
*   **核心价值**：通过复数旋转引入**相对位置信息**，具有更好的**长度外推性** (Extrapolation)
*   **代表模型**：LLaMA, PaLM, GLM, Qwen
*   **关联阅读**：→ [07. ALiBi](./07_ALiBi.md)（另一种位置编码方案）

### 3. [FFN → Gated FFN (SwiGLU)](./03_FFN_to_GatedFFN.md)
*   **演进点**：从标准 ReLU/GELU FFN 到 Gated Linear Units (GLU) 变体
*   **核心价值**：引入**门控机制** (Gating)，增加了模型的非线性表达能力，通常能带来显著的 Performance 提升
*   **代表模型**：PaLM, LLaMA, Mixtral
*   **关联阅读**：→ [09. Sparse MoE](./09_MoE.md)（在 FFN 层引入专家网络）

### 4. [Attention → QK-Norm + 稳定性设计](./04_Attention_Stability.md)
*   **演进点**：应对 Attention Logits 爆炸导致的训练不稳定
*   **核心价值**：通过 **Query-Key Normalization** 或 **Logits Scaling**，防止 Attention Score 过大，稳定训练梯度
*   **代表模型**：ViT-22B, GLM-130B
*   **关联阅读**：→ [01. RMSNorm](./01_LayerNorm_to_RMSNorm.md)（归一化技术）

### 5. [Register / Memory Tokens + StreamLLM](./05_Register_Memory_Token.md)
*   **演进点**：利用额外的 Token 来存储全局信息或作为 "Attention Sink"
*   **核心价值**：
    *   **Register Token**：消除 Vision Transformer 中的伪影
    *   **StreamLLM**：利用 **Attention Sink (首 Token)** 机制，实现**无限长度的流式推理**，防止上下文窗口滑动导致的崩盘
*   **代表模型**：DINOv2, StreamLLM (Llama-2 optimized)
*   **关联阅读**：→ [08. FlashAttention](./08_FlashAttention.md)（长上下文的显存优化）

---

## Part 2: 效率与 Scaling 进阶 (Efficiency & Scaling)

### 6. [MHA → MQA / GQA (Inference Acceleration)](./06_MQA_GQA.md)
*   **演进点**：从 Multi-Head Attention 到 Multi-Query / Grouped-Query Attention
*   **核心价值**：大幅减少 **KV Cache** 的显存占用（最高压缩 32 倍），解决 Memory Bandwidth 瓶颈，从 Memory Bound 转为 Compute Bound
*   **代表模型**：LLaMA-2 (70B), LLaMA-3, Mistral
*   **关联阅读**：→ [08. FlashAttention](./08_FlashAttention.md)（另一维度的显存优化）

### 7. [RoPE → ALiBi (Positional Extrapolation)](./07_ALiBi.md)
*   **演进点**：通过线性偏置 (Linear Bias) 替代显式位置编码
*   **核心价值**：具有极强的**长度外推能力**，在短序列上训练的模型可直接推理长序列
*   **代表模型**：MPT-7B, BLOOM
*   **关联阅读**：→ [02. RoPE](./02_APE_to_RoPE.md)（对比两种位置编码方案）

### 8. [Standard Attention → FlashAttention (IO-Aware)](./08_FlashAttention.md)
*   **演进点**：IO 感知的精确注意力加速算法
*   **核心价值**：通过 **Tiling (分块)** 和 **Recomputation**，将显存复杂度从 $O(N^2)$ 降为 **$O(N)$**，并极大减少 HBM 访问。是长文本训练的基石
*   **代表模型**：All Modern LLMs (DeepSpeed, Megatron, Pytorch 2.0)
*   **关联阅读**：→ [05. StreamLLM](./05_Register_Memory_Token.md), [06. GQA](./06_MQA_GQA.md)（显存优化组合拳）

### 9. [Dense → Sparse MoE (Mixture of Experts)](./09_MoE.md)
*   **演进点**：从稠密网络到稀疏专家网络
*   **核心价值**：**稀疏激活 (Sparse Activation)**。在保持推理计算量 (FLOPs) 不变的情况下，极大增加模型参数量 (Capacity)。需解决 **Load Balancing** 和 **Stability** 问题
*   **代表模型**：GPT-4, Mixtral 8x7B, DeepSeekMoE
*   **关联阅读**：→ [03. Gated FFN](./03_FFN_to_GatedFFN.md)（MoE 的基础模块）

### 10. [RLHF (PPO) → DPO (Direct Preference Optimization)](./10_DPO.md)
*   **演进点**：从复杂的 PPO 强化学习到直接偏好优化
*   **核心价值**：**无需 Reward Model**，无需 Sampling。通过数学转换将 RL 问题变为稳定的监督学习问题。但在复杂推理上可能仍需 Online RL 辅助
*   **代表模型**：Llama-3-Instruct, Mistral-Instruct, Zephyr
*   **关联阅读**：→ [VLM/14. Visual Instruction Tuning](../VLM_Advances/14_Visual_Instruction_Tuning.md)（多模态的指令微调）

---

## 🔗 技术关联图

```
基础架构层:
RMSNorm ←→ QK-Norm (稳定性)
   ↓
RoPE / ALiBi (位置编码)
   ↓
Gated FFN (非线性表达)

效率优化层:
GQA (KV Cache) ←→ FlashAttention (显存) ←→ StreamLLM (无限上下文)
   ↓
MoE (参数扩展)
   ↓
DPO (对齐优化)
```

---

## 📚 学习建议

### 🎯 新手路径
1. 从第 1-3 篇开始，理解 Transformer 基础组件的改进
2. 学习第 4-5 篇，掌握稳定性设计
3. 跳过 6-9，直接看第 10 篇了解对齐方法

### 🚀 进阶路径
1. 重点学习第 6、8 篇（推理优化的核心）
2. 深入研究第 9 篇 MoE（大模型 Scaling 的关键）
3. 对比第 2 和第 7 篇，理解位置编码的权衡

### 💡 实践路径
- **训练优化**：01 → 04 → 08 → 09
- **推理优化**：06 → 08 → 05
- **长文本处理**：02/07 → 05 → 08

---

## 🔄 与其他技术栈的关系

- **→ VLM 技术栈**：LLM 是 VLM 的文本理解基座，VLM 在 LLM 上增加视觉编码器
  - [进入 VLM 文档目录](../VLM_Advances/README.md)

- **→ VLA 技术栈**：VLA 在 VLM 基础上增加动作输出能力
  - [进入 VLA 文档目录](../VLA_Advances/README.md)

---

## 📝 文档说明

- 所有文档包含：原理解析、数学公式、代码示例、应用案例
- 标注了技术的代表模型和工业应用
- 包含跨文档的交叉引用，便于串联学习

---

[⬅️ 返回主目录](../README.md) | [➡️ 前往 VLM 文档](../VLM_Advances/README.md)
