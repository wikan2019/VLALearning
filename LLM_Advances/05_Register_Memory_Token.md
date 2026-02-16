# 05. 引入 Register / Memory Token

在 Transformer 的最新研究中，引入特殊功能的 Token (Tokens with special purposes) 成为一种提升性能的有效手段。其中最具代表性的是 **Register Tokens**（主要在 Vision Transformer 中发现）和 **Memory Tokens**（在长文本 LLM 中应用）。

## 1. Register Tokens (Vision Transformer)
这一发现来自 Meta 的论文 *"Vision Transformers Need Registers"* (2023)。

### 现象：Artifacts in Attention Maps
研究人员发现，在大规模 Vision Transformer (DINOv2) 中，Attention Map 会出现奇怪的高亮区域（Artifacts），通常出现在背景等无关区域。这些区域的 token 并没有实际语义信息，但却拥有极高的 attention 权重。

### 原因分析
模型需要在某些地方“存储”全局信息，但如果没有专门的存储空间，它就会被迫征用（Hijack）一些无关的背景 patch token 来充当“临时寄存器”。这破坏了这些 patch 原本的语义表示。

### 解决方案
显式地引入 **Register Tokens**：
- 在输入序列中追加 $n$ 个（通常是 4-8 个）可学习的 embedding 向量。
- 这些 token **不对应任何图像 patch**，仅作为“全局信息暂存区”。
- 在输出时，直接丢弃这些 token。

### 效果
- Attention Map 变得非常干净，Artifacts 消失，对象定位更准确。
- 下游任务（分类、分割）性能显著提升。
- 这一思想证明了 Transformer 需要“工作内存”（Working Memory）。

## 2. Memory Tokens (LLMs)
在 LLM 领域，类似的思想演变为 **Memory Tokens** 或 **Summary Tokens**，旨在解决长上下文遗忘和计算效率问题。

### Transformer-XL & Compressive Transformer
早期的尝试（Transformer-XL）引入了 **Segment-Level Recurrence**，将上一段的 Hidden States 缓存下来作为 Memory。

### 现代 Memory Token 设计 (如 MemGPT, RMT)
**Recurrent Memory Transformer (RMT)**：
- 在输入序列的开头或结尾添加特殊的 Memory Tokens（如 `[MEM]`）。
- 模型在处理完一段文本后，会将关键信息“压缩”写入到这些 Memory Tokens 中。
- 在处理下一段文本时，这些 Memory Toknes 会作为输入的一部分传入，从而实现跨段落的信息传递。

这实际上将 Transformer 变成了一种 **RNN**（递归神经网络），使其理论上能够处理无限长度的序列，而无需随长度线性增长的计算量。

### StreamLLM (Attention Sinks)
另一种相关的发现是 **Attention Sink** 现象：
- 模型倾向于将大量的 Attention 分数分配给序列开头的几个 token（通常是 `<s>` 或前几个词）。
- 即使这些词没有实际意义，它们也充当了“锚点”（Sink），吸收多余的 Attention 权重。
-如果强行截断开头，模型会崩盘。
- **StreamingLLM** 利用这一特性，通过保留“开头 token (Sinks)” + “最近 token (Local)”，实现了无限长度的流式推理。

## 3. 总结对比

| 概念 | Register Tokens | Memory Tokens / RMT | Attention Sinks |
| :--- | :--- | :--- | :--- |
| **领域** | Vision (ViT) | Language (LLM) | Language (LLM Inference) |
| **解决问题** | Attention Artifacts (高亮噪声) | 无限长上下文 (Long Context) | 流式推理稳定性 |
| **机制** | 额外的空 token 用于存储全局信息 | 读写机制，跨段落传递状态 | 保留开头 token 作为“锚点” |
| **启示** | 模型需要“笔记本”来暂存计算过程中的中间状态 | 显式内存比隐式上下文更高效 | 某些 token 承担了系统级功能而非语义功能 |

**核心观点**：Transformer 不仅仅是模式匹配器，它在进行复杂的计算。引入 Register/Memory Token 实际上是给了模型**独立的“工作内存” (Scratchpad)**，将其与**“输入数据” (Input Data)** 解耦，这是通向更强推理能力的重要一步。
