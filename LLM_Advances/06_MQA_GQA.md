# 06. MHA -> MQA / GQA (Inference Acceleration)

在 LLM 推理阶段，**KV Cache** 的显存占用和加载带宽是主要的性能瓶颈。为了解决这个问题，注意力机制从标准的 **Multi-Head Attention (MHA)** 演进到了 **Multi-Query Attention (MQA)** 和 **Grouped-Query Attention (GQA)**。

## 1. 背景：MHA 的瓶颈
标准的 Multi-Head Attention (MHA) 中，每个 Head 都有自己独立的 Query, Key, Value 矩阵。
假设有 $H$ 个 Head，每个 Head 维度为 $D$：
- Query Heads: $H$ 个
- Key Heads: $H$ 个
- Value Heads: $H$ 个

**问题**：
在生成过程中，KV Cache 的大小为 $2 \times H \times L \times D$。对于长上下文（如 100k+ tokens）的大模型，KV Cache 会迅速占满显存（例如 70B 模型在长文本下 KV Cache 可达几十 GB）。此外，加载这么大的 KV Cache 会导致**Memory Bandwidth Bound**（内存带宽受限），严重拖慢推理速度。

## 2. 演进 I：Multi-Query Attention (MQA)
Noam Shazeer (2019) 提出了 MQA。

### 核心机制
- **多 Query**：保持 $H$ 个 Query Heads。
- **单 KV**：所有 Query Heads **共享同一个** Key Head 和 Value Head。

**KV Cache 减少**：从 $2H$ 降为 $2$。压缩比为 $H$ 倍（例如 32 倍）。

**效果**：
- **优点**：推理速度极大提升，显存占用极大降低。
- **缺点**：模型性能（Accuracy/Perplexity）会有所下降，且训练时可能不稳定。

## 3. 演进 II：Grouped-Query Attention (GQA)
GQA (Ainslie et al., 2023) 是 MHA 和 MQA 的折中方案，被 LLaMA-2/3、Mistral 等主流模型采用。

### 核心机制
将 Query Heads 分成 $G$ 个组（Group），每组内的 Query Heads 共享一对 KV Heads。
- Query Heads: $H$ 个
- Key/Value Heads: $G$ 个 ($1 < G < H$)

通常 $G$ 取值为 8，而 $H$ 可能为 64。KV Cache 压缩比为 $H/G$（例如 8 倍）。

### 优势
GQA 成功地在 **MQA 的速度** 和 **MHA 的质量** 之间找到了最佳平衡点。它能提供接近 MQA 的推理速度，同时保持接近 MHA 的模型性能。

## 4. 总结对比

| 特性 | Multi-Head (MHA) | Multi-Query (MQA) | Grouped-Query (GQA) |
| :--- | :--- | :--- | :--- |
| **KV Heads 数量** | $H$ (N_Heads) | $1$ | $G$ (N_Groups) |
| **KV Cache 大小** | 最大 ($2H$) | 最小 ($2$) | 中等 ($2G$) |
| **推理速度** | 慢 (Memory Bound) | 极快 | 快 |
| **模型质量** | 最好 | 有损失 | 接近 MHA |
| **典型应用** | GPT-3, Original BERT | Falcon-7B, StarCoder | LLaMA-2/3, Mistral, Qwen |

**结论**：在现代 LLM 中，GQA 已成为标准配置，因为它允许模型在有限显存下处理更长的 Context，是实现 Long Context 的关键技术之一。
