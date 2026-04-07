# 07. RoPE -> ALiBi (Positional Extrapolation)

虽然 RoPE 解决了相对位置编码的问题，但在**外推性（Extrapolation）**方面（即：推理长度超过训练长度）仍有局限。**ALiBi (Attention with Linear Biases)** 是另一种无需显式位置嵌入（Positional Embedding）的方案，展现了极强的外推能力。

## 1. 核心思想
Press et al. (2021) 提出的 ALiBi 完全**移除了**输入层的 Positional Embedding（如 Sinusoidal 或 Learned Embedding）。

取而代之的是，ALiBi 直接在 **Attention Score** 上加一个与距离成比例的**静态偏置（Bias）**。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( q_i k_j^T + m \cdot -(i-j) \right) V
$$

其中：
- $i, j$ 是 token 的位置索引。
- $m$ 是一个针对每个 Head 特定的斜率（Slope）。

## 2. 机制详解
- **距离惩罚**：$-(i-j)$ 这一项确保了距离越远的 token，其 Attention Score 会被扣除越多。这引入了一种归纳偏置（Inductive Bias）：**最近的 token 最重要**。
- **Head Slope ($m$)**：不同的 Head 拥有不同的斜率 $m$（例如按几何级数 $2^{-1}, 2^{-2}, ...$ 分配）。有的 Head 关注局部（斜率大，远距离惩罚重），有的 Head 关注全局（斜率小，远距离惩罚轻）。

## 3. 优势：完美外推
ALiBi 的最大卖点是**外推能力**。
- 实验表明，在短序列（如 1024）上训练的 ALiBi 模型，可以直接在推理时处理长序列（如 16k 或更长），而不仅是不崩盘，Perplexity 还能保持稳定甚至下降（利用更多上下文）。
- 相比之下，RoPE 在不做额外处理（如 NTK-Scaling）的情况下，外推能力有限。

## 4. RoPE vs ALiBi
虽然 ALiBi 外推性极强，但目前最主流的大模型（LLaMA 系列）仍然主要使用 RoPE。原因可能包括：
1.  **表达能力**：ALiBi 强制施加了“距离衰减”的先验，这可能限制了模型关注“远距离但在语义上极为关键”的 token 的能力。RoPE 更加灵活。
2.  **长窗口训练可行性**：随着 FlashAttention 等技术的发展，直接在长窗口上训练 RoPE 变得可行，外推的需求可以通过“训练更长”来部分解决。

## 5. 总结

| 特性 | RoPE (Rotary) | ALiBi (Linear Bias) |
| :--- | :--- | :--- |
| **位置注入方式** | 旋转 Q, K 向量 | Attention Score 增加 Bias |
| **外推能力** | 较弱 (需插值/微调) | **极强 (无需微调)** |
| **长距离关注** | 灵活，可关注任意距离 | 受强制衰减影响，可能偏弱 |
| **主流应用** | LLaMA, PaLM, Qwen | MPT (MosaicML), BLOOM |
