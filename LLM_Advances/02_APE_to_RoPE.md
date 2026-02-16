# 02. APE -> RoPE

位置编码（Positional Encoding）是 Transformer 处理序列数据的关键。过去 5 年，位置编码从最初的 **对于绝对位置编码 (Absolute Positional Embedding, APE)** 几乎全面转向了 **旋转位置编码 (Rotary Positional Embedding, RoPE)**。

## 1. 背景：APE (Absolute Positional Embedding)
APE 是最早期的解决方案，用于告诉模型每个 token 在序列中的绝对位置（第 1 个，第 2 个...）。

### 常见形式
1. **Sinusoidal (正弦/余弦)**：Original Transformer 使用的固定公式。
   $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
2. **Learned Embedding**：GPT-2/BERT 使用的可学习参数矩阵，直接将位置索引映射为向量，$P \in \mathbb{R}^{L \times d}$。

### 局限性
- **外推性差 (Extrapolation)**：模型很难处理比训练长度更长的序列。如果你在 1024 长度上训练，模型很难理解位置 1025。
- **缺乏相对位置信息**：APE 显式地编码了 $x_i$ 的绝对位置，但某种程度上忽略了 token 之间的相对距离（$i-j$），而相对距离对理解语言结构往往更重要。

## 2. 演进：RoPE (Rotary Positional Embedding)
RoPE 由 Su et al. (2021) 提出，旨在结合绝对位置编码和相对位置编码的优点。它目前是 LLaMA、PaLM、GLM 等模型的标准配置。

### 核心思想
RoPE 不再将位置编码相加到词向量上（如 $x + p$），而是通过**旋转**的方式，将位置信息注入到 Query (Q) 和 Key (K) 向量中。

### 数学原理
RoPE 受复数运算启发。在二维平面上，将向量旋转角度 $\theta$，其相对角度差自然包含了相对位置信息。

对于位置 $m$ 的向量 $\mathbf{q}$，RoPE 对其成对的维度进行旋转：
$$
f(\mathbf{q}, m) = R_m \mathbf{q}
$$
其中 $R_m$ 是一个旋转矩阵。

关键性质是：
$$
\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = \mathbf{q}^T R_m^T R_n \mathbf{k} = \mathbf{q}^T R_{n-m} \mathbf{k}
$$
这意味着，**两个向量的点积（Attention Score）仅取决于它们的相对位置 $n-m$**，而不取决于绝对位置 $m$ 或 $n$。

### 优势
1. **完美支持相对位置**：Attention 机制自然地感知到 Token 之间的相对距离，这对长文本理解至关重要。
2. **更好的外推性 (Extrapolation)**：相比 APE，RoPE 在处理超过训练长度的序列时表现更好（也就是所谓的“长度外推”能力）。
3. **无需增加参数**：RoPE 是确定性的变换，不需要像 Learned Embedding 那样增加可学习参数。
4. **兼容线性 Attention**：由于使用了旋转操作，它保留了向量模长不变的性质，有助于训练稳定性。

## 3. 相关变体与扩展 (Long Context)
随着对长上下文（Long Context）需求的增加，RoPE 也衍生出了多种变体以进一步增强外推能力：

- **Linear Interpolation (PI)**：在推理时，将位置索引按比例缩小，使其落入训练时的范围内。
- **NTK-Aware Scaled RoPE**：通过调整频率基数，在不微调的情况下支持更长序列。
- **Yarn (Yet another RoPE extension)**：进一步改进插值方法，解决长文本下的 "lost in the middle" 问题。

## 4. 总结对比

| 特性 | APE (Sinusoidal/Learned) | RoPE (Rotary) |
| :--- | :--- | :--- |
| **注入方式** | 加法 ($x + p$) | 乘法/旋转 ($x \times R$) |
| **作用对象** | 输入 Embedding | Attention 层的 Q 和 K |
| **位置感知** | 强绝对位置，弱相对位置 | 强相对位置 (Explicit Relative) |
| **外推能力** | 差 | 较好 (尤其配合插值技巧) |
| **主流模型** | BERT, GPT-2/3 | LLaMA, PaLM, Qwen, Mistral |
