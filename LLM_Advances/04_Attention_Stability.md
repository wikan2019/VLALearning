# 04. Attention -> QK-Norm + Stability

在大规模模型训（LLM）练中，Attention 机制的数值不稳定性是一个主要瓶颈，会导致 Loss Spike（损失函数剧烈震荡）甚至训练崩溃。过去几年，研究者们提出了多种 **Attention 稳定性改进**，最著名的包括 **QK-Norm** (Query-Key Normalization) 和 **QK-LayerNorm**。

## 1. 核心问题：Attention Logits 爆炸
标准的 Scaled Dot-Product Attention 计算如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

随着模型深度增加，Query ($Q$) 和 Key ($K$) 的模长往往会不断增长。虽然有 $\frac{1}{\sqrt{d_k}}$ 进行缩放，但在极深的网络（如 100层+）中，点积 $QK^T$ 的值仍然可能非常大。

这会导致：
1. **Softmax 饱和**：当输入值很大时，Softmax 的梯度会变得极小（梯度消失），导致模型难以训练。
2. **Attention 分布尖锐化**：Attention 权重集中在极少数 token 上，导致“注意力坍缩”。

## 2. 解决方案：QK-Norm / QK-LayerNorm
为了解决这个问题，Google (ViT-22B, 2023) 和 Meta (LLaMA 2/3) 等团队引入了对 Q 和 K 进行归一化的操作。

### 方法一：QK-LayerNorm (ViT-22B)
在进行点积之前，先对 Q 和 K 应用 LayerNorm：

$$
Q' = \text{LayerNorm}(Q)
$$
$$
K' = \text{LayerNorm}(K)
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q' (K')^T}{\sqrt{d_k}}\right)V
$$

这一操作将 Q 和 K 投影到一个半径固定的超球面上，强制约束了 logits 的数值范围。

### 方法二：RMSNorm on QK
类似地，也可以使用 RMSNorm 来归一化 Q 和 K。这对数值稳定性的贡献巨大。

### 方法三：Cosine Similarity Attention
另一种激进的做法是直接使用余弦相似度（Cosine Similarity），这等价于对 Q, K 做 $L_2$ 归一化：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q}{\|Q\|_2} \frac{K^T}{\|K^T\|_2} \cdot \tau \right)V
$$
其中 $\tau$ 是可学习的温度系数。

## 3. 其他稳定性设计

### Gated Attention / NormFormer
- **NormFormer**：在 Attention 和 FFN 的残差连接处添加额外的 LayerNorm。
- **Gated Attention**：引入可学习的门控参数（初期初始化为 0），让模型在训练初期像线性模型一样，逐渐引入非线性交互。

### Head Scaling / Grouped-Query Attention (GQA)
虽然主要为了推理加速，但 **GQA** 通过减少 Key-Value head 的数量，间接影响了 Attention 的分布，也有助于训练的大规模并行效率。

## 4. 总结：从 Attention 到 Stable Attention

| 改进点 | 目的 | 实现方式 | 典型应用 |
| :--- | :--- | :--- | :--- |
| **QK-Norm** | 防止 Logits 爆炸 | LN(Q), LN(K) | ViT-22B, Scaling Vision Transformers |
| **QK-LayerNorm** | 稳定极大模型训练 | 在 Attention 内部加 LN | Google PaLM, CogView |
| **Clip Logits** | 暴力截断 | $\min(\max(x, -c), c)$ | 部分早期大模型 |

**结论**：在十亿（B）甚至百亿参数级别，普通的 Attention 可能还能工作。但到了千亿（100B+）参数级别，**QK-Norm 几乎是“必须”的技巧**，它能显著减少训练过程中的 Loss Spike，确保模型顺利收敛。
