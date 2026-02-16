# 03. FFN -> Gated FFN (SwiGLU)

前馈神经网络（Feed-Forward Network, FFN）是 Transformer 中参数量最大的部分。近年来，FFN 的结构从标准的的两层全连接网络演变成了使用门控机制的 **Gated FFN**，特别是 **SwiGLU** 变体。

## 1. 背景：Standard FFN
在原始 Transformer (2017) 和 GPT-3 中，FFN 的结构非常简单：一个线性层（升维），接一个激活函数（如 ReLU 或 GeLU），再接一个线性层（降维）。

$$
\text{FFN}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
$$

通常中间层维度 $d_{ff} = 4d_{model}$。

## 2. 演进：Gated FFN & SwiGLU
Shazeer (2020) 在论文 "GLU Variants Improve Transformer" 中通过大量实验证明，引入门控线性单元（Gated Linear Unit, GLU）可以显著提升模型性能。目前最流行的变体是 **SwiGLU**。

### 核心机制
Gated FFN 将输入信号分为两路，一路作为“门”（Gate），控制另一路信号的通过量。

**SwiGLU 的计算公式：**

$$
\text{SwiGLU}(x) = \text{Swish}_{\beta}(x W_{gate}) \odot (x W_{in}) W_{out}
$$

其中：
- $x W_{gate}$：计算门控值。
- $\text{Swish}_{\beta}(z) = z \cdot \sigma(\beta z)$：Swish 激活函数（当 $\beta=1$ 时即 SiLU）。
- $x W_{in}$：计算主要特征值。
- $\odot$：逐元素乘法（Hadamard Product）。
- $W_{out}$：输出投影层。

注意：传统的 FFN 只有两个权重矩阵，而 SwiGLU 有三个 ($W_{gate}, W_{in}, W_{out}$)。为了保持参数量一致，通常会将 $d_{ff}$ 从 $4d$ 减小到 $\frac{2}{3} 4d \approx 2.68d$ (如 LLaMA 中的做法)。

### 为什么有效？
1. **更强的表达能力**：乘法交互（Multiplicative Interaction）使得模型能够学习特征之间更复杂的非线性关系。
2. **类似于“注意力”的筛选机制**：门控机制允许模型动态地选择哪些特征需要保留，哪些需要抑制。
3. **梯度的流动**：GLU 结构有助于梯度更顺畅地传播，改善训练稳定性。

## 3. 标准 FFN vs SwiGLU

| 特性 | Standard FFN (ReLU/GeLU) | Gated FFN (SwiGLU) |
| :--- | :--- | :--- |
| **结构** | 线性 -> 激活 -> 线性 | (线性 -> 激活) * (线性) -> 线性 |
| **矩阵数量** | 2个 ($W_{up}, W_{down}$) | 3个 ($W_{gate}, W_{up}, W_{down}$) |
| **中间层维度** | 通常 $4d$ | 通常 $\frac{8}{3}d$ (保持参数量一致) |
| **性能** | 基准 | 显著优于基准 (降低 Perplexity) |
| **应用模型** | GPT-3, BERT | LLaMA, PaLM, OLMo, DeepSeek |

## 4. 代码实现对比 (PyTorch)

**Standard FFN (GeLU):**
```python
class FFN(nn.Module):
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
```

**SwiGLU:**
```python
class SwiGLU(nn.Module):
    def forward(self, x):
        # x_gate (gate path) * x_in (value path)
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))
```

SwiGLU 已经成为现代高性能大模型的标配组件，是构建 Strong Baseline 的不二之选。
