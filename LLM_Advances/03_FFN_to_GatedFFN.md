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

### 详细解析：SwiGLU 的每一项含义

$$
\text{SwiGLU}(x) = \underbrace{\text{Swish}_{\beta}(x W_{gate})}_{\text{Gate 门控信号}} \odot \underbrace{(x W_{in})}_{\text{Value 特征信号}} W_{out}
$$

1.  **$x W_{gate}$ (Gate Liner Layer)**
    *   **作用**：产生“门控”信号的原始 Logits。
    *   **直观理解**：网络通过学习 $W_{gate}$ 来决定哪些维度的特征是重要的（需要保留），哪些是不重要的（需要抑制）。

2.  **$\text{Swish}_{\beta}(\cdot)$ (Activation)**
    *   **定义**：$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$ (通常取 $\beta=1$，即 SiLU)。
    *   **作用**：将 Gate Logits 转换为具体的门控值。
    *   **特性**：
        *   与 Sigmoid 不同，SiLU 不是单纯的 0~1截断，而是允许负值区域有微小的梯度回流（非单调性）。
        *   当 $z$ 很大时，接近线性（保留）；当 $z$ 很小时，接近 0（抑制）。

3.  **$x W_{in}$ (Value Linear Layer)**
    *   **作用**：产生这一层的主要内容特征（Value），类似于 Attention 中的 $V$。
    *   **区别**：传统的 FFN 只有一个升维投影，而 SwiGLU 将“控制” ($W_{gate}$) 和“内容” ($W_{in}$) 解耦了。

4.  **$\odot$ (Element-wise Product)**
    *   **作用**：执行筛选操作。Gate 信号与 Value 信号逐元素相乘。
    *   **意义**：这是一种**动态特征选择**。如果 Gate 的某个维度接近 0，那么对应的 Value 特征就被屏蔽了；如果 Gate 很大，Value 特征就被保留并放大。

5.  **$W_{out}$ (Output Linear Layer)**
    *   **作用**：将筛选后的特征融合，并映射回模型的 hidden dimension。

---

### 原理解析：为什么 SwiGLU 如此有效？（The "Why"）

Shazeer 的论文并没有给出严苛的数学证明，但社区和后续研究总结了以下核心原因：

#### 1. 梯度的“高顺滑性” (Gradient Flow)
*   **ReLU 的问题**：ReLU 是分段线性的，在 $x<0$ 时梯度为 0（死区）。这会导致“神经元死亡”现象。
*   **Swish/SiLU 的优势**：SiLU 是光滑的非线性函数，且非单调（non-monotonic）。它在负半轴也有非零梯度。
*   **GLU 的乘法优势**：在反向传播时，$\odot$ 操作允许梯度通过 Gate 和 Value 两条路径流动。即使 Gate 关闭（接近0），Value 侧的梯度可能被阻断，但 Gate 参数本身仍然可以通过 $x W_{in}$ 的值获得梯度更新。这比单纯的 ReLU $max(0, x)$ 的梯度流更鲁棒。

#### 2. 动态调节能力 (Dynamic Gating)
*   Standard FFN 对所有输入 $x$ 应用相同的非线性变换模式。
*   SwiGLU 引入了 Context-aware 的处理：对于不同的输入 token，网络可以动态地调整 $W_{gate}$ 产生的开闭状态。这更像是一个**每层独立的小型 Attention 机制**（Self-Modulation）。

#### 3. 维度红利 (Dimensionality)
虽然我们将 Hidden Size 缩小到了 $\frac{2}{3}$ 以保持参数量不变，但我们获得了**两个**升维矩阵 ($W_{gate}, W_{in}$)。
*   这意味着网络在“宽度”上其实变宽了（虽然每一路窄了）。
*   两个矩阵可以分别学习不同的特征子空间（一个学控制，一个学内容），这种**解耦**通常比在一个大矩阵里混合学习更高效。

#### 4. 实验验证：Scaling Law
在 DeepMind 的 Chinchilla 和 Meta 的 LLaMA 研发过程中，大量消融实验证明：在同等计算量（FLOPs）下，SwiGLU 的 Perplexity 收敛值总是优于 GeLU FFN。这种 Scaling 优势在模型越大、训练数据越多时越明显。

---

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
        # 1. 计算 Gate 路径: Swish(x * W_gate)
        gate = F.silu(self.w_gate(x))
        # 2. 计算 Value 路径: x * W_in
        value = self.w_in(x)
        # 3. 逐元素相乘并输出: (Gate * Value) * W_out
        return self.w_out(gate * value)
```

SwiGLU 已经成为现代高性能大模型的标配组件，是构建 Strong Baseline 的不二之选。
