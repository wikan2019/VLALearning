# 01. LayerNorm -> RMSNorm

在过去 5 年的大语言模型（LLM）发展中，归一化层（Normalization Layer）的设计发生了重要演变。最显著的变化是从经典的 **Layer Normalization (LayerNorm)** 转向了 **Root Mean Square Normalization (RMSNorm)**。这一转变主要为了提升训练稳定性并减少计算开销。

## 1. 背景：LayerNorm (LN)
LayerNorm 是 Transformer 架构（Attention Is All You Need, 2017）中默认使用的归一化方法。

### 核心机制
LayerNorm 对每个样本的特征向量 $x$ 进行归一化，使其均值为 0，方差为 1。计算公式如下：

$$
\bar{x} = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \bar{x})^2
$$

$$
\hat{x} = \frac{x - \bar{x}}{\sqrt{\sigma^2 + \epsilon}}
$$

最后应用可学习的仿射变换参数 $\gamma$ (缩放) 和 $\beta$ (平移)：

$$
\text{LN}(x) = \hat{x} \odot \gamma + \beta
$$

### 局限性
- **计算开销**：需要计算均值 $\bar{x}$ 和方差 $\sigma^2$。均值的计算在某些硬件上会带来额外的开销。
- **重新中心化（Re-centering）的必要性存疑**：研究发现，强制均值为 0（re-centering）对于模型收敛的贡献可能并不如缩放（re-scaling）重要。

## 2. 演进：RMSNorm
RMSNorm (Root Mean Square Layer Normalization) 由 Zhang et al. (2019) 提出，并在 LLaMA、Gemma、PaLM 等现代主流 LLM 中被广泛采用。

### 核心思想
RMSNorm 省略了计算均值（re-centering）的步骤，只通过均方根（RMS）进行缩放。它假设特征的均值不仅不需要强制为 0，而且其偏移对层归一化的益处贡献微乎其微。

### 计算公式
计算均方根：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

归一化：

$$
\bar{x} = \frac{x}{\text{RMS}(x) + \epsilon}
$$

输出（同样包含可学习的缩放参数 $g_i$，通常省略了平移参数 $\beta$）：

$$
\text{RMSNorm}(x) = \bar{x} \odot g
$$

### 优势
1. **计算效率更高**：减少了均值计算和减法操作，计算图更简单，在 GPU 等硬件上速度提升显著（约 10%-40%）。
2. **效果持平或略优**：在大多数 LLM 任务中，RMSNorm 能够达到与 LayerNorm 相同甚至更好的收敛速度和最终性能。
3. **简化实现**：去除了 $\beta$（偏置项），进一步减少了参数量。

## 3. 对比总结

| 特性 | LayerNorm (LN) | RMSNorm |
| :--- | :--- | :--- |
| **计算统计量** | 均值 Mean & 方差 Variance | 均方根 RMS |
| **操作** | 平移 (Re-centering) + 缩放 (Re-scaling) | 仅缩放 (Re-scaling) |
| **可学习参数** | $\gamma$ (Scale), $\beta$ (Bias) | $g$ (Scale) |
| **计算复杂度** | 较高 | 较低 (节省 ~10-40% 时间) |
| **主流应用** | BERT, GPT-2, Original Transformer | LLaMA, PaLM, Gopher, Chinchilla |

## 4. 为什么 RMSNorm 的数值稳定性更优？（缓解梯度消失/爆炸）

这是理解 RMSNorm 价值的关键问题。要回答它，需要从**梯度传播**的角度来分析。

### 4.1 核心问题：深层网络的梯度困境

在一个 $L$ 层的 Transformer 中，梯度通过链式法则反向传播：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \prod_{l=1}^{L} \frac{\partial x_l}{\partial x_{l-1}} \cdot \frac{\partial \mathcal{L}}{\partial x_L}
$$

如果每一层的雅可比矩阵 $\frac{\partial x_l}{\partial x_{l-1}}$ 的谱范数（最大奇异值）持续大于 1，梯度就会**指数级爆炸**；持续小于 1，则会**指数级消失**。

### 4.2 归一化如何帮助？

归一化层的核心作用是**约束每一层输出的数值范围**。无论是 LayerNorm 还是 RMSNorm，它们都将输出向量的"大小"（范数/方差）限制在一个相对稳定的区间。这直接意味着：

1. **雅可比矩阵的谱范数被约束**：归一化后的输出不会无限放大或缩小，因此每层的乘法因子接近 1，避免了连乘导致的指数级失控。
2. **损失曲面更平滑 (Smoother Loss Landscape)**：研究表明 (Santurkar et al., 2018)，归一化使得损失函数的 Lipschitz 常数更小，梯度变化更平缓，优化更稳定。

### 4.3 为什么 RMSNorm 比 LayerNorm "更"稳定？

虽然二者都能约束数值范围，但 RMSNorm 在实践中表现出更优的稳定性，原因有三：

#### (a) 更简单的计算图 → 更干净的梯度

LayerNorm 的梯度计算中包含均值 $\bar{x}$ 的求导项。由于 $\bar{x}$ 依赖于所有维度的输入，其反向传播会引入额外的**耦合项**：

$$
\frac{\partial \text{LN}(x_i)}{\partial x_j} = \frac{1}{\sigma} \left( \delta_{ij} - \frac{1}{d} - \frac{(x_i - \bar{x})(x_j - \bar{x})}{d\sigma^2} \right)
$$

RMSNorm 没有均值减法，其雅可比矩阵更简洁：

$$
\frac{\partial \text{RMS}(x_i)}{\partial x_j} = \frac{g_i}{\text{RMS}(x)} \left( \delta_{ij} - \frac{x_i x_j}{d \cdot \text{RMS}(x)^2} \right)
$$

少了 $\frac{1}{d}$ 这样的耦合项，梯度的来源更"干净"，在极深网络（100+ 层）中，这种微小的差异会被放大。

#### (b) 尺度不变性 (Scale Invariance)

RMSNorm 具有完美的缩放不变性，即对于任意正标量 $\alpha$：

$$
\text{RMSNorm}(\alpha x) = \text{RMSNorm}(x)
$$

这意味着，即使某一层的输出突然变大（例如由于学习率波动），RMSNorm 也能将其"拉回"到正常范围。LayerNorm 也有类似性质，但由于均值减法的存在，其对**偏移+缩放**的联合不变性稍弱。

#### (c) 避免"零均值假设"带来的信息损失

LayerNorm 强制将特征均值归零，这在某些情况下可能会抹去有用的**偏置信息**（bias information）。RMSNorm 保留了特征的均值，只约束了模长，这使得模型能保留更多信息，减少了因信息丢失导致的梯度噪声。

### 4.4 小结

| 稳定性因素 | LayerNorm | RMSNorm |
| :--- | :--- | :--- |
| 梯度耦合项 | 包含均值相关的耦合项 | 更简洁，耦合更少 |
| 尺度不变性 | 有（但受均值影响） | 更纯粹 |
| 信息保留 | 强制零均值，可能丢失偏置信息 | 保留均值，信息损失更小 |
| 实际效果 | 深层网络中偶有 Loss Spike | 更少 Loss Spike，收敛更稳 |

---

## 5. RMSNorm 与 PreNorm / PostNorm 的关系

**PreNorm 和 PostNorm 描述的是归一化层在残差块中的位置**，而 RMSNorm / LayerNorm 描述的是**归一化层本身的计算方式**。二者是**正交的设计选择**，可以自由组合。

### 5.1 PostNorm（原始 Transformer 方案）

归一化放在残差连接**之后**：

$$
x_{l+1} = \text{Norm}(\text{SubLayer}(x_l) + x_l)
$$

```
Input ─┬─── SubLayer ───┐
       │                 ↓
       └───── Add ───── Norm ── Output
```

**特点：**
- 这是 Vaswani et al. (2017) 原始 Transformer 的方案。
- 理论上最终表示质量更好（因为最后一层有 Norm 规整）。
- **缺点：训练不稳定**，尤其在深层网络中。因为梯度需要"穿过" Norm 层才能到达残差路径，Norm 层的非线性操作会干扰梯度流。
- 通常需要 **Learning Rate Warmup** 来避免训练初期崩溃。

### 5.2 PreNorm（现代 LLM 的主流方案）

归一化放在残差连接**之前**（即子层的输入端）：

$$
x_{l+1} = \text{SubLayer}(\text{Norm}(x_l)) + x_l
$$

```
Input ─┬── Norm ── SubLayer ──┐
       │                      ↓
       └────────── Add ────── Output
```

**特点：**
- 梯度可以通过残差连接（$+x_l$）**直接回传**到更早的层，不被 Norm 干扰。这条"梯度高速公路"使得深层训练更加稳定。
- **不需要 Learning Rate Warmup**（或需求大大降低）。
- 缺点：最后一层的输出没有经过 Norm 规整，因此有些模型会在最后额外加一个 Final Norm。

### 5.3 现代 LLM 的黄金组合：PreNorm + RMSNorm

几乎所有现代主流 LLM（LLaMA, PaLM, Mistral, Qwen 等）都采用了 **Pre-RMSNorm** 的架构：

$$
x_{l+1} = \text{SubLayer}(\text{RMSNorm}(x_l)) + x_l
$$

这个组合的优势叠加：

| 设计选择 | 贡献 |
| :--- | :--- |
| **PreNorm** | 保证梯度的"高速公路"，深层训练不崩溃 |
| **RMSNorm** | 更高效的归一化计算，更简洁的梯度 |
| **组合效果** | 稳定 + 高效 = 适合千亿参数级训练 |

### 5.4 直观理解

```
PostNorm (老方案，不稳定):  梯度必须穿过 Norm → 信号被扭曲
  x → [Attn] → [+] → [LN] → [FFN] → [+] → [LN] → ...
                       ^^^^                   ^^^^
                    梯度需要穿过这里，被干扰

PreNorm (新方案，稳定):  梯度可以通过残差直接回传
  x → [RMSNorm] → [Attn] → [+] → [RMSNorm] → [FFN] → [+] → ...
       ↑                     |                          |
       └─────────────────────┘   直达梯度高速公路  ←────┘
```

### 5.5 总结

| 维度 | PostNorm | PreNorm |
| :--- | :--- | :--- |
| **Norm 位置** | 残差连接之后 | 残差连接之前（子层输入） |
| **梯度流** | 受 Norm 干扰，深层不稳定 | 残差直通，深层稳定 |
| **Warmup 需求** | 必须 | 可选（通常不需要） |
| **搭配的 Norm** | 通常是 LayerNorm | 现代 LLM 用 RMSNorm |
| **代表模型** | Original Transformer, BERT | LLaMA, PaLM, GPT-3+, Mistral |
