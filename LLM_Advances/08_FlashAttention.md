# 08. Standard Attention -> FlashAttention (IO-Aware Exact Attention)

随着 Context Length 的增长（从 2k 到 32k, 100k, 1M），Attention 的 $O(N^2)$ 显存和计算复杂度成为最大瓶颈。**FlashAttention** (Dao et al., 2022/2023) 的出现彻底改变了这一局面。

**关键认知**：FlashAttention **不是**近似算法（Approximate Attention，如 Sparse/Low-Rank），它是**精确注意力（Exact Attention）**。它的输出与标准 Attention 数学上完全一致，但速度快 2-4 倍，显存节省 10-20 倍。

## 1. 核心瓶颈：Memory Wall (显存墙)
在现代 GPU（如 A100）上，计算能力（FLOPs）远超显存带宽（Bandwidth）。
*   **HBM (High Bandwidth Memory)**：容量大（80GB），但读写慢（2TB/s）。
*   **SRAM (Shared Memory)**：容量极小（192KB/SM），但读写极快（19TB/s）。

**标准 Attention 实现的痛点**：
计算 $S = QK^T$, $P = \text{softmax}(S)$, $O = PV$ 时，PyTorch 需要把巨大的 $N \times N$ 矩阵 $S$ 和 $P$ **完整写入 HBM**，然后再读出来计算下一步。
*   对于长文本（$N=128k$），$N \times N$ 矩阵可能高达 100GB+，光是读写这些中间矩阵就耗尽了所有时间，计算单元（Tensor Cores）大部分时间都在空转等待数据。

---

## 2. 为什么 FlashAttention 不是近似计算？
许多加速算法（如 Reformer, Linformer）通过稀疏化或低秩近似来减少计算，这会牺牲精度。
FlashAttention 之所以是**精确的**，是因为它利用了 **Online Softmax** 技巧。

### Online Softmax 公式推导
标准的 Softmax 需要先求出整行的 max 值 $m$ 和分母 $\ell = \sum e^{x_i - m}$。这意味着必须看完全部 $N$ 个元素才能算出一个 Softmax 值。

FlashAttention 利用了 Softmax 的分块计算性质。假设我们将向量 $x$ 分为两块 $x^{(1)}$ 和 $x^{(2)}$：
1.  计算第一块的局部最大值 $m_1$ 和局部指数和 $\ell_1 = \sum e^{x^{(1)} - m_1}$。
2.  计算第二块的局部最大值 $m_2$ 和局部指数和 $\ell_2 = \sum e^{x^{(2)} - m_2}$。
3.  **合并更新 (Merge Step)**：
    *   新的全局最大值：$m_{new} = \max(m_1, m_2)$
    *   新的全局指数和：
        $$ \ell_{new} = e^{m_1 - m_{new}}\ell_1 + e^{m_2 - m_{new}}\ell_2 $$
    *   这就是你提到的公式 $\alpha \cdot \text{softmax}(x_1) + \dots$ 里的系数来源。具体来说，对于第一块的 Softmax 结果，现在的缩放系数 $\alpha$ 为：
        $$ \alpha = \frac{\ell_1 e^{m_1 - m_{new}}}{\ell_{new}} $$

通过在 SRAM 中不断更新 $m$ 和 $\ell$，FlashAttention 可以在处理完所有块后，得到与标准 Softmax **完全一致** 的数值结果。

---

## 3. FlashAttention 如何加速？(IO-Awareness)
核心思想：**Tiling (分块) + Kernel Fusion (算子融合)**

### (a) Tiling (分块计算)
它将 $Q, K, V$ 切分成小块（Block），使其可以完全装入高速的 **SRAM**。
FlashAttention 编写了一个巨大的 CUDA Kernel，在 SRAM 内部一次性完成 $S = QK^T$, $P=\text{softmax}(S)$, $O=PV$ 的所有计算。
**关键点**：巨大的中间矩阵 $S$（Attention Score）和 $P$（Attention Probability）**从未被写入 HBM**。它们在 SRAM 中生成后被立刻使用，用完即弃。这减少了 **90% 以上** 的 HBM 读写量。

### (b) Recomputation (反向传播梯度计算)
在前向传播时，FlashAttention **不存储** $N \times N$ 的 Attention Map $P$。
在反向传播时，我们只有输出梯度 $dO$ 和前向保留的统计量 $L$ (即 log-sum-exp)。

**梯度计算流程**：
1.  从 HBM 读取 $Q_i, K_j, V_j$ 到 SRAM。
2.  **重计算 $S_{ij}$**：在 SRAM 中再次计算 $S_{ij} = Q_i K_j^T$。
3.  **重计算 $P_{ij}$**：利用保存的 $L_i$，恢复出概率值 $P_{ij} = \exp(S_{ij} - L_i)$。
4.  **计算梯度** (标准 Backprop 公式)：
    *   $dV_j \leftarrow P_{ij}^T \cdot dO_i$
    *   $dP_{ij} \leftarrow dO_i \cdot V_j^T$
    *   $dS_{ij} \leftarrow P_{ij} \cdot (dP_{ij} - \sum_k P_{ik} dP_{ik})$
    *   $dQ_i \leftarrow dS_{ij} \cdot K_j$
    *   $dK_j \leftarrow dS_{ij}^T \cdot Q_i$

虽然多进行了一次 Forward 计算（步骤 2 和 3），但由于 $S_{ij}$ 的计算是 Compute-Bound 的，而从 HBM 读取 $N \times N$ 矩阵是 Memory-Bound 的，**重计算往往比读取更快**。

---

## 4. 局限性与反例：FlashAttention 的短板
虽然 FlashAttention 几乎已成为标准，但它并非万能：

1.  **极短序列 (Small Context Length)**：
    *   当 $N$ 很小（例如 $< 512$）时，主要的瓶颈不是显存带宽，而是 Kernel Launch 的开销和 GPU SM 的利用率。此时，FlashAttention 复杂的 Tiling 逻辑可能反而比简单的标准 Attention 慢。
    *   **例子**：在某些 ResNet 风格的 CV 小模型或短文本分类任务中，标准 Attention 可能更快。
2.  **不规则 Mask (Irregular/Sparse Masks)**：
    *   FlashAttention 对 Causal Mask (下三角) 做了专门优化。但如果你需要一个非常稀疏、且分布完全随机的 Attention Mask（例如某些 GNN 或特殊的稀疏 Transformer），FlashAttention 的 Block 遍历机制可能效率很低，因为它必须扫描那些本该被 Mask 掉的 Block。
    *   **例子**：BigBird 或 Longformer 的特定稀疏模式，如果用 FlashAttention 强行跑，可能不如专门写的稀疏 Kernel。
3.  **缺乏 Tensor Core 支持的硬件**：
    *   FlashAttention 高度依赖 FP16/BF16 的 Tensor Core 进行矩阵乘法加速。在不支持 Tensor Core 的老旧 GPU (如 P40, V100 以前) 或 CPU 上，收益有限甚至无法运行。

---

## 5. 为什么它使长 Context 成为可能？(显存开销)
标准 Attention 的显存占用是 **$O(N^2)$**，因为需要存储 $N \times N$ 的 Attention Map 用于反向传播。
*   $N=1024$ 时，$N^2$ 很小。
*   $N=100k$ 时，$N^2 = 100亿$，即使 FP16 也需要 20GB 显存存这一张表，根本存不下。

FlashAttention 的显存占用是 **$O(N)$**（线性）。
*   因为它不存 $N \times N$ 的表，只存 $O(N)$ 级别的统计量 ($L$ 和 $m$)。
*   这使得显存瓶颈被彻底打破。只要能存下 $Q, K, V$ 本身，就能跑 Attention。
*   结果：Context Length 从 2k/4k 飞跃到了 32k, 128k, 甚至 1M (RingAttention)。

---

## 6. 总结

| 特性 | Standard Attention | FlashAttention |
| :--- | :--- | :--- |
| **计算精度** | 精确 | **精确 (Exact)** |
| **算法复杂度** | $O(N^2)$ FLOPs | $O(N^2)$ FLOPs (但在更快的 SRAM 上跑) |
| **显存复杂度** | **$O(N^2)$** (存 Attn Map) | **$O(N)$** (线性显存) |
| **IO 访问** | 读写巨大的 $N \times N$ 矩阵 | 只读写 $Q, K, V, O$ |
| **加速原理** | 无 | 减少 HBM 访问 (Memory Bound -> Compute Bound) |
| **局限性** | 显存爆炸，慢 | 短序列开销大，不规则 Mask 难优化 |
| **影响** | 只有短文本能跑 | **Long Context 时代的基石** |
