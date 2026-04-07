# 09. Dense -> Sparse MoE (Mixture of Experts)

随着模型规模从百亿 (10B) 向千亿 (100B) 甚至万亿 (1T) 迈进，训练和推理成本呈指数级增长。为了在增加模型容量的同时保持计算效率，**Mixture of Experts (MoE)** 架构成为了 Scaling 的关键技术（如 GPT-4, Mixtral 8x7B, DeepSeekMoE）。

## 1. 核心思想：稀疏激活 (Sparse Activation)
传统的 Dense 模型（稠密模型）对于每个输入 token，都会激活网络中的所有参数。这导致计算量（FLOPs）与参数量成正比。

**MoE 模型** 将 FFN 层替换为一组 **Experts（专家网络）**。
- 对于每个 token，只有一小部分 Expert 被激活（例如 8 个中的 2 个）。
- 这意味着：**总参数量极大（High Capacity），但每次推理的计算量极小（Active Parameters << Total Parameters）。**

## 2. 关键组件与路由机制 (Routing Mechanism)
典型的 MoE 层由 **N 个专家 (Experts)** 和一个 **路由网络 (Router / Gate)** 组成。

### 路由网络是怎么工作的？
Router 本质上是一个简单的线性层（Linear Layer），它将输入 token 的特征 $x$ 映射到 $N$ 个专家的打分上。
$$
\text{Scores} = x \cdot W_g
$$
其中 $W_g$ 是可训练的路由权重矩阵。

### 如何选择专家？(Top-K Gating)
这是一个关键点：**我们不会选择所有专家**。如果选择所有专家并加权（Soft-MoE），计算量就无法减少。
我们使用 **Top-K 策略**（通常 $K=1$ 或 $2$）：

1.  **计算分数**：Router 算出所有 $N$ 个专家的分数。
2.  **选出 Top-K**：只保留分数最高的 $K$ 个专家，将其余专家的权重强制置为 $-\infty$（即 Mask 掉）。
3.  **Softmax 归一化**：**仅对这 Top-K 个专家的分数**进行 Softmax，得到归一化权重 $g(x)$。
4.  **稀疏计算**：输入 $x$ **只会被发送给**这 $K$ 个被选中的专家进行前向传播。其余 $N-K$ 个专家完全不参与计算。
5.  **加权聚合**：
    $$ y = \sum_{i \in \text{TopK}} g(x)_i \cdot E_i(x) $$

**回答你的问题**：
*   **是通过 Softmax 选专家吗？** 是的，Router 输出 Score，但 Softmax 通常只作用于 Top-K 个被选中的专家。
*   **是所有专家都会选上吗？** **绝对不是**。如果全选，虽然权重不同，但每个专家都要算一遍，计算量不仅没少，反而多了 Router 的开销。MoE 的精髓就在于**只算 K 个**。

---

## 3. 挑战与解决方案 (详解)
MoE 的训练非常困难，业界花费了数年才解决以下核心难题：

### (a) 负载不均衡 (Load Imbalance)
*   **问题**：Router 可能会“偷懒”，发现把所有 token 都发给某一个“万能专家”损失下降得最快（Expert Collapse）。结果是：这一个专家累死（显存爆，计算排队），其他专家围观。这会导致模型退化为一个小得多的 Dense 模型。
*   **解决方案：辅助损失 (Load Balancing / Auxiliary Loss)**
    *   我们在总 Loss 中加入一项 $\mathcal{L}_{aux}$。
    *   该 Loss 鼓励 Router 输出的**概率分布**（Softmax 结果）和**实际派发分布**（每个专家接收到的 token 占比）尽可能接近均匀分布。
    *   $$ \mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i $$
    *   其中 $f_i$ 是第 $i$ 个专家被选中的频率，$P_i$ 是 Router 给第 $i$ 个专家的平均概率。当两者都均匀时，Loss 最小。

### (b) 专家容量限制 (Expert Capacity)
*   **问题**：即使有负载均衡 Loss，在局部 Batch 内，某些 Token 可能还是会扎堆去同一个专家。比如处理代码时，所有 token 都想去“代码专家”。如果显存放不下怎么办？
*   **解决方案：Capacity Factor & Token Dropping**
    *   我们可以设置一个强制上限（Capacity）：每个专家最多处理 $C$ 个 Token。
        $$ C \approx \frac{\text{Total Tokens}}{\text{Num Experts}} \times \text{Capacity Factor} (e.g., 1.1) $$
    *   **Token Dropping**：如果某个专家爆满了，多出来的 Token 会被**直接丢弃**（不经过该专家处理，直接通过 Residual Connection 传递，或者去次优专家）。虽然这听起来很粗暴，但 Google 研究（Switch Transformer）发现这比让模型变慢更好。

### (c) 训练稳定性 (Stability) & Router Z-Loss
*   **问题**：Router 的 Softmax 输出有时会非常极端（比如某个 logit 特别大），导致梯度在 Router 处爆炸或消失，训练发散。
*   **解决方案：Router Z-Loss**
    *   Google 在 PaLM/ST-MoE 中提出，强制惩罚过大的 Router Logits。
    *   $$ \mathcal{L}_{z} = \log^2(\sum e^{\text{logits}}) $$
    *   这鼓励 Router 输出比较小的数值，增加数值稳定性。

### (d) 显存开销 (Memory)
*   **问题**：模型参数量太大，单卡存不下。
*   **解决方案：专家并行 (Expert Parallelism, EP)**
    *   不同于数据并行（复制模型）或模型并行（切分层），EP 是将**不同的专家网络**放置在不同的 GPU 上。
    *   **All-to-All 通信**：在 Router 分发 Token 时，GPU 之间需要进行一次 All-to-All 通信，把 Token 发给持有对应专家的 GPU。计算完后，再通过 All-to-All 把结果传回来。这是 MoE 推理的主要延迟来源。

---

## 4. 典型模型
- **Switch Transformer (Google)**: 极端的 Top-1 Routing。
- **Mixtral 8x7B (Mistral AI)**: 高效的 Top-2 Routing，使用了 Megablocks 的 Dropless MoE 技术（优化了 Token Dropping 问题）。
- **DeepSeekMoE**: 提出了 **Fine-Grained Experts**（把 1 个大专家切成 4 个小专家）和 **Shared Experts**（专门划拨一部分专家作为“共享知识库”，所有 Token 必选），进一步提升了知识的专业化和冗余度。

## 5. 总结

| 特性 | Dense Model | Sparse MoE Model |
| :--- | :--- | :--- |
| **参数量** | $P$ | $N \times P$ (大得多) |
| **计算量 (FLOPs)** | $P$ | $\approx P$ (保持不变) |
| **激活方式** | 全激活 | **稀疏激活 (Sparse Is Better)** |
| **优势** | 训练稳定，部署简单 | 在同等算力下，极大提升上限 |
| **代表作** | GPT-3, LLaMA-2 | GPT-4, Mixtral 8x7B, DeepSeek |
