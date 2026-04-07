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

## 2. Memory Tokens (LLMs) 的详细实现
在 LLM 领域，**Memory Tokens** 或 **Summary Tokens** 的引入旨在解决 Transformer 的上下文窗口有限问题，使其具备处理无限流式数据（Streaming Data）或超长文档的能力。

### 2.1 Recurrent Memory Transformer (RMT)
RMT 将 Transformer 改造为类似 RNN 的循环结构，但保留了并行的自注意力机制。

#### 具体实现方式
1.  **输入格式封装 (Input Wrapping)**：
    对于每个长文档的分段（Segment），我们在输入的**开头**和**结尾**分别添加 $k$ 个特殊的 Memory Tokens `[mem]`。
    
    $$ X_{segment} = [\text{[mem]}_{read}, x_1, x_2, ..., x_n, \text{[mem]}_{write}] $$

2.  **读写机制 (Read-Write Mechanism)**：
    *   **Read操作**：`[mem]read` 初始化为上一段输出的 `[mem]write` 向量（梯度不截断或截断取决于训练策略）。
    *   **Processing**：Transformer 处理整个序列。`[mem]read` 中的历史信息通过 Self-Attention 广播给当前段的文本 token。
    *   **Write操作**：在处理结束时，模型将当前段的关键信息压缩写入到 `[mem]write` 中，供下一段使用。

3.  **训练策略 (BPTT)**：
    为了让模型学会利用 Memory 跨段传递信息，通常使用 **Backpropagation Through Time (BPTT)**。梯度会穿过 Memory token 回传到前几个段，使模型学会“该记住什么，该遗忘什么”。

#### 应用实例
*   **处理 100万+ Token**：RMT 在 BERT/T5 基础上微调后，成功处理了超过 100 万 token 的序列，检索精度保持较高。
*   **长篇小说理解**：读取整本小说前文，并在后文中回答关于前文细节的问题。

---

### 2.2 StreamLLM (Attention Sinks)

StreamLLM 是一种 **无需重新训练** 即可让 LLM 处理无限长上下文（Infinite Context）的推理优化技术。它主要解决了“滑动窗口”策略导致模型崩盘的问题。

#### 1. 核心崩溃原因：为什么简单“滑动窗口”不行？
在流式对话中，当显存（KV Cache）满了，最直观的做法是“滑动窗口”：丢弃最早的 token，腾出空间给新 token。
*   **现象**：一旦丢弃了序列开头的第一个 token（通常是 `<s>`），模型的 Perplexity（困惑度）会瞬间暴涨至几千甚至几万，输出完全变成乱码。
*   **原因分析 (Attention Sink 假说)**：
    *   **Softmax 分布特性**：在计算 Attention Score 时，Softmax 函数强制所有分数的和为 1：$\sum \text{softmax}(x) = 1$。
    *   **由于 Sink Token 的存在**：模型在训练时，倾向于把大量“多余”的注意力权重分配给第一个 token（或前几个）。即使这些 token 没有实际语义，它们也充当了**“垃圾桶”（Attention Sink）**，吸收了那些“不需要关注任何具体的词”的注意力分数。
    *   **丢弃后的灾难**：如果我们删除了这个“垃圾桶”（Sink Token），模型原本分配给它的巨大分数（例如 0.5）就无处可去。由于 Softmax 的归一化性质，这 0.5 的权重会被迫重新分配给剩余的 token（即最近的 token）。
    *   **结果**：剩余 token 的注意力权重被错误地大幅抬高，导致特征表示发生剧烈偏移（Variance Shift），模型产生严重的幻觉或乱码。

#### 2. 具体实现方式：Sink Cache + Rolling Cache
StreamLLM 的解决方案非常简单：**死保开头，循环滚动中间。** 其 KV Cache 由两部分组成：

1.  **Sink Cache (稳定锚点)**：
    *   始终保留序列**最开头**的 4 个 token 的 KV 值（例如：`<s>`, `User`, `:`, `Hi`）。
    *   这些 token 的位置编码固定为 $[0, 1, 2, 3]$。
    *   它们就像是“定海神针”，稳住了 Attention 的 Softmax 分布。

2.  **Rolling Cache (滑动窗口)**：
    *   保留从当前时刻 $t$ 往前倒推的 $L$ 个 token（例如最近的 1024 个）。
    *   当新 token 进来时，丢弃 Rolling Cache 中最旧的一个（FIFO）。

#### 3. 关键细节：位置编码 (Positional Encoding) 的特殊处理
在 Rolling Cache 中，当新 token 进来并挤掉旧 token 时，我们如何处理位置编码？这里有一个容易误解的“相对位置不变性”问题。

*   **问题**：如果我们每次都把 cache 里的 token 位置重置为 $[0, 1, ..., L]$，那么我们需要重新计算所有 Key 的 RoPE 旋转（因为 $Keys$ 依赖于绝对位置 $m$），计算量巨大。
*   **StreamLLM 的策略**：**不重置，直接顺延**。
    *   **Sink Tokens**：始终使用绝对位置 $[0, 1, 2, 3]$。
    *   **Rolling Tokens**：使用其**原始的绝对位置**。例如，对话进行到第 10,000 步时，Rolling Cache 里的 token 位置就是 $[8977, ..., 10000]$（假设窗口 1024）。
    *   **注意力计算**：
        *   当 Query（位置 10001）去查询 Sink（位置 0）时，相对距离是 10001。
        *   当 Query（位置 10001）去查询 Rolling（位置 10000）时，相对距离是 1。
*   **为什么这能工作？（相对位置不变性）**
    *   RoPE 的核心性质是 Attention Score 仅取决于相对距离 $(m-n)$。
    *   对于 **Rolling Cache**：虽然绝对位置很大，但它们与 Query 的**相对距离**始终保持在 0~1024 范围内（即训练时的上下文长度内）。因此，模型能完美理解局部上下文。
    *   对于 **Sink Cache**：虽然相对距离巨大（10000+），超出了训练长度，但由于 Sink Token 的作用主要是“吸收分数”而非“提供语义”，实验发现 LLM 对 Sink Token 的位置不敏感（只要它在开头就行）。
    *   **结论**：我们不需要特殊切断或重置位置编码，只需让位置索引随时间自然增长即可。这使得 KV Cache 无需刷新，推理效率极高。

#### 应用实例
*   **无限长对话机器人**：让 LLaMA-2-70B 连续运行几个月，处理数十亿 token 的对话，显存占用恒定。
*   **实时语音转写/翻译流**：持续处理无休止的语音输入流。

---

## 3. 总结对比

| 核心特性 | Register Tokens (ViT) | Memory Tokens (RMT) | Attention Sinks (StreamLLM) |
| :--- | :--- | :--- | :--- |
| **存在形式** | Input 中的空 token | Segment 首尾的特殊 token | 复用开头的普通 token (系统性 Artifact) |
| **主要功能** | **暂存区 (Scratchpad)** | **信息传递 (Pass Context)** | **数值稳定 (Numerical Stability)** |
| **解决痛点** | 消除 Attention 图中的噪声 | 突破上下文长度限制 (Infinite Context) | 突破显存限制，实现无限推流 |
| **实现代价** | 需从头训练或微调 | 需专门构造数据微调 (BPTT) | **无需训练**，推理端修改代码即可 |

**核心观点**：
*   **Register Tokens** 告诉我们模型需要“草稿纸”来存全局信息。
*   **RMT** 告诉我们模型可以通过“接力棒”的方式传递无限长的信息。
*   **Attention Sinks** 告诉我们模型内部有一些奇怪但极其重要的“废弃回收站”机制，尊重这一机制就能实现高效的流式推理。
