# 10. RLHF (PPO) -> DPO (Direct Preference Optimization)

在 LLM 的对齐（Alignment）阶段，**RLHF (Reinforcement Learning from Human Feedback)** 曾是业界标准（如 InstructGPT/ChatGPT）。然而，RLHF 流程复杂、训练不稳。**DPO (Direct Preference Optimization)** 的出现（Rafailov et al., 2023）将这一过程简化为直接的监督学习，彻底改变了 LLM 的对齐范式。

## 1. 背景：RLHF 的复杂性 (Pipelines)
标准的 PPO-based RLHF 需要维护 **4 个模型** 并不是一件容易的事：
1.  **Reference Model** (冻结的 SFT 模型，防止遗忘)
2.  **Reward Model** (训练好的奖励模型)
3.  **Policy Model** (当前正被训练的 LLM)
4.  **Critic/Value Model** (用于 PPO 价值估计)

**痛点**：
- **资源消耗极大**：同时加载 4 个模型，显存爆炸。
- **超参敏感**：PPO 对超参数极其敏感，容易出现 Reward Hacking 或训练发散。

## 2. DPO 的核心思想与数学原理
DPO 的最大贡献在于发现了一个数学转换：**最优策略与奖励函数是一一对应的**。

### (a) 为什么不需要 Reward Model？
在 PPO 中，我们需要显式训练一个奖励模型 $r(x, y)$ 来告诉 Policy 哪句话写得好。
DPO 的推导表明，最优策略 $\pi^*(y|x)$ 可以直接表示为：
$$ r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + Z(x) $$
这意味着，**有了策略就等于有了奖励函数**。我们可以将这个 $r(x,y)$ 代入 Bradley-Terry 偏好模型，直接推导出仅包含策略 $\pi$ 的损失函数，从而消去了显式的 $r(x,y)$。

### (b) DPO vs SFT：本质区别
*   **SFT (Supervised Fine-Tuning)**：
    *   目标：**模仿**。最大化参考答案（Human Demo）的似然概率。
    *   缺点：只知道“什么是对的”，不知道“什么是错的”。容易导致“复读机”或产生幻觉。
*   **DPO (Direct Preference Optimization)**：
    *   目标：**对比**。同时**提高**好答案（Winner）的概率，**降低**坏答案（Loser）的概率。
    *   公式直观含义：最大化 Winner 和 Loser 之间的**相对 margin**。
    *   **效果**：模型不仅学会了怎么说对，还学会了避开常见的错误陷阱（如啰嗦、拒绝回答、有害内容）。

---

## 3. 为什么 DPO 比 PPO 更稳定？
PPO 是一种**在线强化学习 (Online RL)** 算法，而 DPO 本质上是**离线监督学习 (Offline Supervised Learning)**。

1.  **没有 Sampling (采样)**：PPO 训练时，模型需要实时生成回答（Sampling），这非常慢且具有随机性。DPO 直接在静态数据集上计算概率，速度快且确定性强。
2.  **没有 Value Estimation (价值估计)**：PPO 需要训练一个 Critic (Value Model) 来估计未来收益，这是一个高方差（High Variance）的估计过程，极不稳定。DPO 没有这个步骤。
3.  **超参也更少**：PPO 需要调优 Clip Range, Entropy Coeff, GAE 等一堆超参。DPO 主要只需要调一个 $\beta$。

---

## 4. DPO 在顶尖模型 (GPT-5.2 / Claude 4.6) 中的应用分析
针对您提到的 Chat-GPT 5.2 和 Claude 4.6（假设为最新的 SOTA 模型），业界对于 DPO 的态度发生了一些分化：

### (a) 为什么它们可能**不用**纯离线 DPO？
虽然 DPO 稳定且高效，但在**推理 (Reasoning)** 和 **逻辑 (Math/Code)** 领域，DPO 存在明显短板：
*   **分布偏移 (Distribution Shift)**：DPO 只能学习数据集里已有的好坏对比。如果模型在推理过程中走到了数据集没覆盖的路径，DPO 无法提供反馈。
*   **上限受限**：DPO 的上限是数据集的质量。而 PPO（或 **Iterative DPO / Online DPO**）允许模型通过**探索 (Exploration)** 发现比人类标注数据更好的解法（类似于 AlphaGo 自己跟自己下棋）。

### (b) SOTA 模型的选择：Online RL 是王道
对于像 GPT-5 这样追求极致逻辑能力的模型，更有可能采用的是 **Online RL (Iterative Training)**：
1.  **Online DPO / IPO**：模型生成一批新数据 -> 奖励模型打分 -> 构建 DPO 数据 -> 训练一步 -> 循环。
2.  **Search & Learning (STaR)**：通过思维链（CoT）搜索，只把做对的步骤加入训练集。

### (c) 综合应用
通常的做法是组合拳：
*   **通用对话 / 风格 / 安全性**：使用 **Offline DPO**，因为数据量大且容易标注，效果极其稳定。
*   **数学 / 代码 / 复杂推理**：使用 **PPO / Online DPO / GRPO (Group Relative Policy Optimization)**，因为需要模型自我探索突破上限。

---

## 5. 总结

| 特性 | SFT (模仿学习) | DPO (偏好优化) | PPO (在线强化学习) |
| :--- | :--- | :--- | :--- |
| **核心逻辑** | 提高 $P(y_{good})$ | 提高 $P(y_{w})$，降低 $P(y_{l})$ | 探索生成 $\to$ 获取奖励 $\to$ 更新策略 |
| **稳定性** | 极高 | 高 (类似监督学习) | 低 (对超参极其敏感) |
| **Reward Model** | 不需要 | 不需要 (隐式集成) | **必须** |
| **上限** | 数据集平均水平 | 数据集最优水平 | **可能超越人类 (通过探索)** |
| **适用场景** | 知识注入，格式规范 | 风格对齐，减少幻觉/有害性 | 复杂逻辑，数学推理，代码生成 |

