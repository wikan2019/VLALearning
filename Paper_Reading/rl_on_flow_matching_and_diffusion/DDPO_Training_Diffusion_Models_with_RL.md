# Training Diffusion Models with Reinforcement Learning (DDPO)

> **论文信息**
> - **标题**: Training Diffusion Models with Reinforcement Learning
> - **作者**: Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine
> - **机构**: UC Berkeley, MIT
> - **发表**: arXiv:2305.13301, 2023 年 5 月（NeurIPS 2023）
> - **arXiv**: [2305.13301](https://arxiv.org/abs/2305.13301)
> - **PDF**: [arxiv.org/pdf/2305.13301](https://arxiv.org/pdf/2305.13301)
> - **项目主页**: [rl-diffusion.github.io](http://rl-diffusion.github.io)
> - **关键词**: 扩散模型、online RL、政策梯度、去噪 MDP、RWR、DDPO、RLAIF

---

## 目录

1. [论文动机与核心问题](#1-论文动机与核心问题)
2. [方法概述](#2-方法概述)
3. [技术细节](#3-技术细节)
4. [奖励函数设计](#4-奖励函数设计)
5. [实验结果](#5-实验结果)
6. [讨论与局限](#6-讨论与局限)
7. [和后两篇的关系](#7-和后两篇的关系)

---

## 1. 论文动机与核心问题

扩散模型预训练优化的是（变分）似然：让 \(p_\theta(x_0)\) 贴近训练数据。实际使用关心的却是**下游目标**——观感、可压缩性、提示词对齐、分子活性——这些目标往往：

- 写不进 caption，没法靠 prompt engineering 控住
- 不是可微的像素损失
- 也不是整条去噪轨迹的精确 \(\log p(x_0)\)（扩散的精确似然本身就不好算）

因此不能直接对最终样本做最大似然。本文把**逐步去噪**写成多步决策，用每一步高斯转移的精确 \(\log\pi\) 做政策梯度，从而只需要黑盒奖励 \(r(x_0,c)\)。

```
似然预训练:  max  log p_data(x)
下游目标:    max  E_{x~p_θ}[ r(x, c) ]     ← DDPO 要优化的
```

---

## 2. 方法概述

同一目标 \(\mathcal{J}(\theta)=\mathbb{E}_{c,x_0\sim p_\theta}[r(x_0,c)]\) 对应两种 MDP 写法：

| | RWR（奖励加权回归） | DDPO（本文） |
|---|---|---|
| MDP | 单步：动作 = 整张图 \(x_0\) | 多步：动作 = 下一步潜变量 \(x_{t-1}\) |
| 策略密度 | \(p_\theta(x_0\mid c)\)，精确值不可得 | \(p_\theta(x_{t-1}\mid x_t,c)=\mathcal{N}(\mu_\theta,\sigma_t^2 I)\)，闭式 |
| 更新 | 用 \(\exp(\beta r)\) 加权 DDPM 损失 | REINFORCE / 重要性采样政策梯度 |
| 理论 | 加权的是 ELBO，不是真 log-likelihood | 逐步 \(\log\pi\) 精确，梯度对准 \(\mathcal{J}\) |

经验结论：**把去噪当序列决策，比把整张图当一个动作再加权似然更有效。**

---

## 3. 技术细节

### 3.1 反向过程已经是高斯策略

标准 sampler 下：

\[
p_\theta(x_{t-1}\mid x_t,c)=\mathcal{N}\bigl(x_{t-1};\;\mu_\theta(x_t,c,t),\;\sigma_t^2 I\bigr)
\]

这是 DDPO 能算 \(\log\pi\)、能做重要性比率的原因。网络结构不用改，变的是损失：不再回归 \(\tilde\mu\)，而是最大化逐步 log-prob 加权的奖励。

### 3.2 去噪 = 多步 MDP

| 符号 | 对应 |
|------|------|
| 状态 \(s_t\) | \((c, t, x_t)\) |
| 动作 \(a_t\) | \(x_{t-1}\) |
| 策略 \(\pi(a_t\mid s_t)\) | \(p_\theta(x_{t-1}\mid x_t,c)\) |
| 转移 | 确定性：下一状态就是 \((c,t-1,x_{t-1})\) |
| 奖励 | 仅在 \(t=0\) 给 \(r(x_0,c)\)，中间步为 0 |

轨迹累积回报就等于最终图的奖励，所以 \(\max\mathcal{J}_\text{RL}=\max\mathcal{J}_\text{DDRL}\)。

### 3.3 两个梯度估计器

**DDPO\(_\text{SF}\)**（score function / REINFORCE），必须用当前 \(\theta\) 采的数据，一批只能更新一次：

\[
\nabla_\theta\mathcal{J}
=\mathbb{E}\Big[\sum_t\nabla_\theta\log p_\theta(x_{t-1}\mid x_t,c)\;r(x_0,c)\Big]
\]

**DDPO\(_\text{IS}\)**（重要性采样 + PPO 式 clip），可用 \(\theta_\text{old}\) 的轨迹对 \(\theta\) 多步更新：

\[
\nabla_\theta\mathcal{J}
=\mathbb{E}\Big[\sum_t\frac{p_\theta(x_{t-1}\mid x_t,c)}{p_{\theta_\text{old}}(x_{t-1}\mid x_t,c)}
\nabla_\theta\log p_\theta(x_{t-1}\mid x_t,c)\;r(x_0,c)\Big]
\]

后者略强，因为同样的 rollout 能走更多梯度步。这已经是后来 GRPO / Flow-GRPO 的雏形：行为策略采样，目标策略用比率纠偏。

### 3.4 和 RWR 的差别

RWR 把生成看成单步 MDP，用

\[
w=\frac1Z\exp(\beta r)\quad\text{或}\quad w=\mathbf{1}[r\ge C]
\]

去加权 \(\mathcal{L}_\text{DDPM}\)。加权的不是 \(\log p(x_0)\)，只是变分界，因此对 \(\mathcal{J}\) 只是很粗的近似。DDPO 不碰这条界，直接在逐步高斯上做 PG。

---

## 4. 奖励函数设计

基座均为 **Stable Diffusion v1.4**。任务从「好算」到「像真下游」：

1. **JPEG 可压缩 / 不可压缩**  
   分辨率固定 512²，用压缩后文件大小。caption 里几乎没有文件大小，prompt 控不住，适合当 RL 案例。
2. **美学质量**  
   LAION aesthetics predictor（CLIP 上的线性头，约 17.6 万人类打分）。属于 RLHF。
3. **提示词对齐（无额外人工标注）**  
   LLaVA 描述生成图，再和原 prompt 算 BERTScore recall。类似 RLAIF：用 VLM 代替大规模人类对齐标注。

---

## 5. 实验结果

- 三个可计算任务上，**DDPO > RWR**（含稀疏加权）；DDPO\(_\text{IS}\) 略优于 DDPO\(_\text{SF}\)。
- 定性策略符合奖励、也暴露黑客：美学 → 把写实图收成插画；可压缩 → 去背景、抹细节；不可压缩 → 加高频噪声和锐边。
- VLM 对齐：`a(n) [animal] [activity]`（骑车 / 下棋 / 洗碗）。预训练对「海豚骑自行车」成功率为 0，单条 prompt 没有信号；**跨 prompt 迁移**后这类难例也能涨。
- 泛化：45 种动物上微调，能迁到未见动物、非动物物体，以及未见活动（如「考试」）。

---

## 6. 讨论与局限

- 实验 prompt 分布窄（动物 + 少量活动）。
- **过优化 / 奖励黑客**被作者明确标为未系统研究的问题（附录 A）；后来 Flow-GRPO 用 KL、VideoAlign 用恒定 \(\beta\) 专门压这件事。
- 贡献是框架：任意黑盒 \(r(x_0,c)\) + 逐步精确似然 → 政策梯度。不依赖偏好对数据集。

---

## 7. 和后两篇的关系

DDPO 证明：**扩散反向过程天生是随机策略，online PG 可行。**  
Flow Matching 默认是确定性 ODE，没有这一步高斯，所以：

- VideoAlign 先走 **offline DPO/RWR**，避开逐步 \(\log\pi\)
- Flow-GRPO 用 **ODE→SDE** 把 DDPO 的逐步高斯策略搬到 flow 上，再用 GRPO 换掉 critic
