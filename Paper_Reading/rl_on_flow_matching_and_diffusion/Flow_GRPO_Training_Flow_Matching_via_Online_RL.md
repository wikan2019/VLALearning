# Flow-GRPO: Training Flow Matching Models via Online RL

> **论文信息**
> - **标题**: Flow-GRPO: Training Flow Matching Models via Online RL
> - **作者**: Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang
> - **机构**: CUHK MMLab, 清华大学, 快手 Kling, 南京大学, 上海 AI Lab
> - **发表**: arXiv:2505.05470, 2025 年 5 月（NeurIPS 2025）
> - **arXiv**: [2505.05470](https://arxiv.org/abs/2505.05470)
> - **PDF**: [arxiv.org/pdf/2505.05470](https://arxiv.org/pdf/2505.05470)
> - **代码**: [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)
> - **关键词**: Flow Matching、online RL、GRPO、ODE-to-SDE、Denoising Reduction、可验证奖励

---

## 目录

1. [论文动机与核心问题](#1-论文动机与核心问题)
2. [方法概述](#2-方法概述)
3. [技术细节](#3-技术细节)
4. [实验结果](#4-实验结果)
5. [消融与分析](#5-消融与分析)
6. [讨论与局限](#6-讨论与局限)
7. [和前两篇的关系](#7-和前两篇的关系)

---

## 1. 论文动机与核心问题

Flow Matching / Rectified Flow 已是图像（SD3、FLUX）和视频的主流骨干：ODE 少步采样、质量高。但组合生成（数量、空间关系、属性绑定）和文字渲染仍然差。LLM 上 online RL（PPO / GRPO，R1 / o1）把推理拉起来了；扩散侧已有 DDPO 一类政策梯度，Flow 侧此前主要是 **offline** DPO / RWR（见 VideoAlign）。

把 GRPO 接到 Flow 上有两个硬冲突：

1. **ODE 是确定性的。** 初值一固定，轨迹唯一。GRPO 需要组内多样本、逐步 \(p(x_{t-1}\mid x_t)\) 和探索；Dirac 转移既没有密度，也几乎没有随机性。
2. **在线采样太贵。** 高质量推理要几十步去噪，每步 GRPO 还要一组候选。

Flow-GRPO 的回答：ODE→SDE（边缘不变，逐步变高斯）+ 训练时减少去噪步数（Denoising Reduction）。

---

## 2. 方法概述

```
prompt 集合
    → SDE 采样（训练 T=10）得到一组图 {x0^i} 及整条轨迹
    → 规则 / PickScore 打分，组内标准化得到 A
    → 对每步算高斯 logπ_θ / logπ_old，GRPO + KL
    → 更新同一个 v_θ
推理仍用原来的 ODE（SD3.5-M 默认 T=40）
```

网络结构不变，只换采样器和损失。贡献三点：第一次把 GRPO 接到 Flow；训练不必用满推理步数；KL 能压奖励黑客，且**不等于早停**（无 KL 的高奖励，加 KL 后用更长训练也能追上，同时保住画质）。

---

## 3. 技术细节

### 3.1 Flow 作为 MDP

Rectified Flow：\(x_t=(1-t)x_0+t x_1\)，回归 \(v=x_1-x_0\)。

去噪 MDP 与 DDPO 同构：\(s_t=(c,t,x_t)\)，\(a_t=x_{t-1}\)，奖励只在最终图上。差在默认策略：DDPO 的 \(p(x_{t-1}\mid x_t)\) 已是高斯；Flow 的欧拉步是

\[
x_{t+\Delta t}=x_t+\Delta t\,v_\theta
\]

没有概率。

### 3.2 GRPO 目标

同一 prompt 采 \(G\) 张：

\[
\hat A^i=\frac{R(x_0^i,c)-\mathrm{mean}(R)}{\mathrm{std}(R)}
\]

\[
r_t^i(\theta)=\frac{p_\theta(x_{t-1}^i\mid x_t^i,c)}{p_{\theta_\text{old}}(x_{t-1}^i\mid x_t^i,c)}
\]

\[
\mathcal{J}=\mathbb{E}\Big[\frac1G\sum_i\frac1T\sum_t\min\bigl(r_t\hat A,\;\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat A\bigr)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_\text{ref})\Big]
\]

没有 critic；baseline 是组内均值。这是 online、近 on-policy：数据来自 \(\pi_{\theta_\text{old}}\)。

### 3.3 ODE → SDE（不改结构）

构造与 ODE **各时刻边缘相同** 的反向 SDE：

\[
\mathrm{d}x_t=\Big(v_t-\frac{\sigma_t^2}{2}\nabla\log p_t\Big)\mathrm{d}t+\sigma_t\,\mathrm{d}w
\]

Rectified Flow 下可写成只含 \(v_\theta\) 的形式，Euler–Maruyama：

\[
x_{t+\Delta t}
=x_t+\Big[v_\theta+\frac{\sigma_t^2}{2t}\bigl(x_t+(1-t)v_\theta\bigr)\Big]\Delta t
+\sigma_t\sqrt{\Delta t}\,\epsilon
\]

\(\sigma_t=a\sqrt{t/(1-t)}\)，\(a\) 控探索。于是

\[
\pi_\theta(x_{t+\Delta t}\mid x_t)=\mathcal{N}(\mu_\theta,\;\sigma_t^2\Delta t\,I)
\]

\(\log\pi\)、KL 都有闭式。实现：存下 \(\{x_t\}\)，再前向 \(v_\theta\) 算 \(\mu_\theta\)，对**已经走到的** \(x_{t+\Delta t}\) 做高斯 `log_prob`。奖励仍只看最终图。

并发工作若让网络同时预测速度均值和方差，需要改头、往往重训；本文刻意只改求解器。

### 3.4 Denoising Reduction

训练采集用 \(T=10\)，推理仍 \(T=40\)。少步样本更糊，但对「相对谁更好」仍然有信息；采集成本大约 4× 下降，最终奖励几乎不掉。再减到 \(T=5\) 不稳定。这是 train–test 步数不一致，靠的是 RL 学的是「往高奖励走」，不是模仿某条精确 ODE 轨迹。

---

## 4. 实验结果

基座 **SD3.5-M**。三个任务：

| 任务 | 奖励 | 结果（加 KL） |
|------|------|----------------|
| 组合生成 GenEval | 检测器规则（计数 / 位置 / 颜色） | overall **0.63 → 0.95**，超过 GPT-4o 的 0.84 |
| 文字渲染 | 编辑距离 \(r=\max(1-N_e/N_\text{ref},0)\) | OCR **0.59 → 0.92** |
| 人类偏好 | PickScore | PickScore 21.72 → 23.31，画质指标不掉 |

GenEval 细项：Counting 0.50→0.95，Position 0.24→0.99，Attr. Binding 0.52→0.86。

和 SFT、Flow-DPO（含 online 每 40 步更新采集模型）比，Flow-GRPO 明显更高。Online DPO 好于 offline DPO，但仍低于 GRPO；\(\beta\) 过小 DPO 会崩。

DrawBench 上用 Aesthetic / DeQA / ImageReward / UnifiedReward 盯黑客：任务分涨的同时，**加 KL 后面质量基本持平**。作者原文：几乎没有出现奖励黑客，奖励提升没有以画质或多样性明显下降为代价。

T2I-CompBench++（训练是 GenEval 模板，测试是更开放的组合）上也涨，尤其 2D-Spatial 0.285→0.545。未见物体类、把计数从 2–4 泛化到 5–6 甚至 12，都有正迁移。

---

## 5. 消融与分析

**KL。** 组合生成 / OCR 上去掉 KL，任务分仍高，但 DrawBench 画质明显掉（如 GenEval 设定下 DeQA 4.07→2.77）。PickScore 任务上去掉 KL 画质分不一定掉，但 **多样性塌成一种风格**（不同 seed 几乎一样）。KL 不是早停：配好之后能追到无 KL 的任务分，只是更慢。

**噪声强度 \(a\)。** 过小（0.1）探索不够、奖励涨得慢；中等（约 0.7）最好；再大画质坏掉、奖励归零、训练失败。

**组大小 \(G\)。** PickScore 上 \(G=6,12\) 会不稳直至崩；\(G=24\) 稳定。组太小则 \(A\) 的方差太大。

**可验证奖励 vs 审美 RM。** GenEval / OCR 这种规则分最能体现 online RL 的优势（和 LLM 的 RLVR 同类）；PickScore 也能涨，但更依赖 KL 保多样性。

---

## 6. 讨论与局限

- 实验以图像 T2I 为主；视频 rollout 更贵，SDE 对时间一致性更伤，不能默认同样 0.63→0.95。
- 训练 \(T=10\)、测试 \(T=40\) 是经验，缺充分理论。
- 需要存整条轨迹再算 log-prob，显存和实现比 DPO 重。
- 成功高度依赖 **奖励可检验、组内有区分度**。模糊审美 RM 上仍可能黑，只是本文用 KL + 可验证任务把黑客压住了。

---

## 7. 和前两篇的关系

| | DDPO | VideoAlign | Flow-GRPO |
|---|---|---|---|
| 骨干 | 扩散 | 视频 Flow | 图像 Flow |
| 逐步 \(\log\pi\) | 天生有 | 不用 | ODE→SDE 造出来 |
| 数据 | 在线 rollout | 离线偏好 / RM 重标 | 在线 rollout |
| 算法 | REINFORCE / PPO-IS | DPO / RWR / 引导 | GRPO（无 critic） |
| 黑客 | 指出但未系统治 | 恒定 \(\beta\) | KL + 可验证奖励 |

谱系可以记成：

```
DDPO:  扩散反向高斯 → 证明 online PG 可行
VideoAlign: Flow 上先把 offline DPO 做对（并指出 β_t 不能抄扩散）
Flow-GRPO: 给 Flow 补上逐步高斯，把 LLM 的 GRPO 接进来
```

同构的实现细节：高分轨迹 → \(A>0\) → 最大化该轨迹的逐步 \(\log\pi\) → \(\mu_\theta\) 被拉向实际走到的 \(x_{t+\Delta t}\) → 下次 SDE（及上线 ODE）更常生成高奖励模式。
