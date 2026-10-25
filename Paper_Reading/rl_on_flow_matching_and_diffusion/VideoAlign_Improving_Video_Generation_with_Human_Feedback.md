# Improving Video Generation with Human Feedback (VideoAlign)

> **论文信息**
> - **标题**: Improving Video Generation with Human Feedback
> - **作者**: Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, Wanli Ouyang
> - **机构**: CUHK MMLab, 清华大学, 快手 Kling, 上海交通大学, 上海 AI Lab
> - **发表**: arXiv:2501.13918, 2025 年 1 月（NeurIPS 2025）
> - **arXiv**: [2501.13918](https://arxiv.org/abs/2501.13918)
> - **PDF**: [arxiv.org/pdf/2501.13918](https://arxiv.org/pdf/2501.13918)
> - **项目主页**: [gongyeliu.github.io/videoalign](https://gongyeliu.github.io/videoalign/)
> - **关键词**: 视频生成、RLHF、Flow Matching、偏好数据、VideoReward、Flow-DPO、Flow-RWR、Flow-NRG

---

## 目录

1. [论文动机与核心问题](#1-论文动机与核心问题)
2. [方法概述](#2-方法概述)
3. [偏好数据与 VideoReward](#3-偏好数据与-videoreward)
4. [Flow 对齐算法](#4-flow-对齐算法)
5. [实验结果](#5-实验结果)
6. [讨论与局限](#6-讨论与局限)
7. [和前后两篇的关系](#7-和前后两篇的关系)

---

## 1. 论文动机与核心问题

视频生成（尤其 rectified flow）已经能出较真的画面，但仍有运动发飘、提示词对不齐、和人的偏好不一致。语言和图像上 RLHF 已验证；视频这边当时还早，卡在两件事：

**奖励不可靠。** 已有偏好集多来自 Sora 之前的短、低分视频。用它们训出的 RM 会漏掉细空间细节和长程动态，又对现代模型已经压住的闪烁过度惩罚。VLM 奖励的标注范式、多属性纠缠、奖励黑客也缺少系统研究。

**生成器是 Flow，不是扩散。** 现成 Diffusion-DPO / RWR 接到速度场上会出新问题。文中指出：有的 flow DPO 甚至比未对齐基线更差。根子是把扩散里 **随 \(t\) 变的 KL 权重** 原样搬过来，策略会在高噪声段过拟合，图像/视频出现伪影。

本文做完整管线：现代 T2V 上的大规模多维偏好 → 奖励模型 → 统一 KL 正则 RL 目标下的三种 Flow 对齐算法。

---

## 2. 方法概述

```
16k prompts × 12 个 T2V 模型
        → 108k 视频 / 182k 三元组
        → 人工标 VQ / MQ / TA（含平局）
                ↓
        VideoReward（Qwen2-VL-2B + BTT）
                ↓
        ┌───────────┬───────────┬───────────┐
        Flow-DPO    Flow-RWR    Flow-NRG
        （训练期）    （训练期）    （推理期奖励引导）
```

统一目标与 RLHF 相同：

\[
\max_{p_\theta}\ \mathbb{E}[r(x_0,y)]-\beta\,\mathrm{KL}\bigl(p_\theta(\cdot\mid y)\,\|\,p_\text{ref}(\cdot\mid y)\bigr)
\]

三种算法是这个目标的不同求解：DPO 解析成偏好分类，RWR 做成奖励加权回归，NRG 做成推理时改速度场。

注意：这里是 **offline**（DPO/RWR 吃固定偏好或 RM 重标的数据），**没有** DDPO / Flow-GRPO 那种在线 rollout + \(\log\pi\)。

---

## 3. 偏好数据与 VideoReward

### 3.1 数据

- 8 类 prompt（动物、建筑、食物、人、植物、场景、载具、物体），GPT-4o 扩写后人工清洗，得到 16k。
- 12 个 T2V：含 Gen2 / SVD / Pika 等早期模型，以及 Dreamina、Luma、Gen3、Kling 1.0/1.5、PixVerse v2 等现代模型。
- 每个三元组：同一 prompt + 两个不同模型的视频；标注 **Visual Quality / Motion Quality / Text Alignment** 的 A 赢 / 平 / B 赢，另有 1–5 分点式分。
- 留 13k 三元组作验证（prompt 不与训练重叠）。
- 另建 **VideoGen-RewardBench**：在 VideoGen-Eval 上补 26.5k 现代视频对的人工偏好，用来评当代 RM。

### 3.2 奖励模型设计（消融结论）

骨干：Qwen2-VL-2B。

| 选择 | 结论 |
|------|------|
| BT 成对 vs 点式回归 | 数据变多两边都涨，**BT 一直更好**（成对能分开点式分相同的细微差） |
| 是否建模平局 | 普通 BT 会把平局对打出很大 \(\Delta r\)；**BTT（Bradley-Terry with Ties, \(\theta=5\)）** 让平局聚在 0 附近 |
| token 位置 | 最后 token 一个头会把「画质」和 prompt 缠在一起（换 prompt 同一视频 VQ 变了）。把 `[VQ][MQ]` 放在视频后、prompt 前，`[TA]` 放在 prompt 后 |

训练时 2 fps、约 \(448\times448\) 保比例；更高分辨率、更多帧、按时间间隔而不是固定帧数，对运动质量更重要。

VideoGen-RewardBench（现代视频）上，VideoReward 整体准确率约 **61.3%（含平局）/ 73.6%（不含平局）**，高于 VideoScore、LiFT、VisionReward。早期短视频榜 GenAI-Bench 上仍有竞争力，说明没有完全过拟合现代模型。

---

## 4. Flow 对齐算法

Rectified Flow：\(x_t=(1-t)x_0+t x_1\)，网络回归 \(v=x_1-x_0\)。

### 4.1 Flow-DPO

把 Diffusion-DPO 的噪声误差换成速度误差。关系是

\[
\|\epsilon^*-\epsilon_\text{pred}\|^2=(1-t)^2\|v^*-v_\text{pred}\|^2
\]

直接代进去会得到 **时变** KL 系数 \(\beta_t=\beta(1-t)^2\)：\(t\to1\)（纯噪声）时惩罚消失，模型在高噪声段猛对齐。扩散里这套还凑合；**Flow 上会奖励黑客和伪影**。

本文改成 **全程恒定 \(\beta\)**，各维对齐更稳。直觉：让 \(v_\theta\) 靠近赢家速度、远离输家速度，强度由「当前误差 vs 参考模型误差」的差决定。

### 4.2 Flow-RWR

最优策略 \(p_\theta\propto p_\text{ref}\exp(r/\beta)\) 的 EM 近似：用 \(\exp(r)\) 加权速度回归。同样丢掉 \((1-t)^2\)。比 DPO 弱，因为没有成对「赢/输」的对比结构。

### 4.3 Flow-NRG（推理期）

把 \(p_\text{ref}\exp(w\cdot r)\) 写成速度修正：

\[
\tilde v_t(x_t\mid y)=v_t(x_t\mid y)-w\frac{t}{1-t}\nabla r(x_t,y)
\]

类似 classifier guidance。像素空间反传要过 VAE，太贵，于是在 **noisy latent** 上另训轻量时变奖励 \(r_\theta(\cdot,t)\)（同一对视频加相同噪声，假设相对偏好不变），复用生成器前几层。用户可在推理时给 VQ/MQ/TA 不同 \(w\)，不用为每种口味重训。

对齐实验对内部 Transformer + rectified flow 视频模型做 **LoRA**；训练期用 VideoReward 当模拟人类反馈重标数据。

---

## 5. 实验结果

多维对齐（VQ:MQ:TA = 1:1:1）要点：

| 方法 | 现象 |
|------|------|
| SFT / Flow-RWR | 有涨，幅度有限 |
| Flow-DPO，\(\beta_t=\beta(1-t)^2\) | VQ/MQ 赢率可以很高，但 **奖励黑客明显**，语义/TA 崩 |
| Flow-DPO，恒定 \(\beta\) | VQ/MQ/TA 同时高，黑客少。例如 VideoGen-Eval 上 VQ 赢率约 90.95，TA 约 68.26；TA-Hard 上 TA 约 73.24 |
| Flow-NRG | 不改权重即可调多目标配比 |

单维对齐 TA 时，恒定 \(\beta\) 的 Flow-DPO 仍是最稳的训练期方法。

---

## 6. 讨论与局限

- 视频 RLHF 当时的主瓶颈是 **RM 质量和多属性是否可分**，不是还没有 GRPO。本文把 RM 和 offline 对齐做扎实，没有走 online 政策梯度。
- 生成器是内部模型 + LoRA，复现依赖开源视频 Flow 骨干。
- 含平局的 RM 准确率仍只有约 61%，给后面 online RL 留下「奖励不可靠」的隐患（Flow-GRPO 在图像上改用可验证奖励，部分绕开这个问题）。
- Flow-DPO 的关键教训：**扩散的 \(\beta_t\) 日程不能直接抄到 rectified flow。**

---

## 7. 和前后两篇的关系

- 相对 **DDPO**：视频 Flow 没有逐步高斯 \(\log\pi\)，本文不改采样器，用 DPO/RWR 在速度场上做 **offline** 对齐。
- 相对 **Flow-GRPO**：同一批作者线（Liu / 快手 Kling / Ouyang）。VideoAlign 解决「视频偏好数据和 RM」；Flow-GRPO 四个月后在**图像 Flow** 上把 online GRPO 跑通（ODE→SDE）。图像可验证任务（计数、OCR）让 online RL 的奖励比视频审美分干净得多。
