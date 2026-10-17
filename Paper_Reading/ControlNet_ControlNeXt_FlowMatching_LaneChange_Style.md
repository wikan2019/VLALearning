# ControlNet / ControlNeXt × Flow Matching：自动驾驶变道风格适配阅读笔记

> **文档用途**  
> 整理「ControlNext / ControlNet 相关改动，如何配合 Flow Matching 做不同变道风格适配」的问答结论与推荐阅读路径，便于后续精读与实验设计。  
> **关键词**: ControlNet, ControlNeXt, Cross Normalization, Flow Matching, Classifier-Free Guidance, 变道风格, 激进度控制, E2E Planning

---

## 目录

1. [核心结论](#1-核心结论)
2. [ControlNet → ControlNeXt：改了什么](#2-controlnet--controlnext改了什么)
3. [Flow Matching 中的风格可控（AD 侧现成工作）](#3-flow-matching-中的风格可控ad-侧现成工作)
4. [落地配方：FM + 变道风格适配](#4-落地配方fm--变道风格适配)
5. [与「直接 ControlNet 做 AD」的边界](#5-与直接-controlnet-做-ad的边界)
6. [推荐论文与阅读顺序](#6-推荐论文与阅读顺序)
7. [后续可展开的精读清单](#7-后续可展开的精读清单)

---

## 1. 核心结论

**ControlNet / ControlNeXt 是视觉生成里的「条件注入」改法**；自动驾驶规划里更常见的做法是把同一思想迁到 **Flow Matching 轨迹生成**——用额外条件（变道方向、激进程度、目标点）引导速度场 \(v_\theta(x_t,t,c)\)，而不是原样搬一套图像 ControlNet。

一句话映射：

| 视觉生成 | 自动驾驶轨迹规划 |
|---------|------------------|
| Canny / Pose / Depth 控制图 | 变道方向、激进度、目标点、plan anchor |
| Control 支路注入 UNet/DiT | 轻量 Control Encoder 注入 FM 骨干 |
| Zero-Conv / CrossNorm | 分布对齐后残差相加，避免冲垮预训练 FM |
| LoRA 改画风 | LoRA / 条件标量改驾驶风格 |

---

## 2. ControlNet → ControlNeXt：改了什么

### 2.1 对比表

| 维度 | **ControlNet** | **ControlNeXt** |
|------|----------------|-----------------|
| 结构 | 复制一整条 UNet 下采样支路，与主模型并行 | 去掉重支路，只用**轻量卷积**提取控制特征 |
| 注入 | Zero-Convolution（初值为 0）慢慢把控制加进主分支 | **Cross Normalization (CN)**：用主分支特征的均值/方差归一化控制特征后再相加 |
| 训练 | 主模型冻结，只训控制支路 | 主模型**选择性微调**一小部分参数（如 attention 线性层） |
| 代价 | 可控性强，但参数多、慢、收敛常「突然跳变」 | 可训练参数约减 **90%**，收敛更快 |
| 风格叠用 | 相对笨重 | 易与 **LoRA** 叠用，改风格无需重训控制支路 |

### 2.2 ControlNeXt 关键设计点

1. **轻量控制模块**：控制能力主要由「主模型局部微调」承担，控制支路只负责提特征与对齐。  
2. **Cross Normalization**：解决控制特征与去噪中间特征分布不匹配导致的训练不稳定；替代 Zero-Conv 的慢启动。  
3. **早期注入 + 残差**：在首个 block 之后注入，保留主分支恒等路径。  
4. **Plug-and-play + LoRA**：同一控制模块可搭不同 LoRA 权重换风格。

### 2.3 迁到轨迹规划时的类比

- **ControlNet 式**：独立 control encoder（风格 / 变道标签）→ 多尺度加到 FM 骨干（可控更强、参数更多）。  
- **ControlNeXt 式**：轻量 control MLP + CrossNorm 对齐后注入，并只解冻少量层或加 LoRA（更适合在已有 GoalFlow / GuideFlow 上增量改）。

**论文与代码**

- 论文: [ControlNeXt: Powerful and Efficient Control for Image and Video Generation](https://arxiv.org/abs/2408.06070)  
- 项目页: https://pbihao.github.io/projects/controlnext/index.html  
- 代码: https://github.com/dvlab-research/ControlNeXt  
- 经典 ControlNet: Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)

---

## 3. Flow Matching 中的风格可控（AD 侧现成工作）

图像 ControlNet 很少直接用于变道规划；AD 侧已有更对口的路线：

| 工作 | 条件如何进入 FM | 与「变道风格」的关系 |
|------|-----------------|---------------------|
| **GuideFlow / CATG** | plan anchor / goal / command + **激进度 EP∈[0,1]**，并用 CFG | 推理时调 EP≈1 → 更激进；可区分抢道式并线 vs 保守并线 |
| **GoalFlow** | **目标点**强约束 FM 生成 | 目标点选到左/右邻车道不同纵向位置 → 不同变道落点与时机 |
| **Flow Planner** | CFG 对邻车等条件重加权 | 调 guidance 强度 \(\omega\)，改变对交互的激进/保守响应 |
| **Transfusor**（扩散，非 FM） | 离散标签：左右变道 + 低/中/高激进 | 最直接的「变道风格可控」标注范式，可迁到 FM |

### 3.1 GuideFlow / CATG 的风格控制要点

- 用 NavSim 相关的 **EP（Ego Progress，沿车道中心线进度）** 作为激进度条件，范围 \([0,1]\)。  
- 训练：对 GT 轨迹在线算 EP，作为条件输入；配合条件 dropout（CFG）。  
- 推理：直接设定 EP（接近 1 更激进，接近 0 更保守）。  
- 同时有安全约束注入（CVF / CF / EBM 等），避免「为了风格失控」。

### 3.2 GoalFlow 的目标点约束

- 先从目标点词表选 goal，再 Flow Matching 生成轨迹。  
- 变道场景下：goal 落在左/右邻车道不同位置，等价于控制「何时变、变到哪」。  
- 推理步数少（可到 1 步量级），适合作为骨架再叠风格条件。

### 3.3 Flow Planner 的 CFG 交互重加权

\[
\tilde{v}_t = (1-\omega)\,v_t^{\text{uncond}} + \omega\,v_t(x_t,t,C)
\]

- 训练时对条件（如邻车信息）做 Bernoulli mask。  
- 推理时增大 \(\omega\) 强化「必须响应交互」；降低 \(\omega\) 更接近无条件/默认行为。  
- 对变道：可把 \(C\) 拆成「场景 + 变道标签 + 风格标量」，对风格子条件单独做 CFG。

---

## 4. 落地配方：FM + 变道风格适配

把变道风格拆成两维条件，再注入 FM 速度场 \(v_\theta(x_t,t,c)\)。

### 4.1 条件设计

**A. 意图条件（离散）**

- 左变道 / 右变道 / 保持车道  
- 可选：急变道 vs 缓变道（由横向加速度峰值、变道完成时间聚类得到）

**B. 风格条件（连续标量，更像「可控强度」）**

- 激进度 \(s \in [0,1]\)，可由下列信号归一化合成：  
  - EP / 进度速度  
  - 变道完成时间（越短越激进）  
  - 相对邻车时距 / TTC  
  - 峰值横向加速度 / 曲率变化率  
- 训练：从 GT 轨迹在线算 \(s\)，并以概率 \(p\) dropout（CFG）  
- 推理：策略或用户直接给定 \(s\)（如保守 0.2 / 中性 0.5 / 激进 0.9）

### 4.2 结构注入（借鉴 ControlNet / ControlNeXt）

```
场景特征 (BEV / Agent / Map)
        ↓
   FM 轨迹骨干（预测 v_t 或直接预测 x_1）
        ↑  CrossNorm / 残差相加
轻量 Control Encoder(变道标签 + 风格标量 s)
        ↑ 可选：目标点 / lane-change 锚轨迹
           （GoalFlow / GuideFlow 式强约束）
```

建议优先级：

1. **先做 GoalFlow / GuideFlow 骨架**（场景 + goal/anchor + FM）  
2. **再加风格标量 \(s\) + CFG**（GuideFlow EP 路线，改动最小）  
3. **注入方式按 ControlNeXt**：轻量 encoder + CrossNorm，必要时 **LoRA** 做车队/城市风格适配  
4. 若必须「强贴合某条参考变道曲线」，再升级为 **ControlNet 式多尺度注入**

### 4.3 采样时再控一把

\[
\tilde{v}_t = (1-\omega)\,v_t^{\text{uncond}} + \omega\,v_t(x_t,t,c_{\text{lane}},s)
\]

- \(\omega\) 越大，变道方向与风格越「硬」。  
- 再叠加安全机制：GuideFlow 的 CVF/CF/EBM，或 ResAD 式「惯性参考 + 残差」等，避免风格压过安全。

### 4.4 数据标注（成败关键）

对每条变道段自动打标签：

| 字段 | 示例 |
|------|------|
| 方向 | left / right / keep |
| 完成时间分位数 | 快 / 中 / 慢 → 映射到 \(s\) |
| 相对时距 / TTC | 激进切入 vs 留足间隙 |
| 合成风格 \(s\) | 连续标量，训练用 GT，推理可指定 |

注意：稀有激进样本需 **CFG + 过采样 / 重加权**，否则模型易塌到「中庸变道」。

---

## 5. 与「直接 ControlNet 做 AD」的边界

- **DriveDreamer-2** 等：ControlNet 用在 **BEV 地图 / 驾驶视频合成**，不是 ego 变道轨迹本身。  
- **做规划变道风格**：优先参考 **GuideFlow / GoalFlow / Flow Planner + 风格条件**；把 ControlNeXt 的 **轻量注入 + CrossNorm + LoRA** 当作工程改法。  
- **ResAD**（归一化残差轨迹建模）与风格控制正交：可先简化轨迹表示与优化，再在残差空间上做条件生成。本地 PDF：[`ResAD_Normalized_Residual_Trajectory_Modeling_CVPR2026.pdf`](ResAD_Normalized_Residual_Trajectory_Modeling_CVPR2026.pdf)。

---

## 6. 推荐论文与阅读顺序

### 6.1 推荐阅读顺序（建议按此精读）

| 顺序 | 论文 | 为什么先读它 | 链接 |
|:----:|------|--------------|------|
| **1** | ControlNet | 理解「条件支路 + Zero-Conv」经典范式 | [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) |
| **2** | **ControlNeXt** | 理解轻量注入、CrossNorm、LoRA 叠风格；为迁到 FM 打基础 | [arXiv:2408.06070](https://arxiv.org/abs/2408.06070) · [GitHub](https://github.com/dvlab-research/ControlNeXt) |
| **3** | **GuideFlow** / CATG | **最贴「FM + 激进度/风格控制」**；EP 条件可直接借鉴 | [GuideFlow arXiv:2511.18729](https://arxiv.org/abs/2511.18729) · [CATG / Beyond Imitation](https://arxiv.org/html/2510.26292) |
| **4** | **GoalFlow** | 目标点约束变道落点与时机；FM 规划强基线 | [arXiv:2503.05689](https://arxiv.org/abs/2503.05689) · [GitHub](https://github.com/YvanYin/GoalFlow) · CVPR 2025 |
| **5** | **Flow Planner** | CFG 调交互强弱；邻车条件 mask 的训练技巧 | [arXiv:2510.11083](https://arxiv.org/abs/2510.11083) · [GitHub](https://github.com/DiffusionAD/Flow-Planner) |
| **6** | **Transfusor** | 变道方向 × 激进等级的标注与条件融合范式（扩散，可迁 FM） | [arXiv:2308.14943](https://arxiv.org/abs/2308.14943) |

### 6.2 扩展阅读（按需）

| 论文 | 用途 | 链接 |
|------|------|------|
| ResAD | 轨迹残差 + PRNorm，优化再平衡；可与条件 FM 正交组合 | [arXiv:2510.08562](https://arxiv.org/abs/2510.08562) · [本地 PDF](ResAD_Normalized_Residual_Trajectory_Modeling_CVPR2026.pdf) |
| DiffusionDrive | 锚定词表的多模态扩散规划；对比「静态词表 vs 动态条件」 | CVPR 2025 相关工作 |
| T2I-Adapter | 比 ControlNet 更轻、可控偏弱的对照 | 与 ControlNeXt 对照读 |
| Classifier-Free Guidance | 理解 \(\omega\) 如何在采样时放大条件 | Ho & Salimans 等 CFG 文献 |
| Rectified Flow / Flow Matching 基础 | 理解 \(v_t\)、直线路径、少步 ODE | Lipman et al. / Liu et al. Rectified Flow |

### 6.3 建议的精读问题清单

读 GuideFlow / GoalFlow / Flow Planner 时，可带着以下问题记笔记：

1. 风格/意图条件具体是什么张量？如何编码进 cross-attn / MLP？  
2. 训练时条件 dropout 概率与推理 \(\omega\) 如何设置？有无消融？  
3. 安全约束是在 **训练** 还是仅在 **推理** 注入？  
4. 变道相关指标：DAC、EP、TTC、LK 等是否随风格条件单调变化？  
5. 若改成 ControlNeXt 式注入，需要动哪些模块、冻结哪些参数？

---

## 7. 后续可展开的精读清单

- [ ] 精读 ControlNeXt：画清 CrossNorm 公式与注入位置  
- [ ] 精读 GuideFlow：整理 EP 条件、CFG、CVF/CF/RFE 三件套  
- [ ] 精读 GoalFlow：goal vocabulary 构建与 scorer  
- [ ] 精读 Flow Planner：交互 CFG 与 tokenization  
- [ ] 对照 Transfusor：定义本项目的变道风格标签体系  
- [ ] 设计一版「GoalFlow 骨架 + EP/风格标量 + ControlNeXt 注入」模块接口与伪代码  
- [ ] （可选）与 ResAD 残差表示结合：在归一化残差空间上做条件 FM  

---

## 参考速查

```
视觉可控生成:     ControlNet → ControlNeXt (轻量 + CrossNorm + LoRA)
AD 轨迹生成:      GoalFlow / GuideFlow / Flow Planner (Flow Matching)
风格控制首选:     连续激进度 s 或 EP + CFG；离散变道方向标签
工程注入首选:     ControlNeXt 式轻量支路，而非完整 ControlNet 复制
安全兜底:         GuideFlow 约束 / 锚点 / 能量修正；勿让风格压过约束
```

---

*文档整理自仓库内关于 ControlNet / ControlNeXt / Flow Matching 变道风格适配的讨论，便于后续详细阅读与实验设计。*
