# HiDrop: Hierarchical Vision Token Reduction in MLLMs

> **论文信息**
> - **标题**: HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit
> - **作者**: Hao Wu*, Yingqi Fan*, Jinyang Dai, Junlong Tong, Yunpu Ma, Xiaoyu Shen†
> - **机构**: Eastern Institute of Technology (宁波东方理工大学数字孪生研究所) / 中科大 / 上海交大 / LMU Munich
> - **发表**: arXiv:2602.23699v1, 2026年2月
> - **代码**: https://github.com/EIT-NLP/HiDrop
> - **PDF**: [HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.pdf](HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.pdf)
> - **关键词**: MLLM、视觉 Token 剪枝、Late Injection、Concave Pyramid Pruning、Early Exit、FlashAttention

---

## 目录

1. [论文概述](#1-论文概述)
2. [研究动机与核心问题](#2-研究动机与核心问题)
3. [关键发现：MLLM 的分层处理动力学](#3-关键发现mllm-的分层处理动力学)
4. [方法详解：HiDrop](#4-方法详解hidrop)
5. [工程落地：消除动态剪枝的隐性开销](#5-工程落地消除动态剪枝的隐性开销)
6. [实验结果](#6-实验结果)
7. [消融实验](#7-消融实验)
8. [关键创新点总结](#8-关键创新点总结)
9. [个人思考与评价](#9-个人思考与评价)

---

## 1. 论文概述

HiDrop 是面向 **Multimodal Large Language Models (MLLMs)** 的**分层视觉 Token 压缩框架**。核心主张：剪枝不应按固定比例“一刀切”，而应**对齐 MLLM 各层真实的功能分工**。

| 组件 | 作用阶段 | 一句话 |
|------|----------|--------|
| **Late Injection** | 浅层 | 跳过被动浅层，在真正开始融合时再注入视觉 token |
| **Concave Pyramid Pruning** | 中层 | 凹形日程：融合早期猛剪、后期缓剪 |
| **Early Exit** | 深层 | 语言主导阶段彻底丢掉剩余视觉 token |
| **ILVAS + Differentiable Top-K** | 中层选层/选 token | 决定“在哪剪”和“剪谁” |

**主要结果（LLaVA-1.5-7B）**：

- 平均压缩约 **∼90%** 视觉 token（576 → 64 / 48）
- 相对原模型保留约 **96.5%–98.4%** 平均性能
- 训练时间约 **1.72×** 加速（159.3 → 94.4 GPU hours）
- 推理视觉相关 FLOPs 约降 **88.9%**（3.82T → 0.42T）

---

## 2. 研究动机与核心问题

### 2.1 为什么要剪视觉 Token？

现代 MLLM 多为 **connector 架构**：视觉编码器把图像切成大量 patch，投影进 LLM 嵌入空间。视觉 token 数 \(N_v\) 通常远大于文本 token 数 \(N_t\)，而 Self-Attention 代价约为 \(\mathcal{O}(N_v^2 d)\)。分辨率一升，计算量平方暴涨。

```
图像 → ViT → 576 个视觉 token ─┐
文本 → Tokenizer → 文本 token ─┼→ LLM Transformer (Attention O(n²))
                               ↑
                     瓶颈主要来自视觉 token
```

### 2.2 Progressive Pruning 方向对，但现有方法两处误解

已有工作（FastV、TwigVLM、PDrop 等）采用**逐步减少视觉 token**，直觉是：浅层保留细节、深层只留语义重要 token。HiDrop 指出两个根本问题：

| 误解 | 现有做法 | HiDrop 的诊断 |
|------|----------|---------------|
| **误解浅层** | 认为浅层对多模态融合很关键，必须早早保留/处理视觉 | 浅层几乎不变换视觉表征，也几乎不做跨模态融合——只是**传递器 (propagators)** |
| **刚性日程** | 固定比例、均匀间隔的金字塔剪枝 | 冗余在中层最高、深层可完全丢掉；均匀日程吃不透效率 |

```
现有 progressive pruning 典型形态:

(a) FastV:     早期某一层一次性猛剪
(b) TwigVLM:   早剪 + 深层再清掉剩余视觉
(c) PDrop:     均匀比例、均匀间隔逐步剪
(d) HiDrop:    浅层晚注入 → 中层凹形猛剪 → 深层早退出
```

---

## 3. 关键发现：MLLM 的分层处理动力学

论文用 **training-free 探针**（LLaVA-v1.5-7B + GQA）刻画层间行为，得到三阶段划分。

### 3.1 浅层：Propagators（被动传递）

两项证据：

1. **模态内相似度**：视觉 token 在相邻浅层几乎不变（高 cosine similarity）→ LLM 骨干对视觉几乎不做细化  
2. **跨模态影响**：固定指令换不同图像，浅层指令表征几乎不变 → **跨模态融合尚未发生**

结论：浅层主要是管道，不是融合器。在浅层“认真算视觉 Attention”是浪费。

### 3.2 中层：Sparse Fusion Hubs（稀疏融合枢纽）

- 文本表征开始随图像显著变化 → **真正的融合发生在这里**
- 但融合高度稀疏：少数关键视觉 token 就够“锚定”文本，其余大量冗余

训练式极端剪枝：中层把视觉 token 从 576 压到 **1**，GQA 仍可保留约 **99.6%** 原性能 → 中层是理想的猛剪窗口。

### 3.3 深层：Language-Dominant Reasoning（语言主导推理）

- 视觉直接影响持续下降  
- Training-free “early exit”：在浅/中层丢掉视觉会崩；在融合完成后（如 **layer 24+**）丢掉，性能几乎不掉  

结论：深层可以只靠中层已经融合好的表征做语言推理，不再需要原始视觉 token。

```
层功能示意 (LLaVA-1.5-7B 实验设定):

Layer 1 ─────────── 8/9 ─────────────── 25 ─────────── 32
│  浅层 Propagator  │  中层 Fusion Hub  │  深层 Language │
│  几乎不融合视觉    │  稀疏融合 + 高冗余 │  可不看视觉    │
│  → Late Injection │  → Concave Prune  │  → Early Exit  │
└───────────────────┴───────────────────┴───────────────┘
        注入点 Linj≈9              退出点 Lexit≈25
```

---

## 4. 方法详解：HiDrop

### 4.1 总体框架

把 LLM 层划分为浅 / 中 / 深三档，分别用不同策略处理视觉流：

```
┌─────────────────────────────────────────────────────────────┐
│                         HiDrop                              │
│                                                             │
│  浅层 (1 .. Linj-1):  只跑文本；视觉完全旁路                   │
│           ↓ Late Injection                                   │
│  注入层 Linj:         拼接  h = [视觉 ; 文本]                 │
│           ↓                                                  │
│  中层 + Filtering Layers F:                                  │
│     ILVAS 选“在哪剪” + Differentiable Top-K 选“剪谁”         │
│     Concave Pyramid: 前期 keep ratio 掉得快，后期变缓          │
│           ↓ Early Exit                                       │
│  深层 (Lexit .. L):   丢掉全部视觉，纯文本继续推理              │
└─────────────────────────────────────────────────────────────┘
```

对 LLaVA-1.5-7B 的默认超参：

| 超参 | 取值 | 依据 |
|------|------|------|
| \(L_{\text{inj}}\) | 9 | 视觉层间相似度局部最小点（融合起点） |
| \(L_{\text{exit}}\) | 25 | deep-to-shallow masking 性能平台区 |
| Filtering layers \(\mathcal{F}\) | {10, 14, 16, 18} | ILVAS 曲线局部极大值 |

### 4.2 Late Vision Injection（晚注入）

- **不做**：从第 1 层就带上全部视觉 token 做 Attention  
- **改做**：前 \(L_{\text{inj}}-1\) 层纯文本前向；到 \(L_{\text{inj}}\) 再注入完整视觉 token  

意义：

1. 浅层不再为视觉付 \(\mathcal{O}(N_v^2)\) 代价  
2. 不在“还没融合”时过早丢掉可能有用的信息（对比“浅层就猛剪”）  
3. 首次把“**故意延迟视觉输入**”当成效率手段，而不仅仅是剪枝  

### 4.3 Early Vision Exit（早退出）

在 \(L_{\text{exit}}\) 之后丢弃全部剩余视觉 token，后续层只处理文本流。  
与 Late Injection 合在一起，形成有限长度的 **“视觉处理窗口”**——视觉只在真正需要的中层存在。

### 4.4 Concave Pyramid Pruning（凹形金字塔剪枝）

中层冗余极高，但不宜均匀剪：

- **融合早期**：可以更激进地降 keep ratio（凹下去）  
- **融合后期**：放缓，保住关键语义 token  

相对 FastV（单点猛剪）和 PDrop（均匀金字塔），凹形日程更贴合“中层先猛后缓”的冗余结构。论文称相对 SOTA progressive 方法，剪枝可约 **4.8× 更激进**，性能几乎不掉。

### 4.5 Where to prune：ILVAS

**Inter-Layer Visual Attention Similarity (ILVAS)**：衡量某一层 top-K 高注意力视觉 token，在后续层是否仍稳定重要。

- ILVAS 高 → 该层对 token 重要性评估已稳定 → 适合作为 **filtering layer**  
- 在中层扫 ILVAS 曲线，取局部极大作为 \(\mathcal{F}\)

### 4.6 Which tokens to prune：Differentiable Top-K

硬 Top-K 不可微，端到端训练不友好。HiDrop 采用 **Differentiable Top-K (DTop-K)**：

1. 把重要性分数 \(c\) 映射为归一化秩分数 \(c'\)  
2. 用可学习阈值 \(a\) + sigmoid 生成 soft mask  
3. 前向仍用硬阈值做离散选择；反向有连续梯度  

重要性估计默认用 **多轮 last-token attention**（相对“全文本 token 聚合 + L2”等方案，效果与成本更均衡）。

---

## 5. 工程落地：消除动态剪枝的隐性开销

动态改变视觉 token 集合，容易出现“理论 FLOPs 降了、实测不加速”。HiDrop 给了三件套：

### 5.1 Persistent Position Encoding

Late injection / progressive drop / early exit 会跨层改变存活视觉集合。若每层重新紧凑编号，RoPE 相对几何会错位。

**做法**：输入时为每个视觉 token 分配**持久位置 ID**——浅层虽无视觉，索引预留；注入后激活；剪枝/退出时 ID 不变。RoPE 始终用固定 ID 旋转 Q/K。

消融：Persistent PE > Group PE > Compacted PE（PDrop 式重编号最差）。

### 5.2 Efficient Attention Compatibility

主 Attention 仍对完整序列走 **FlashAttention** 等高效 kernel；  
token 选择另开一个很轻的辅助 Attention（主要是 **最后一个文本 token ↔ 视觉 token**），开销可忽略。

### 5.3 Parallel Decoupling of Vision-related Operations

Late Injection 使注入前为纯文本：

- 文本侧：跑浅层 Transformer  
- **并行**：跑视觉编码器 + projector，缓存视觉 KV  
- 注入层：拼接文本 KV 与视觉 KV；后续剪枝只更新缓存视觉 KV 上的索引，不必重算 projection  

于是视觉 KV 投影可移出 prefill 关键路径，理论加速更容易变成实测加速。

---

## 6. 实验结果

### 6.1 设置

| 项目 | 内容 |
|------|------|
| 架构 | LLaVA-1.5 |
| Backbone | MobileLLaMA-2.7B / Vicuna-7B / Vicuna-13B |
| 基准 | 11 个：MMEᴾ, MMB, MMBCN, GQA, VQAv2, SQAᴵ, VizWiz, TextVQA, POPE, SEEDᴵ, MMStar |
| 硬件 | 8× A100 40GB |
| 效率指标 | 训练 GPU hours；视觉相关 Infer FLOPs；Prefill latency |

### 6.2 主结果（LLaVA-1.5-7B，同预算对比）

相对原模型（576 tokens = 100%）：

| 平均保留 token | 压缩率 | HiDrop Avg | 代表对比 |
|----------------|--------|------------|----------|
| 80（↓86.1%） | 中等 | **98.4%** | TwigVLM 97.9%，PDrop 96.8% |
| 64（↓88.9%） | 强 | **98.3%** | TwigVLM 95.3%，PDrop 94.2% |
| 48（↓91.7%） | 极强 | **96.5%** | TwigVLM 87.3%；PDrop 同协议下难达此压缩 |

要点：压缩越狠，HiDrop 相对 progressive 基线的优势越大。

### 6.3 训练 / 推理效率（保留约 64 tokens）

| 模型 | 方法 | Avg Vis Tokens | Train hours | Infer TFlops | Prefill (ms) | Avg |
|------|------|----------------|-------------|--------------|--------------|-----|
| 7B | Vanilla | 576 | 159.3 | 3.82 | 63.6 | 100% |
| 7B | PDrop | 270 | 107.3 | 1.78 | 43.7 | ∼100% |
| 7B | **HiDrop** | **64** | **94.4** | **0.42** | **32.6**（可至 28.8） | **∼98.6%** |
| 13B | Vanilla | 576 | 297.2 | 7.44 | 122.8 | 100% |
| 13B | **HiDrop** | **64** | **175.8** | **0.82** | **48.6** | **∼98.7%** |

解读：

- 7B 训练约 **1.69×**（文中写约 1.72× / 40.7% 时间减少）  
- 相对 PDrop：剪得更狠约 **4.8×**，性能仅掉约 **1.6%**  
- Prefill 可通过并行解耦视觉 KV、减少 dropping stages 进一步压低  

---

## 7. 消融实验

| 消融项 | 结论 |
|--------|------|
| **Late Injection / Early Exit 窗口** | \(L_{\text{inj}}=9\)、\(L_{\text{exit}}=25\)（或 8–24 / 9–25 窗口）近最优；更早注入白花钱，更晚注入伤精度；更早退出伤精度，更晚退出白花钱 |
| **Hard vs Differentiable Top-K** | Diff Top-K：PT+FT 下 Avg 97.7% → **99.7%**；仅 FT 也有提升 |
| **Token 重要性估计** | multi-round last-token 与 all-token+L2 接近；前者默认更省 eager attention 成本 |
| **位置编码** | Persistent PE 最好（97.6%）；Compacted（重编号）最差 |
| **Filtering layers** | ILVAS 选出的 {10,14,16,18} 优于均匀控制组 {12,15,18,21} |
| **数据规模** | 指令数据 665k→1M 时，HiDrop 与基座同步涨分；压缩不是瓶颈 |

---

## 8. 关键创新点总结

| # | 贡献 | 意义 |
|---|------|------|
| **C1** | **纠正浅层误解** | 浅层是 propagator 而非 fusion 层 → 引出 Late Injection |
| **C2** | **Late Injection** | 首次系统地把“延迟视觉注入”当作效率手段 |
| **C3** | **Concave Pyramid + Early Exit** | 剪枝日程对齐中层高冗余、深层语言主导的真实动力学 |
| **C4** | **ILVAS + Differentiable Top-K** | 分别回答“在哪剪”“剪谁”，且可端到端训练 |
| **C5** | **工程三件套** | Persistent PE、FlashAttention 兼容选择、视觉计算并行解耦 → 理论加速落到实测 |
| **C6** | **SOTA 效率–精度权衡** | ∼90% token 压缩、性能接近原模型、训练显著加速 |

---

## 9. 个人思考与评价

### 优点

1. **问题诊断清晰**：不是堆另一个启发式 schedule，而是先用探针证明浅/中/深功能不同，再设计方法——叙事完整、说服力强。  
2. **Late Injection 直觉好**：很多工作只在“已进入 LLM 的视觉”上剪；HiDrop 意识到“浅层根本不该为视觉买单”。  
3. **强调实测加速**：Persistent PE / FlashAttention / 并行解耦，针对动态剪枝常见的“FLOPs 好看、latency 不好看”。  

### 局限 / 待观察

1. **主实验集中在 LLaVA-1.5 系**：对 Qwen-VL、LLaVA-NeXT 高分辨率多图等是否同样成立，需更多验证。  
2. **\(L_{\text{inj}}/L_{\text{exit}}/\mathcal{F}\) 带模型相关**：换 backbone 要重标定（文中 2.7B 与 7B 已不同）；自动化程度仍有提升空间。  
3. **极致压缩（48 tokens）** 平均仍掉约 3.5%：对 OCR / 细粒度定位类任务是否足够，要看具体场景。  
4. Conclusion 中出现 “HiDivDrop” 笔误，应为 HiDrop。  

### 与 VLA / 具身方向的关联（可选视角）

若把 VLA 看成“带动作头的 MLLM”，视觉 token 同样是瓶颈。HiDrop 的分层结论（浅层传、中层融、深层语言）对设计：

- **何时把视觉注入策略网络**  
- **Action Chunk 生成前能否 Early Exit 视觉**  
- **高分辨率多相机下的 token 预算**  

都有启发；但机器人控制对细粒度视觉更敏感，直接套用 ∼90% 剪枝需额外压测。

---

## 参考文献（原文核心）

1. **HiDrop** — Wu, Fan, Dai, Tong, Ma, Shen. *HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit*. arXiv:2602.23699, 2026.  
2. **FastV** — Chen et al. *An image is worth 1/2 tokens after layer 2*. ECCV 2024.  
3. **PDrop / PyramidDrop** — Xing et al. *PyramidDrop*. arXiv:2410.17247, 2024.  
4. **TwigVLM** — Shao et al. *Growing a Twig to Accelerate Large Vision-Language Models*. arXiv:2503.14075, 2025.  
5. **Differentiable Top-K** — Liu et al. *Differentiable Model Scaling using Differentiable Topk*. ICML 2024.  
6. **LLaVA-1.5** — Liu et al. *Improved Baselines with Visual Instruction Tuning*.  
