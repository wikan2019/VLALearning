# VLM 常见问题 × Locate / Steer / Improve 阅读地图

> **说明**：本文不是已发表综述，而是对照 LLM 的 *Locate, Steer, and Improve*（[arXiv:2601.14004](https://arxiv.org/abs/2601.14004)），整理 **VLM / MLLM** 上「有没有同构总览、最近的方法综述、以及按问题拆开的 Locate+Steer 单点论文」。
>
> 与仓库内 [VLA 对照草稿](../VLA_Common_Issues_Locate_Steer_Improve.md) 的差别：本文管 **看图说话 / VQA / 幻觉 / 视觉 token 效率**；那篇管 **Vision-Language-Action**。

---

## 1. 结论：没有同级别的 VLM 总览

**没有一份和 *Locate, Steer, and Improve* 同级别、专门覆盖「VLM 常见问题 × Locate / Steer / Improve」的综述。**

LLM 那篇明确写的是 **dense decoder-only LLM**。VLM/MLLM 只在未来方向里点到视觉 token、跨模态注意力；Limitation 也写明 **没有系统覆盖多模态架构**。

因此下面分三层读：

1. **最近的方法综述**（会探针 / patching / SAE，但不按「幻觉、安全、效率」做成应用流水线）
2. **问题专项综述**（尤其是幻觉：成因、评测、缓解，机制定位很少）
3. **单点方法论文**（一类问题的 Locate+Steer，不是总览）——第 4 节展开

---

## 2. 最接近的文档

| 文档 | 管什么 | 缺什么 |
|------|--------|--------|
| **[Mechanistic Interpretability Meets VLMs](https://iclr-blogposts.github.io/2025/blog/vlm-understanding/)**（ICLR 2025 Blogpost, Liu / Zhang / Yeung-Levy） | VLM 上的 **MI 方法**：探针、activation patching、logit lens、SAE、自动解释 | 偏「看懂内部」，没有按幻觉/安全/效率做成 Locate→Steer→Improve 应用表 |
| **幻觉综述**（如 [arXiv:2410.15359](https://arxiv.org/abs/2410.15359)） | 物体幻觉的成因、评测、缓解（数据、训练、解码、后处理） | 大多不是机制定位；很少写成「哪个对象 + 哪种干预」 |
| **单点方法论文**（本文 §4） | 某一类问题的 Locate + Steer | 不是总览 |
| 本仓库 **[HiDrop](../HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.md)**、DeepVision-VLA PDF | 效率（视觉 token 剪枝）、VLA 视觉增强 | 各只覆盖一条问题 |

**推荐读法**：用 LLM 的 LSI 当骨架，把可解释对象换成 **视觉 token / connector / 跨模态头 / 视觉编码器层**，再补第 4 节论文。

---

## 3. VLM 比 LLM 多出来的对象与问题映射

VLM 常见 connector 架构：视觉编码器切 patch → projector 进 LLM。可解释对象在 LLM 那套（残差、头、FFN、SAE）之外，至少还要加：

- 视觉 token \(\mathbf{E}_v\)、图像 patch 注意力
- connector / projector
- 跨模态注意力（文本 query 看图像 key）
- 视觉编码器内部层与 CLS / 寄存器 token

| 常见问题 | 典型 Locate | 典型 Steer / Improve |
|----------|-------------|----------------------|
| **物体幻觉 / 乱编属性** | 视觉贡献图、早层 vs 深层视觉信号、跨模态头是否盯住物体 | 潜空间往「图里真有的区域」推；解码时抬视觉 logit |
| **英语/文本主导、不看图** | 词汇投影、attn 质量在图 token vs 文 token | 向量算术 / 缩放跨模态头 |
| **视觉 token 太贵** | 层功能（浅层传、中层融、深层语言） | 晚注入、中层猛剪、早退出 |
| **细粒度差（OCR、指代）** | 哪些层还在用高分辨率视觉 | 少剪，或把视觉特征灌进更深层 |
| **多图 / 视频丢失** | Retrieval / 视觉头、KV 里哪些视觉状态 | 只给关键头长 KV，其余短 cache |

---

## 4. 单点方法论文：一类问题的 Locate + Steer

下面每条都是 **「先诊断内部对象，再干预」**，但只服务一个问题，不能替代总览。Steer 类型对照 LLM 综述：幅度操纵、向量算术、定向优化。

### 4.1 物体幻觉（Object Hallucination）

#### VaLSe：视觉贡献图 → 潜空间 steering

- **论文**：*Seeing It or Not? Interpretable Vision-aware Latent Steering to Mitigate Object Hallucinations*（Chen / Zheng et al., 2025）
- **链接**：[arXiv:2505.17812](https://arxiv.org/abs/2505.17812) · [代码](https://github.com/Ziwei-Zheng/VaLSe)
- **Locate**：对视觉相关输出 token 做 **visual contribution map**，看预测被哪些图像区域拉动；对比非语义 special token，去掉伪激活。
- **Steer**：用贡献图构造正/负样本（挡住无关区域 vs 原图），在潜空间做方向差，把表征推向「图里真正相关的物体」。
- **Improve**：减物体幻觉；同时当可解释工具。作者也指出现有 OH 指标偏粗。

#### VISTA：视觉信息丢失 → 激活 steering + 早层 logit

- **论文**：*The Hidden Life of Tokens*（Li et al., ICML 2025）
- **链接**：[arXiv:2502.03628](https://arxiv.org/abs/2502.03628) · [代码](https://github.com/LzVv123456/VISTA)
- **Locate**（生成过程中的三类现象）：
  1. **gradual visual information loss**：越往后生成，视觉 grounded token 越不受宠
  2. **early excitation**：语义 token 在早于最后一层达到峰值
  3. **hidden genuine information**：正确视觉 token 即使没被采中，logit 排名仍不低
- **Steer**（训练免费、推理期）：
  - **VSV（Visual Steering Vector）**：在激活空间强化视觉线索，对抗视觉丢失
  - **SLA（Self-Logits Augmentation）**：用早层激活抬语义 token
- **Improve**：开放生成幻觉约降 40%；多种架构、多种解码都可用。

#### VEGAS：视觉编码器注意力 → 中层注入 + 自适应 steering

- **论文**：*VEGAS: Vision-Encoder Attention Guided Adaptive Steering*（Wang et al., CVPR 2026 Findings）
- **链接**：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_VEGAS_Mitigating_Hallucinations_in_Large_Vision-Language_Models_via_Vision-Encoder_Attention_CVPRF_2026_paper.html)
- **Locate**：幻觉时常对应 **LLM 最终视觉注意力不集中**；视觉编码器自己的注意力更盯物体。视觉–文本冲突在 **LLM 中层**最强。
- **Steer**：把视觉编码器 attention map 注入 LLM 中层；对「没盯住关键物体」的 token，按注意力熵（VABE）自适应改 logit。
- **Improve**：推理期减幻觉，强调「该用哪张注意力图」。

#### SteerVLM：配对行为激活 → 轻量 steering 模块

- **论文**：*SteerVLM: Robust Model Control through Lightweight Activation Steering for Vision Language Models*（Sivakumar et al., EMNLP 2025 Findings）
- **链接**：[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1285/)
- **Locate**：从「目标行为 / 相反行为」配对 prompt 的潜嵌入里学该调哪些激活（连接语言模态与图像上下文）。
- **Steer**：极小模块（约原模型 **0.14%** 参数）做维度调制与跨层自适应 steering，不改原权重、不用手抽静态向量。
- **Improve**：指令遵循 + 幻觉缓解；附带 VNIA 数据集。更偏「学一个 steering 头」，而不是手工概念方向。

| 方法 | Locate 对象 | Steer 类型 | 是否训练 |
|------|-------------|------------|----------|
| VaLSe | 视觉贡献图 / 区域 | 潜空间向量算术 | 需构造正负样本 |
| VISTA | 层间视觉 logit 演化 | 激活向量 + 早层 logit | 训练免费 |
| VEGAS | 编码器 vs LLM 视觉注意力、中层冲突 | 中层注入注意力 + 自适应 logit | 训练免费 |
| SteerVLM | 配对行为激活 | 学出来的幅度/方向调制 | 训很小模块 |

### 4.2 视觉 token 效率（算力 / 延迟）

这类工作的 Locate 往往是 **层功能诊断**，Steer 是 **少算视觉 token**（幅度为 0 的极限：丢掉 token）。

#### HiDrop：浅层晚注入、中层凹形剪、深层早退出

- **论文 / 笔记**：[HiDrop 笔记](../HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.md) · [arXiv:2602.23699](https://arxiv.org/abs/2602.23699)
- **Locate**：浅层几乎不融合视觉（propagator）；中层稀疏融合、冗余极高；深层语言主导，可不再看视觉。
- **Steer / Improve**：Late Injection、Concave Pyramid Pruning、Early Exit；约压 90% 视觉 token，训练约 1.72×。

#### FastV / PyramidDrop / TwigVLM

- **FastV**（Chen et al., ECCV 2024）：早期某层一次性猛剪视觉 token。Locate 偏「第 2 层之后图像只值一半 token」。
- **PyramidDrop / PDrop**（Xing et al., 2024）：均匀金字塔逐步剪。
- **TwigVLM**（Shao et al., 2025）：早剪 + 深层清掉剩余视觉。

HiDrop 认为它们 **误解浅层、日程太死**，但仍是同一类「定位冗余视觉 → 推理/训练少算」。

#### Lei et al.：解释驱动的视觉 token 压缩

- **Locate**：梯度检测，把注意力模式对到解释结果上，标出不重要视觉 token。
- **Steer**：输入阶段剪掉这些视觉 token（幅度操纵的删除版）。
- 综述 *Locate, Steer, and Improve* §5 效率部分有引用。

### 4.3 跨模态主导 / 不看图

- **Hinck et al. (2024)**：分析 VLM 英语主导回复。Locate 内部注意力与隐状态；Steer 对注意力/隐状态做向量算术，压英语偏置。
- **Wendler 式词汇投影**（LLM 综述引用，VLM 可迁移）：中层偏英语概念空间。VLM 上还可看「图 token 投到词表像不像物体名」。
- **Phukan et al. (2025)**：*Beyond Logit Lens*，用上下文嵌入做 VLM 幻觉检测与 grounding，属词汇投影/表征对齐类 Locate。

### 4.4 细粒度视觉进深层（偏 VLA，但对象仍是 VLM 骨干）

- **DeepVision-VLA**（[arXiv:2603.15618](https://arxiv.org/abs/2603.15618)，本仓库 PDF：`Paper_Reading/DeepVision_VLA_Look_Before_Acting_2603.15618.pdf`）
- **Locate**：VLA 深层对视觉 token 越来越不敏感。
- **Steer**：VL-MoT 把视觉基础模型多层特征注入 VLA 深层；AGVP 用浅层注意力剪无关视觉 token。
- **Improve**：仿真 / 真机操作成功率相对基线约 +9.0% / +7.5%。
- 问题是 **动作精度**，手段仍是「视觉表征该在哪一层、留哪些 token」。

### 4.5 单点论文怎么用 LSI 骨架读

```
读一篇 VLM 干预论文时问三句：

1. Locate：诊断的是哪类对象？
   视觉 token / 跨模态头 / 编码器注意力 / 残差方向 / 层号
2. Steer：干预属于哪一类？
   幅度（剪、缩放、注入 attn） / 向量算术 / 定向优化（只训一小块）
3. Improve：对应哪项目标？
   对齐（少幻觉、肯看图） / 能力（OCR、指代） / 效率（少 token）
```

同一问题可以有多条 Locate+Steer 路径（幻觉：贡献图、logit 演化、编码器注意力、学出来的 steering 模块），目前没有综述把它们按因果强度、代价、副作用对齐成 LLM 第 5 章那样的表。

---

## 5. 和 LLM 综述的衔接

| LLM LSI | 接到 VLM 时 |
|---------|-------------|
| 可解释对象 §2 | 加上 \(\mathbf{E}_v\)、projector、跨模态 attn、ViT 层 |
| Locate §3 | 探针/因果/梯度/词汇投影仍可用；对象换成视觉位置和模态 |
| Steer §4 | 幅度操纵常表现为 **剪视觉 token / 缩放跨模态头**；向量算术多在残差上推「看图」方向 |
| Improve §5 | 对齐侧重幻觉与视觉 grounding；效率侧重视觉 token 与 KV；能力侧重 OCR、计数、空间关系 |

ICLR Blogpost 的五种技术，大致落在 Locate，较少走到「永久定向优化」：

| Blogpost 方法 | LSI 位置 |
|---------------|----------|
| Probing | Locate §3.4 |
| Activation patching | Locate §3.2 / Steer 幅度（patch） |
| Logit lens | Locate §3.5 |
| SAE | 对象 §2.5，再接到幅度/向量算术 |
| Automated explanation | 给特征/头贴标签，辅助 Locate |

---

## 6. 建议阅读顺序

1. LLM 骨架：[Locate_Steer_Improve_Actionable_MI_Survey.md](Locate_Steer_Improve_Actionable_MI_Survey.md) + 原论文 [2601.14004](https://arxiv.org/abs/2601.14004)
2. VLM 方法盘点：[MI Meets VLMs](https://iclr-blogposts.github.io/2025/blog/vlm-understanding/)
3. 幻觉单点：VISTA → VaLSe → VEGAS → SteerVLM
4. 效率单点：HiDrop 笔记，并对照 FastV / PDrop
5. 若做机器人：DeepVision-VLA + [VLA 对照草稿](../VLA_Common_Issues_Locate_Steer_Improve.md)

---

## 7. 本文未覆盖、需要另找专项综述的问题

- 训练数据偏差、合成 caption 噪声（偏数据，不是内部对象）
- 安全 / jailbreak 在 VLM 上的图像攻击（多走对抗样本，MI 定位仍少）
- 视频 VLM 的时序幻觉与长上下文 KV
- MoE-VLM 的路由对象

这些都不构成「VLM 版 LSI 总览」的替代品。
