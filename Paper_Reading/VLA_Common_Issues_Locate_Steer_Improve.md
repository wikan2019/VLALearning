# VLA 常见问题 × Locate / Steer / Improve 对照草稿

> **文档用途**  
> 把 LLM 综述 *Locate, Steer, and Improve* 的流水线，接到 **Vision-Language-Action (VLA)** 的常见失败上。目前**没有**一篇现成的 VLA 版 LSI 综述；本文用 2025–2026 已发表的机制论文 + 本仓库既有笔记，先搭一张对照表，便于诊断「差在哪、改哪、怎么改」。  
> **性质**：对照草稿，不是系统综述。Steer 证据在部分问题上仍薄。  
> **关键词**: VLA、机制可解释性、Locate / Steer / Improve、SAE、激活注入、指令接地、失败模式  
> **相关笔记**: [Locate, Steer, and Improve（LLM）](locate_steer_improve/README.md) · [π0 系列](../VLA_Advances/22_Pi0_Generalist_Policy.md) · [HiDrop（视觉 token）](HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.md) · [DeepVision-VLA PDF](DeepVision_VLA_Look_Before_Acting_2603.15618.pdf)

---

## 目录

1. [一句话定位](#1-一句话定位)
2. [LLM 框架迁到 VLA 时要换什么](#2-llm-框架迁到-vla-时要换什么)
3. [总对照表](#3-总对照表)
4. [问题分述：Locate → Steer → Improve](#4-问题分述locate--steer--improve)
5. [工具怎么接到 VLA 对象上](#5-工具怎么接到-vla-对象上)
6. [建议排查顺序](#6-建议排查顺序)
7. [草稿缺口](#7-草稿缺口)
8. [文献与链接](#8-文献与链接)

---

## 1. 一句话定位

VLA 把 **视觉 + 语言 → 动作** 塞进同一套（或杂交的）Transformer。失败往往不是「模型不够大」，而是内部在用错误通路：只看图不听指令、把演示轨迹背下来、动作头把多模态平均掉、闭环里漂/僵/幻影抓取。

**Locate / Steer / Improve** 在这里仍然适用：

```
Locate   找出承载该失败的内部对象（视觉残差、语言通路、专家通路、SAE 特征、动作头）
Steer    推理时改激活 / 方向 / 特征幅度，验证因果
Improve  换数据与训练隔离、改动作生成范式、或只更新被定位的子集
```

LLM 综述给出方法分类；VLA 侧目前是**若干实证研究**，按失败类型拼起来。

---

## 2. LLM 框架迁到 VLA 时要换什么

LSI 原文的可解释对象是 decoder-only LLM 的残差、头、FFN、SAE。VLA 至少要加四类：

| VLA 特有对象 | 为什么必须单列 |
|--------------|----------------|
| **视觉 token / ViT 残差** | 分辨率高、数量远大于文本；常主导动作（见 §4.1） |
| **语言 token / 指令通路** | 场景无歧义时可能被忽略；有歧义时才因果必要 |
| **动作头**（离散 Binning / Flow Matching / FAST） | 连续控制的精度与多模态，不完全在 VLM 残差里 |
| **专家 / 策略通路**（π0.5、SmolVLA、GR00T 等） | 与 VLM 通路可分：专家偏运动程序，VLM 偏目标语义 |

闭环还多了 **状态–动作互信息、动作熵** 这类轨迹级诊断（Tri-Info），它们不是单步 token 的 MI，但可当作 Locate 的粗筛。

LSI 的三种 Steer 仍然对应得上：

| LSI Steer | VLA 上常见落地 |
|-----------|----------------|
| Amplitude Manipulation | SAE 特征缩放、头/层激活注入或消融 |
| Vector Arithmetic | 残差上加概念方向；探针引导的线性最优控制干预 |
| Targeted Optimization | 只训动作专家、Knowledge Insulation、稀疏微调语言神经元 |

---

## 3. 总对照表

| # | 常见问题 | Locate（看什么） | Steer（怎么验证/临时改） | Improve（怎么治本） | 主要依据 |
|---|---------|------------------|---------------------------|---------------------|----------|
| **P1** | 不听指令，只跟视觉走 | 空/错指令 + 激活注入；语言扰动 + patching | 注入语言通路 / 错指令激活，看轨迹是否跟着任务变 | 歧义场景增数据；隔离 VLM 语义 vs 专家运动 | [2603.19233](https://arxiv.org/abs/2603.19233), [2610.10178](https://arxiv.org/html/2610.10178v1) |
| **P2** | SFT 后死记轨迹，换物/换场景崩 | SAE：记忆特征 vs 通用运动原语 | 放大通用 grasp 特征、压 episode 记忆特征 | 更大更多样数据（DROID）；Knowledge Insulation；少在小 LIBERO 上猛 SFT | [2603.19183](https://arxiv.org/abs/2603.19183), LIBERO-PRO |
| **P3** | 视觉表征在深层变弱，精细操作差 | 层间对视觉 token 的敏感度；探针读物体位置 | DeepVision：多层视觉注入；AGVP 按浅层注意力剪无关 token | 深层补视觉、浅层剪冗余 | [2603.15618](https://arxiv.org/abs/2603.15618) |
| **P4** | 视觉 token 太多，训/推太慢 | 浅层传递、中层融合、深层语言主导（MLLM 探针） | 晚注入 / 凹形剪枝 / 早退出（HiDrop） | 按层预算压缩；VLA 需另测对动作精度的伤害 | [2602.23699](https://arxiv.org/abs/2602.23699) |
| **P5** | 加速 vs 减速、左绕 vs 右绕被平均掉 | 看动作头：MSE / 离散 Binning vs Flow / VQ | 推理换采样（Flow 落入单峰）；不是改一个神经元能彻底解决 | 换 π0 Flow Matching 或 π0-FAST 的 VQ 码本 | [π0 笔记](../VLA_Advances/22_Pi0_Generalist_Policy.md) |
| **P6** | 闭环漂、僵住、幻影抓取 | Tri-Info：动作熵↑=漂，熵↓=冻，状态–动作 MI↓=幻影抓 | 目前多为检测/停机，内部 steering 证据少 | 失败预测器 + 回退；根因仍要回到 P1/P2/P5 | [2606.19998](https://arxiv.org/html/2606.19998) |
| **P7** | 想控风格/偏好（更快、更轻、拒危险） | 线性探针：特征可观测性；SAE 运动原语 | 最小线性干预把表征推到目标区；缩放对应 SAE 特征 | 推理期对齐用户偏好，避免全量重训 | [2603.05487](https://arxiv.org/abs/2603.05487), [2603.19183](https://arxiv.org/abs/2603.19183) |
| **P8** | 新环境 / 开放世界掉点 | 知识冲突、环境绑定（π0.5 的隔离思想） | 高层语义 vs 低层运动解耦（架构层，不全是激活 steer） | Knowledge Insulation + 异构混合训练 + 双层决策 | [π0.5 节](../VLA_Advances/22_Pi0_Generalist_Policy.md) |

---

## 4. 问题分述：Locate → Steer → Improve

### P1 不听指令（视觉主导）

**现象**：换指令轨迹几乎不变；空 prompt 仍能「完成」看起来合理的动作。

**Locate**

- *Not All Features Are Created Equal*：六模型、39 万+ rollout。把 **baseline 激活注入空指令** 的 episode，行为几乎复原 → **视觉通路足够驱动动作**。跨任务注入会把机器人拽向**源任务的空间轨迹**（X-VLA 上 99.8% 对齐源轨迹）→ 学的是绑在场景坐标上的运动程序，不是抽象任务。
- 语言是否被用，取决于**任务结构**：场景唯一指定目标时语言可被忽略；同一场景多目标时语言才关键（X-VLA `libero_goal` 错指令 94%→10%；`libero_object` 则 60–100% 都走视觉）。
- 多通路模型（π0.5 / SmolVLA / GR00T）：**专家通路**编码运动程序，**VLM 通路**编码目标语义；专家注入造成的行为位移大约是 VLM 的 2 倍。
- *Language Grounding*（π0.5、GR00T N1.7）：同义改写、不存在物体不敏感；**空指令和方向词**冲击大。因果位点：GR00T 偏早期周期性 cross-attn；π0.5 更分散。

**Steer**

- 激活注入 / activation patching：把语言通路或错指令激活贴进当前前向，看末端是否改目标。
- Attribution patching 在 GR00T 上接近真 patch，在 π0.5 上不可靠 → 验证时优先 **activation patching**。

**Improve**

- 评测与训练要包含「同场景多目标」；只测视觉可解任务会高估语言接地。
- 架构上让语义（VLM）与运动（专家）子空间可分，微调时保护 VLM（见 P2 的 insulation）。

---

### P2 小数据 SFT 记忆演示

**现象**：LIBERO 原协议 >90%，LIBERO-PRO 扰动后接近 0。

**Locate**

- 在残差上训 SAE（TopK + AuxK）。**大多数特征是某条训练演示的记忆**；少数是跨场景的 grasp / carry 等运动原语。
- 用激活统计（不必 rollout）把特征标成 general vs memorized。

**Steer**

- 缩放单个 SAE 特征可因果改变 LIBERO 行为；**通用特征**跨任务跨场景可复用，记忆特征会把策略锁回某条轨迹。

**Improve**

- 小数据 SFT 会不成比例放大记忆特征。
- 更大更多样数据（DROID）或 **Knowledge Insulation**（停梯度、语义与专家分离）更利于通用特征。
- 不要把 benchmark 高分当成泛化；用扰动协议压测。

---

### P3 深层「看不见」精细视觉

**现象**：折叠、插线等毫米级任务差；语言推理层对视觉 token 越来越不敏感。

**Locate**

- DeepVision-VLA：多种动作范式下，**越深的层对视觉 token 越不敏感**。
- 浅层注意力仍能标出任务相关视觉 token（AGVP 的依据）。

**Steer / Improve**

- **VL-MoT**：视觉专家多层特征注入 VLA 骨干深层（晚融合、补视觉）。
- **AGVP**：用浅层注意力剪无关视觉 token，少算力、保住操作线索。
- 仿真 / 真机相对先前 SOTA +9.0% / +7.5%（论文数字，迁移时需复现）。

---

### P4 视觉 token 二次方太贵

**现象**：高分辨率、多相机时训推墙。

**Locate**（借 MLLM HiDrop，VLA 需复测）

- 浅层：视觉几乎只传递；中层：稀疏融合、冗余最高；深层：语言主导，可早退视觉。

**Steer / Improve**

- Late Injection + 凹形剪枝 + Early Exit；ILVAS 选剪层，可微 Top-K 选 token。
- **对 VLA 的警告**：剪 90% 对 VQA 可能无事，对接触力学可能致命。先在抓取 / 插入上做因果消融，再上压缩。

---

### P5 多模态动作被平均

**现象**：该加速却刹、该左绕却直行。

**Locate**

- 看动作头损失：MSE / 逐维 Binning → 最优解是条件期望（mode averaging）。
- Flow Matching / 扩散：从噪声落入单峰；FAST+VQ：码本里没有「两条路的平均」。

**Steer**

- 换采样种子或温度，看是否在加速/减速之间切换。若始终输出中间值 → 生成范式问题，不是某个 SAE 特征能单独修好。

**Improve**

- π0：Flow Matching 动作头；π0-FAST：频域 VQ，精度来自编码对象而不是更细的均匀 Bin。

---

### P6 闭环失败：漂 / 冻 / 幻影抓

**现象**：中途乱走、卡住、空抓却当已抓住。

**Locate**

- Tri-Info 把控制看成闭环信息管道：  
  - **Drift**：动作熵骤升  
  - **Freeze**：熵塌缩  
  - **Phantom grasp**：状态–动作互信息下降  
- 跨架构、跨环境、sim2real 可迁移检测（文称真机约 83%，需当结果而非定论）。

**Steer**

- 公开工作偏 **检测与停机**，缺少「对哪条内部方向做 steering 就能从冻里拉出来」的回路级证据。
- 应回绑 P1（没在听目标）、P2（在背轨迹）、P5（动作分布塌到错误模式）。

**Improve**

- 运行时：监测器 + 回退 / 重规划。  
- 训练时：针对该失败模式补对比数据，而不是只加通用 BC。

---

### P7 推理期控偏好，不想重训

**现象**：要更保守、更快、避开某类接触。

**Locate**

- *Observing and Controlling Features*：线性探针 → **feature-observability**；π0.5 与 OpenVLA 的表征空间里，许多任务相关量可线性读出。
- SAE 运动原语（grasp、抬升等）提供无标签概念轴。

**Steer**

- 用最优控制式的**最小线性干预**把内部状态推到目标区，闭环能力可保留。
- 缩放通用 SAE 特征，语义与行为一致时可跨场景复用。

**Improve**

- 适合在线对齐用户偏好；不能替代缺数据或缺正确动作范式。副作用要测（LLM 里推「快乐」会伤隐私意识，VLA 类比：放大「快」可能更撞）。

---

### P8 开放世界 / 换厨房就崩

**现象**：训练厨房可以，新布局、新台面不行。

**Locate**

- π0.5 诊断：环境表象与操作本质绑在一起（虚假相关）。
- 表征级：不变子空间 vs 环境子空间；梯度级：共享参数 vs 环境 Adapter。

**Steer / Improve**（偏架构与训练，不全是推理激活）

- Knowledge Insulation + 异构数据混合 + 高层语义规划 / 低层运动解耦。
- 部署时丢掉 \(\theta_{\text{env}}\)，只靠 \(\theta_{\text{shared}}\)。

与 P2 一致：小数据 SFT 会把环境 ID 写进运动特征；insulation 是预防，SAE steering 是补丁。

---

## 5. 工具怎么接到 VLA 对象上

| LSI 工具 | VLA 上优先打在哪 | 注意 |
|----------|------------------|------|
| Magnitude Analysis | 视觉 vs 语言 token 范数；SAE \(a_j\)；动作熵 | 高激活 ≠ 因果 |
| Causal Attribution / Patching | 指令 token、cross-attn、专家 vs VLM 通路、末段动作 token | 空指令 / 方向词是好的 corr 输入 |
| Probing | 物体位姿、是否抓住、任务 ID、指令是否被编码 | 可解码 ≠ 策略在用 |
| Vocabulary Projection | VLM 骨干词表；动作头若是离散 token 也可投动作词 | Flow 头没有词表，投不全 |
| SAE | 残差；尽量 **per-token**（多数架构 mean-pool 会伤动作保真，X-VLA 例外） | 记忆特征很多，先做 generality 过滤 |
| Circuit Discovery | 视觉 token → 语言 → 动作头 的边 | VLA 上公开回路还很少 |
| Vector Arithmetic / 线性干预 | 残差、专家子空间 | 保持闭环稳定性 |
| Targeted Optimization | 只训专家 / 语言特异神经元；冻 VLM | 防 P2 记忆膨胀 |

---

## 6. 建议排查顺序

场景 S 表现差时，不必先上 SAE：

```
1. 行为对照
   空指令 / 错指令 / 换物体 / 换布局 → 像 P1 还是 P2 还是 P8
2. 闭环粗筛（可选 Tri-Info）
   熵↑漂 / 熵↓冻 / MI↓幻影抓 → P6，再回绑根因
3. 动作头检查
   是否总输出「中间动作」→ P5
4. 通路注入
   视觉激活 vs 语言激活 vs 专家激活 谁能复现失败轨迹 → P1
5. SAE（需要概念级旋钮时）
   general vs memorized；再缩放验证 → P2 / P7
6. 治本
   数据与 insulation（P2/P8）或换动作范式（P5）或深层补视觉（P3）
```

---

## 7. 草稿缺口

以下问题**还没有**够格写进「标准 LSI 处方」的文献密度：

- 接触力 / 触觉失败的内部对象  
- 双臂协调回路  
- 动作头（Flow）内部的 circuit，而不只是 VLM 残差  
- P6 三类失败对应的可 steer 方向  
- 高分辨率多相机下 HiDrop 式剪枝对操作精度的因果影响  
- 统一 benchmark：同一失败、同一对象、比较幅度 vs 向量 vs 定向优化（LLM LSI 的 Table 4 在 VLA 上尚未建立）

因此本文是 **对照草稿**：表中 Improve 列有的是机制实验，有的是架构论文的合理外推，阅读时请回链原论文。

---

## 8. 文献与链接

### 框架与本仓库

| 资源 | 链接 |
|------|------|
| Locate, Steer, and Improve（LLM 综述） | [arXiv:2601.14004](https://arxiv.org/abs/2601.14004) · [笔记](locate_steer_improve/Locate_Steer_Improve_Actionable_MI_Survey.md) |
| π0 / FAST / π0.5 | [22_Pi0_Generalist_Policy.md](../VLA_Advances/22_Pi0_Generalist_Policy.md) |
| HiDrop | [arXiv:2602.23699](https://arxiv.org/abs/2602.23699) · [笔记](HiDrop_Hierarchical_Vision_Token_Reduction_2602.23699.md) |
| DeepVision-VLA | [arXiv:2603.15618](https://arxiv.org/abs/2603.15618) · [PDF](DeepVision_VLA_Look_Before_Acting_2603.15618.pdf) |

### VLA 机制与失败（对照表直接引用）

| 论文 | arXiv | 在本文中的角色 |
|------|-------|----------------|
| Observing and Controlling Features in VLA Models | [2603.05487](https://arxiv.org/abs/2603.05487) | P7：可观测 + 线性可控 |
| Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA | [2603.19183](https://arxiv.org/abs/2603.19183) · [drvla.github.io](https://drvla.github.io) | P2：记忆 vs 原语；特征 steering |
| Not All Features Are Created Equal | [2603.19233](https://arxiv.org/abs/2603.19233) · [Action Atlas](https://action-atlas.com) | P1：视觉主导、专家 vs VLM |
| Do VLAs Understand Instructions? | [2610.10178](https://arxiv.org/html/2610.10178v1) | P1：指令扰动 + patching 位点 |
| Tri-Info | [2606.19998](https://arxiv.org/html/2606.19998) | P6：漂 / 冻 / 幻影抓 |
| Pure VLA Models: A Comprehensive Survey | [2509.19012](https://arxiv.org/abs/2509.19012) | 架构谱系，不含 LSI |

读法建议：先 LLM LSI 懂工具 → 再 P1/P2 两篇机制文 → 再按失败类型查表。
