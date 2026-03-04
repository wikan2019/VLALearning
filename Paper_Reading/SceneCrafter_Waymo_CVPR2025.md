# SceneCrafter: Controllable Multi-View Driving Scene Editing

> **论文信息**
> - **标题**: SceneCrafter: Controllable Multi-View Driving Scene Editing
> - **作者**: Zehao Zhu, Yuliang Zou, Chiyu Max Jiang, Bo Sun, Vincent Casser, Xiukun Huang, Jiahao Wang, Zhenpei Yang, Ruiqi Gao, Leonidas Guibas, Mingxing Tan, Dragomir Anguelov
> - **机构**: Waymo, UT Austin, Johns Hopkins University, Google DeepMind
> - **发表**: CVPR 2025
> - **arXiv**: [2506.19488](https://arxiv.org/abs/2506.19488)

---

## 目录

1. [论文概述](#1-论文概述)
2. [研究动机与问题定义](#2-研究动机与问题定义)
3. [方法详解](#3-方法详解)
4. [实验结果](#4-实验结果)
5. [消融实验](#5-消融实验)
6. [下游任务评估](#6-下游任务评估)
7. [关键创新点总结](#7-关键创新点总结)
8. [个人思考与评价](#8-个人思考与评价)

---

## 1. 论文概述

SceneCrafter 是 Waymo 提出的一个**多视角驾驶场景编辑模型**，用于自动驾驶的**传感器仿真 (Sensor Simulation)**。它能够对真实驾驶日志中的多摄像头图像进行**3D 一致性的可控编辑**，支持以下编辑操作：

| 编辑类型 | 具体操作 | 条件控制 |
|---------|---------|---------|
| **全局编辑 (Global)** | 修改天气（晴/雨/雪/雾） | CLIP 文本编码 |
| **全局编辑 (Global)** | 修改时间（白天→黄昏→夜晚） | 太阳角度 + 位置编码 |
| **局部编辑 (Local)** | 插入/删除车辆等目标 | 3D 包围盒 + 类型编码 |
| **布局编辑 (Layout)** | 修改道路结构 | HD Map 条件 |

### 核心思想

```
真实驾驶场景 → SceneCrafter 编辑 → 逼真的仿真场景
                    ↓
         保持 3D 几何一致性
         精确控制多维条件
         覆盖多个摄像头视角
```

与纯生成式方法不同，SceneCrafter 基于**真实驾驶日志的编辑**，因此生成的场景更加**可信**、**接地** (grounded in reality)，更适合用于自动驾驶系统的全栈仿真评估。

---

## 2. 研究动机与问题定义

### 2.1 为什么需要场景编辑？

自动驾驶仿真主要有三种范式：

```
┌─────────────────────────────────────────────────────────────────┐
│                    自动驾驶仿真方法对比                           │
├──────────────────┬──────────────────┬───────────────────────────┤
│   行为仿真        │   纯生成式仿真     │   编辑式仿真 ✓            │
│ (Behavior Sim)   │ (Generation Sim)  │ (Editing Sim)            │
├──────────────────┼──────────────────┼───────────────────────────┤
│ 绕过感知系统       │ 不受真实场景约束    │ 基于真实场景进行修改        │
│ 无法评估端到端模型  │ 生成结果难以取信    │ 保留真实场景的可信度        │
│ 无法模拟天气变化   │ 3D 一致性差       │ 天生具备场景真实性          │
│ 代表：Waymax      │ 代表：MagicDrive  │ 代表：SceneCrafter ✓     │
└──────────────────┴──────────────────┴───────────────────────────┘
```

### 2.2 驾驶场景编辑的三大挑战

| # | 挑战 | 描述 | SceneCrafter 的解决方案 |
|---|------|------|----------------------|
| 1 | **跨摄像头 3D 一致性** | 8 个摄像头视角之间的编辑结果必须几何一致 | View-Spatial Joint Attention + Raymap 条件 |
| 2 | **空街道先验学习** | 训练数据几乎都是"有车的街道"，模型难以学习"没车的街道"长什么样 | Masked Training 策略 |
| 3 | **配对训练数据获取** | 编辑前后的图像对(paired data)难以从真实世界获取 | Teacher-Student 合成数据框架 |

---

## 3. 方法详解

### 3.1 整体架构：Teacher-Student 框架

SceneCrafter 采用**两阶段**训练流程：

```
┌───────────────────────────────────────────────────────────────┐
│                     Stage 1: 训练 Teacher 模型                 │
│                                                               │
│  ┌──────────────┐                    ┌──────────────────────┐ │
│  │  Teacher-G   │ ── 生成全局编辑     │  合成配对数据          │ │
│  │ (全局编辑教师) │    训练数据 ──────→│  (weather/time 变化)  │ │
│  └──────────────┘                    └──────────────────────┘ │
│                                                               │
│  ┌──────────────┐                    ┌──────────────────────┐ │
│  │  Teacher-L   │ ── 生成局部编辑     │  合成配对数据          │ │
│  │ (局部编辑教师) │    训练数据 ──────→│  (agent 增删)         │ │
│  └──────────────┘                    └──────────────────────┘ │
│                                                               │
│                        ↓ 共 1M 配对数据                         │
│                                                               │
│              Stage 2: 训练统一 Student 编辑模型                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                  SceneCrafter Student                    │  │
│  │  source images + 任意编辑条件 → 编辑后的目标图像             │  │
│  │  支持全局编辑 + 局部编辑 + 混合编辑                         │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 基础模型：Multi-View Diffusion Model

SceneCrafter 构建在 CAT3D 多视角扩散模型之上，包含两个关键设计：

#### (a) View-Spatial Joint Attention

将 LDM 中的 2D attention block 扩展为 3D attention block（2D 空间 + 1D 跨视角），使模型能在多个摄像头视角之间共享注意力：

```
标准 LDM: Attention(Q_i, K_i, V_i)  ← 每个视角独立计算
    ↓
SceneCrafter: Attention(Q_i, K_{0:N}, V_{0:N})  ← 跨所有 N 个视角联合计算
```

**关键优势**: 复用预训练 LDM 参数，2D → 3D inflate 不引入额外参数。

#### (b) Raymap 相机位姿条件

使用 **Raymap**（射线图）编码每个像素位置的射线原点和方向。所有相机位姿相对于第一个相机做归一化，确保全局刚性变换不变性：

```
输入 = concat(noisy_latents, raymaps)
     → U-Net 联合处理位姿和图像信息
```

### 3.3 Teacher 模型的条件控制机制

Teacher 模型支持 **4 类条件**，全部通过 cross-attention 注入 U-Net：

```
                    ┌── 天气 c_w: CLIP 文本编码("sunny"/"rainy"/"foggy"/"snowy")
                    │
                    ├── 时间 c_t: 太阳角度(根据当地时间+地理位置计算) + 位置编码
全局条件 c_g ───┤
                    ├── HD Map c_r: 车道线段(起/止位置+类型)
局部条件 c_l ───┤                → PerceiverIO 压缩至 512 tokens → MLP
                    │
                    └── Agent Boxes c_b: 最多 256 个目标
                                        [x, y, z, l, w, h, yaw, type]
                                        → one-hot(type) + concat → MLP
                    
前景掩码 m ───── 3D 包围盒投影到各视角生成二值掩码，拼接到 latent 通道维度

Raymap p ──── 射线方向+原点，拼接到 latent 输入
```

**训练细节**：对每类条件施加 **10% dropout**，使模型在推理时即使缺少某些条件也能稳健工作。

### 3.4 合成配对数据生成

这是 SceneCrafter 的**核心创新之一**——如何生成高质量的编辑前后配对数据。

#### 3.4.1 全局编辑数据：改进的 Prompt-to-Prompt

原始 Prompt-to-Prompt (P2P) 通过冻结 cross-attention 权重来保持编辑前后的几何一致性。但在多视角驾驶场景中，直接应用 P2P 效果不佳。SceneCrafter 做了三个关键改进：

**改进 1: 替换 Self-Attention 权重（而非 Cross-Attention）**

```
原始 P2P:  冻结 image-to-text cross-attention 权重
    ↓ 效果不佳（驾驶场景用的是多模态条件，不是纯文本）
    
SceneCrafter: 冻结 image-to-image self-attention 权重
    ↓ 效果显著（在像素级别保持了几何一致性）
```

**直觉解释**：Self-attention 权重编码了图像的空间结构信息。冻结 self-attention 意味着"保持图像的空间布局不变"，而让 cross-attention 自由变化来响应不同的编辑条件（如从晴天到雨天）。

**改进 2: 增加更多的场景控制信号**

加入 agent boxes 和 HD map 条件后，模型能更好地保持场景中的细粒度几何结构（如车辆位置、道路布局），从而生成更一致的配对数据。

**改进 3: 仅使用白天图像作为 source**

实验发现，用白天图像作为 source 生成效果最佳。target 时间则从全天随机采样，source 和 target 随机翻转顺序。

#### 3.4.2 局部编辑数据：Masked Training + Multi-View Repaint + Alpha Blending

**Step 1: Masked Training（学习空街道先验）**

```python
# 核心思想：对前景和背景施加不同的噪声
z_t = (1 - m) ⊙ (α_t·z_0 + σ_t·ε)  +  m ⊙ z_0
#      └── 背景：正常加噪 ──────────┘  └── 前景：保持清晰（零噪声）

# 训练损失仅计算在背景像素上
L = E[||(1 - m) ⊙ (ε - ε_θ(z_t, t, c_g, c_l, m))||²]
```

**直觉解释**：通过在训练时只对背景加噪/去噪，模型学会了"背景应该长什么样"——即 **空街道先验**。即使所有训练数据都是有车的街道，模型也能推断出去掉车后街道的样子。

**Step 2: Multi-View Repaint（3D 一致性修复）**

在推理时，使用类似 RePaint 的迭代修复策略，但**同时处理所有视角**以确保 3D 一致性：

```
z_{t-1}^{background} ~ N(ᾱ_t·z_0, (1 - ᾱ_t)·I)    ← 已知背景，直接采样
z_{t-1}^{foreground} ~ N(μ_θ(...), Σ_θ(...))         ← 多视角扩散模型联合去噪

z_{t-1} = m ⊙ z_{t-1}^{foreground} + (1-m) ⊙ z_{t-1}^{background}
```

**Step 3: Alpha Blending（生成任意交通密度）**

获得"空街道" $\mathcal{I}_{empty}$ 和"满街道" $\mathcal{I}_{full}$ 后，通过 alpha blending 可以生成具有任意数量车辆的场景：

$$\mathcal{I}_{sampled} = \mathbf{m}_{sampled} \odot \mathcal{I}_{empty} + (1 - \mathbf{m}_{sampled}) \odot \mathcal{I}_{full}$$

其中 $\mathbf{m}_{sampled}$ 是从所有车辆掩码中随机抽样组合而成的复合掩码。

```
空街道 ← Masked Training + MV Repaint
   +                                    → 配对训练数据
满街道 ← 原始驾驶日志                        (任意数量车辆的场景对)
   ↓
Alpha Blending (随机选择要保留的车辆)
```

### 3.5 Student 编辑模型

Student 模型在 Teacher-G 的权重基础上初始化，增加了 **source image 条件分支**：

```
输入 = concat(z_t, z_0^{source})   ← 在 channel 维度拼接
     → U-Net (保留 Teacher-G 的所有条件机制)
     → 编辑后的目标图像
```

**关键设计选择**：
- Source image 通过 **concatenation**（拼接）而非 cross-attention 注入 → FID 提升 13.1
- 使用 **bounding box 条件**而非 mask 条件控制 agent 编辑 → 对小目标更精确

---

## 4. 实验结果

### 4.1 数据规模与训练配置

| 配置项 | 数值 |
|-------|------|
| 训练数据 | **13,867,496** 个驾驶视频片段 |
| 每段结构 | 17 帧 × 8 摄像头，10Hz |
| 标注信息 | 相机位姿、天气、时间、HD Map、3D 包围盒 |
| 合成配对数据 | **1M** 对 |
| 训练硬件 | 128× Google TPU v5 |
| 训练步数 | Teacher: 100k, Student: 200k |
| 学习率 | 1e-5 |
| Batch Size | 128 |
| 推理速度 | ~10 秒 / 8 张 512×512（单 A100） |

### 4.2 评估指标

SceneCrafter 提出了一套完整的评估框架：

| 指标 | 衡量维度 | 说明 |
|------|---------|------|
| **FID** ↓ | 真实感 (Realism) | 编辑图像与原始图像的分布距离 |
| **CLIP Score** ↑ | 可控性 (Controllability) | 编辑结果与编辑指令的语义对齐度 |
| **User Study** ↑ | 编辑质量 (Editing Quality) | 11 位人类评估者对 20 组图像的偏好率 |
| **3D LPIPS** ↓ | 3D 一致性 (**新指标**) | 相邻视角重叠区域的感知相似度 |

#### 3D LPIPS 指标（新提出）

利用相机间的重叠视野，将 $C_i$ 投影到 $C_{i+1}$（以及反过来），比较重叠区域的 LPIPS：

```
C_front_left ←──重叠──→ C_front
               ↓ 投影
         提取重叠区域 patch 对
               ↓
         计算 LPIPS 距离
```

8 个摄像头产生 16 对比较（每对邻居双向投影），综合得到 3D 一致性评分。

### 4.3 定量结果对比

#### 全局编辑 (Global Editing)

| 方法 | Time FID ↓ | Time CLIP ↑ | Time 偏好率 | Weather FID ↓ | Weather CLIP ↑ | Weather 偏好率 |
|------|-----------|-------------|------------|--------------|----------------|---------------|
| SDEdit | 60.4 | 0.204 | 2.7% | 78.3 | 0.203 | 1.8% |
| P2P* | 46.8 | 0.223 | 13.6% | 55.4 | 0.207 | 12.7% |
| **SceneCrafter** | **37.2** | **0.220** | **83.6%** | **38.9** | **0.221** | **85.5%** |

**关键观察**：
- SceneCrafter 在 FID 上超过 SDEdit **40.6**，超过 P2P* **16.5**
- 用户偏好率碾压式领先：**83.6%** vs 13.6% vs 2.7%
- 在 CLIP Score 上与最佳 baseline 持平或略优

#### 局部编辑 (Local Editing)

| 方法 | Insertion FID ↓ | Removal FID ↓ |
|------|----------------|---------------|
| 2D-RePaint | 30.6 | 31.9 |
| MV-RePaint | 26.0 | 28.5 |
| **SceneCrafter** | **23.5** | **21.7** |

**关键观察**：
- Box 条件比 mask 条件更精确，特别是对小目标
- 2D-RePaint 各视角独立处理，缺乏 3D 一致性
- MV-RePaint 没有 masked training 先验，车辆去除不干净

#### 生成任务 (Generation)

| 方法 | FID ↓ | 3D LPIPS ↓ |
|------|-------|-----------|
| Real (上界) | 11.5 | 0.186 |
| CAT3D | 121.3 | 0.249 |
| SceneCrafter (w/o cond.) | 68.5 | 0.254 |
| **SceneCrafter (full)** | **36.2** | **0.187** |

**关键发现**：SceneCrafter 的 **3D LPIPS 达到 0.187，几乎与真实数据的 0.186 持平**！这说明模型生成的多视角图像具有接近真实水平的 3D 一致性。

---

## 5. 消融实验

### 5.1 Prompt-to-Prompt 设计选择

| Self-Attention 替换 | 增加条件 | 仅用白天 Source | FID ↓ | CLIP ↑ |
|:------:|:------:|:------:|:-----:|:------:|
| ✗ | ✗ | ✗ | 57.1 | 0.204 |
| ✓ | ✗ | ✗ | 41.5 | 0.202 |
| ✓ | ✓ | ✗ | 39.9 | 0.214 |
| ✓ | ✓ | ✓ | **36.2** | **0.223** |

**逐步分析**：
1. 替换 self-attention 权重：FID 从 57.1 → 41.5（**-15.6**），显著提升几何一致性
2. 增加 agent boxes + HD map 条件：CLIP 从 0.202 → 0.214，提升可控性
3. 仅用白天 source：FID 再降 3.7，质量进一步提升

### 5.2 Source Image 条件注入方式

| 方式 | FID ↓ | CLIP ↑ |
|------|-------|--------|
| Cross-Attention | 50.3 | 0.203 |
| **Concatenation** | **37.2** | **0.220** |

Concatenation 在 FID 上**提升 13.1**，说明对于像素级特征（source images, masks, raymaps），直接拼接比 cross-attention 更有效。

---

## 6. 下游任务评估

SceneCrafter 的核心应用场景是作为仿真器为下游感知模型提供测试环境。论文验证了编辑后的图像在下游任务中的表现：

### 6.1 语义分割

在 SceneCrafter 生成的图像上运行 Panoptic-DeepLab 模型：
- 26 类的聚合类别分布 **KL 散度仅为 0.01133**
- 分割掩码清晰且语义对应
- 唯一的微小差异出现在植被、建筑和天空类别（因为模型对这些元素的条件控制最弱）

### 6.2 3D 目标检测

在前视摄像头上运行单目 3D 检测模型：
- 检测器在生成的图像上表现**与真实图像相当**
- **很少出现漏检 (false-negative)** 情况
- 说明 SceneCrafter 生成的车辆外观和几何属性足够逼真，不会误导感知模型

---

## 7. 关键创新点总结

### 7.1 技术创新

| # | 创新点 | 重要性 |
|---|-------|-------|
| 1 | **Self-Attention 替换 P2P** | 将 P2P 从单图文本编辑扩展到多视角多模态编辑 |
| 2 | **Masked Training** | 从"满街道"数据中自监督学习"空街道"先验 |
| 3 | **Multi-View Repaint** | 确保修复结果在多视角间 3D 一致 |
| 4 | **Alpha Blending** | 从空/满街道对生成任意交通密度的配对数据 |
| 5 | **3D LPIPS 指标** | 首个衡量多视角生成一致性的感知指标 |
| 6 | **Box vs Mask 条件** | 用包围盒替代掩码，对小目标编辑更精确 |

### 7.2 工程规模

```
数据量:     ~1400 万视频片段 × 17 帧 × 8 视角 ≈ 19 亿帧图像
合成数据:   100 万配对数据
训练资源:   128× TPU v5 × 100k-200k 步
推理速度:   10 秒 / 8 张 512×512 (A100)
           0.1 秒 (128× A100 集群)
```

---

## 8. 个人思考与评价

### 8.1 优势分析

1. **方法论层面**：
   - Teacher-Student 数据合成框架非常优雅，解决了"配对数据难以获取"的根本问题
   - Masked Training 的设计直觉清晰、实现简洁，是一个很好的自监督学习范例
   - Self-Attention 替换 P2P 的 insight 具有普适性，可推广到其他多模态编辑任务

2. **工程层面**：
   - 数据规模（1400 万片段）和计算资源（128 TPU v5）体现了 Waymo 的工业级实力
   - 完整的评估框架（FID + CLIP + User Study + 3D LPIPS + 下游任务）非常全面

3. **应用层面**：
   - 直接服务于 AV 全栈仿真，具备明确的工业价值
   - 支持端到端模型评估，顺应行业趋势

### 8.2 局限性思考

1. **时序连续性**：当前方法处理的是**单帧**多视角编辑，尚未扩展到视频级编辑（帧间时序一致性）
2. **分辨率限制**：512×512 分辨率对于自动驾驶感知模型可能偏低，更高分辨率的编辑质量有待验证
3. **场景多样性**：主要聚焦于天气/时间/车辆的编辑，对行人、自行车等弱势道路使用者 (VRU) 的编辑能力未详述
4. **因果一致性**：编辑天气条件后，路面积水、车辆灯光等相关联的物理效应是否能自动生成？
5. **闭环仿真**：论文展示的是开环编辑，如何与行为仿真结合形成完整的闭环仿真 pipeline 需要进一步探索

### 8.3 与其他工作的关系

```
SceneCrafter 在自动驾驶仿真 pipeline 中的位置：

行为仿真 (Waymax, CTG++)
    ↓ 生成新的交通行为轨迹
    ↓
场景编辑 (SceneCrafter) ← 本文
    ↓ 根据轨迹编辑传感器图像
    ↓
端到端评估 (EMMA, UniAD)
    ↓ 在编辑后的仿真场景中评估
    ↓
模型迭代改进
```

SceneCrafter 填补了"行为仿真"和"传感器输入"之间的关键空白，使得端到端自动驾驶系统可以在**逼真的、可控的仿真环境**中进行全栈评估。

### 8.4 延伸阅读

| 方向 | 相关工作 |
|------|---------|
| 行为仿真 | Waymax, CTG++, SimAgents |
| 重建式仿真 | UniSim, NeRF-based |
| 生成式仿真 | MagicDrive, Panacea, Vista |
| 世界模型 | GAIA-1, Drive-WM, GenAD |
| 端到端规划 | EMMA, UniAD, VAD |

---

## 附录：模型配置详细对比

| 配置 | Teacher-G | Teacher-L | Student |
|------|-----------|-----------|---------|
| **条件** | | | |
| Time of day | ✓ | ✓ | ✓ |
| Weather | ✓ | ✓ | ✓ |
| Agent boxes | ✓ | ✗ | ✓ |
| HD maps | ✓ | ✓ | ✓ |
| Foreground masks | ✗ | ✓ | ✗ |
| Raymaps | ✓ | ✓ | ✓ |
| Source image | ✗ | ✗ | ✓ |
| **训练** | | | |
| Batch size | 128 | 128 | 128 |
| Learning rate | 5e-4 | 5e-4 | 5e-4 |
| Training steps | 100k | 100k | 200k |
| Training data | Real | Real | Synthetic |
| Masked training | ✗ | ✓ | ✗ |
| Pretrained model | CAT3D | CAT3D | Teacher-G |
| **扩散** | | | |
| Denoising steps | 50 | 50 | 50 |
| Noise schedule | Linear | Linear | Linear |
| Sampler | DDIM | DDIM | DDIM |
| z-shape | 64×64×8 | 64×64×8 | 64×64×16 |
| **能力** | | | |
| 生成 | ✓ | ✓ | ✓ |
| 编辑 | ✗ | ✗ | ✓ |

> **注意**: Student 模型 z-shape 为 64×64×**16**（因为需要拼接 source image latent），而 Teacher 模型为 64×64×8。
