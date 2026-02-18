# 29. BEV 感知：鸟瞰视角下的统一表示

> BEV (Bird's Eye View) 表示是现代自动驾驶感知的基石——它将多个相机和传感器的输入统一到同一个俯视平面坐标系中，为下游的检测、预测和规划提供一致的空间表达。

[⬅️ 返回 AD 目录](./README.md) | [⬅️ 28. 架构概述](./28_AD_Overview.md) | [30. UniAD ➡️](./30_UniAD.md)

---

## 1. 为什么需要 BEV？

自动驾驶车辆通常搭载 **6–8 个环视相机**，每个相机只能提供一个视角的 2D 图像。但驾驶决策需要在**三维空间**中进行——判断前车距离、相邻车道是否有车、行人将往哪走。这之间存在根本性的表示鸿沟：

```
问题：多视角 2D 图像 → 如何得到统一的 3D 空间理解？

       前视相机        左前相机       右前相机
      ┌────────┐     ┌────────┐    ┌────────┐
      │ 2D 图像 │     │ 2D 图像 │    │ 2D 图像 │
      └───┬────┘     └───┬────┘    └───┬────┘
          │              │             │
          ▼              ▼             ▼
     ┌─────────────────────────────────────┐
     │          ??? 视角转换 ???             │
     └──────────────────┬──────────────────┘
                        ▼
              ┌──────────────────┐
              │  BEV 鸟瞰图特征    │
              │  (统一 X-Y 平面)   │
              │  H × W × C        │
              └──────────────────┘
                        ▼
              检测 / 分割 / 预测 / 规划
```

### BEV 表示的三大优势

1. **空间统一性**：所有相机和传感器的信息映射到同一个自车坐标系 (ego-centric coordinate)，消除了视角差异。
2. **度量准确性**：BEV 网格中每个格子对应真实世界中固定的物理尺寸（如 0.5m × 0.5m），距离和尺寸可以直接度量。
3. **融合友好性**：相机特征、LiDAR 点云、雷达检测结果都可以自然地投影到同一个 BEV 平面上进行融合。

---

## 2. 从 2D 到 BEV 的两大技术路线

将 2D 图像特征转换到 BEV 空间是最核心的技术挑战，目前有两条主流路线：

```
技术路线对比：

路线 A: 自底向上 (Bottom-Up)              路线 B: 自顶向下 (Top-Down)
────────────────────────                ────────────────────────
代表: LSS, BEVDet                       代表: BEVFormer, DETR3D

2D 图像 → 预测深度 → 提升到 3D            BEV Query → 投影到图像 → 采样特征
(显式几何推理)                            (隐式注意力学习)
```

### (a) 自底向上 (Bottom-Up): LSS 方案

**Lift-Splat-Shoot (LSS, Philion & Fidler, 2020)** 是 BEV 感知的开山之作。核心思路是三步走：

**Step 1 — Lift (提升)**：对图像中每个像素预测一个**离散深度分布 (Categorical Depth Distribution)**。

对于位置 $(u, v)$ 处的图像特征 $\mathbf{c} \in \mathbb{R}^C$，网络同时预测深度分布 $\boldsymbol{\alpha} \in \mathbb{R}^D$（$D$ 个离散深度桶），通过外积构造 3D 特征：

$$
\mathbf{f}_{3D}(u, v, d) = \alpha_d \cdot \mathbf{c}(u, v), \quad d \in \{d_1, d_2, \ldots, d_D\}
$$

其中 $\alpha_d = \text{softmax}(\text{DepthNet}(\mathbf{x}))_d$ 是像素 $(u,v)$ 落在深度 $d$ 的概率。

**Step 2 — Splat (投射)**：利用相机内参 $K$ 和外参 $[R|t]$ 将 3D 特征投射到 BEV 网格中：

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = R^{-1} \left( d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - t \right)
$$

将 $(X, Y)$ 量化到 BEV 网格坐标，对落入同一格子的特征进行 **求和池化 (Sum Pooling)**。

**Step 3 — Shoot (应用)**：在 BEV 特征上接下游任务头（检测、分割等）。

```
LSS 流程图：

 图像 I ──→ Backbone ──→ 图像特征 c(u,v) ──┐
                                            │ outer product
                          DepthNet ──→ α(d) ─┘
                                            ↓
                                  3D 特征: α_d · c(u,v)
                                            ↓
                              相机投影 (K, R, t)
                                            ↓
                                ┌──────────────────┐
                                │  BEV Grid (求和)   │
                                └──────────────────┘
```

**BEVDet (Huang et al., 2022)** 在 LSS 基础上增加了：
*   更强的图像 Backbone (如 ResNet-50 → Swin Transformer)
*   时序 BEV 对齐：将历史帧的 BEV 特征通过自车运动 (ego-motion) 变换后与当前帧拼接
*   数据增强策略提升鲁棒性

### (b) 自顶向下 (Top-Down): BEVFormer 方案

**BEVFormer (Li et al., 2022)** 走了完全不同的路线——不显式预测深度，而是让**可学习的 BEV Query** 主动去图像中"查询"需要的信息。

**核心架构**：

```
                    ┌─────────────────────────────────────────────┐
                    │            BEVFormer Encoder Layer            │
                    │                                              │
 BEV Queries ──→  │  Temporal Self-Attention                      │
 Q ∈ R^(H×W×C)    │       ↓                                      │
                    │  Spatial Cross-Attention ←── Multi-Cam Feats │
                    │       ↓                                      │
                    │  Feed-Forward Network                        │
                    │       ↓                                      │
                    │  Updated BEV Features                        │
                    └─────────────────────────────────────────────┘
                                      ×L 层
```

**Spatial Cross-Attention (空间交叉注意力)**：

对于 BEV 网格中位置 $p = (x, y)$ 处的 Query $Q_p$：

1. 沿 $Z$ 轴采样 $N_{\text{ref}}$ 个 3D 参考点：$(x, y, z_i)$，$i = 1, \ldots, N_{\text{ref}}$
2. 将每个 3D 参考点投影到各个相机图像上：$\mathcal{P}(x, y, z_i, \text{cam}_j)$
3. 在投影位置使用**可变形注意力 (Deformable Attention)** 采样图像特征

$$
\text{SCA}(Q_p) = \frac{1}{|\mathcal{V}_{\text{hit}}|} \sum_{j \in \mathcal{V}_{\text{hit}}} \sum_{i=1}^{N_{\text{ref}}} \text{DeformAttn}(Q_p, \mathcal{P}(p, z_i, j), F_j)
$$

其中 $\mathcal{V}_{\text{hit}}$ 是 3D 参考点投影落入的相机集合，$F_j$ 是第 $j$ 个相机的图像特征。

**Temporal Self-Attention (时序自注意力)**：

将上一时刻的 BEV 特征 $B_{t-1}$ 根据自车运动进行空间对齐，然后与当前 BEV Query 做自注意力：

$$
\text{TSA}(Q_p) = \text{DeformAttn}(Q_p, p, \{Q_p, B'_{t-1}(p)\})
$$

其中 $B'_{t-1}$ 是经过 ego-motion 对齐后的历史 BEV 特征。

### 两条路线的对比

| 维度 | 自底向上 (LSS / BEVDet) | 自顶向下 (BEVFormer) |
|------|------------------------|---------------------|
| **深度信息** | 显式预测深度分布 | 隐式学习，无需深度 |
| **计算瓶颈** | 3D 体素构建和池化 | Deformable Attention |
| **精度** | 依赖深度预测质量 | 受限于采样点数量 |
| **LiDAR 监督** | 可用 LiDAR 深度做监督 | 不需要 |
| **推理速度** | 较快（无注意力开销） | 较慢（多层 Attention） |
| **代表模型** | BEVDet, BEVDepth, SOLOFusion | BEVFormer, BEVFormer v2 |

---

## 3. 多传感器融合: BEVFusion

虽然纯视觉 BEV 已经取得显著进展，但 **LiDAR 提供精确的深度信息**、**相机提供丰富的语义信息**——两者互补。**BEVFusion (Liu et al., 2023)** 提出在 BEV 空间中统一融合多种传感器。

```
融合架构：

Camera 分支:                         LiDAR 分支:
┌────────────┐                      ┌────────────────┐
│ 6 × 图像    │                      │ 点云            │
│ Images      │                      │ Point Cloud     │
└─────┬──────┘                      └──────┬─────────┘
      ↓                                    ↓
  Image Backbone                     3D Backbone
  (Swin / ResNet)                   (VoxelNet / PointPillars)
      ↓                                    ↓
  LSS / BEVFormer                    Flatten to BEV
      ↓                                    ↓
┌─────────────┐                    ┌──────────────┐
│ Camera BEV   │                    │ LiDAR BEV     │
│ Features     │                    │ Features      │
└─────┬───────┘                    └──────┬───────┘
      │                                    │
      └──────────┬─────────────────────────┘
                 ↓
        ┌─────────────────┐
        │ BEV Fusion       │
        │ (Concat + Conv / │
        │  Attention)       │
        └────────┬────────┘
                 ↓
        Unified BEV Features
                 ↓
        检测 / 分割 / 预测
```

### 为什么 BEV 是理想的融合空间？

*   **空间对齐**：相机和 LiDAR 的 BEV 特征天然处于同一坐标系，无需复杂的对齐操作。
*   **分辨率可控**：BEV 网格分辨率可以统一设定（如 0.5m/格），不受传感器原始分辨率的限制。
*   **时序扩展自然**：不同时间步的 BEV 特征可以通过 ego-motion 简单变换后堆叠。

BEVFusion 关键设计：使用 **高效 BEV 池化 (Efficient BEV Pooling)** 将 LSS 的 GPU 显存占用降低 40×，使得相机分支的延迟与 LiDAR 分支匹配（约 25ms），实现真正的实时融合。

---

## 4. BEV 感知的下游任务

BEV 统一表示的核心价值在于它可以同时服务多个下游任务：

### (a) 3D 目标检测 (3D Object Detection)

在 BEV 特征上预测 3D 边界框 $(x, y, z, w, l, h, \theta)$，包括位置、尺寸和朝向角。

### (b) BEV 语义分割 (BEV Segmentation)

对 BEV 网格逐像素分类：道路 (Road)、车道线 (Lane)、人行横道 (Crosswalk)、车辆 (Vehicle) 等。

### (c) 运动预测 (Motion Prediction)

基于多帧 BEV 特征预测周围 Agent 的未来轨迹。

### (d) 在线地图构建 (Online Map Construction)

从 BEV 特征直接回归矢量化地图元素，替代传统的离线高精地图。

### 代表性 BEV 模型对比

| 模型 | 年份 | 2D→BEV 方式 | 时序融合 | 传感器 | 核心贡献 |
|------|------|-------------|---------|--------|---------|
| **LSS** | 2020 | 深度预测 + Splat | ✗ | Camera | 首创 Lift-Splat 范式 |
| **BEVDet** | 2022 | LSS + 增强 | 时序拼接 | Camera | 工程优化，数据增强 |
| **BEVFormer** | 2022 | Deformable Attn | Temporal Self-Attn | Camera | Query-based BEV 生成 |
| **BEVFusion** | 2023 | LSS (相机) + 体素化 (LiDAR) | ✗ | Camera + LiDAR | 高效多模态融合 |
| **StreamPETR** | 2023 | Sparse 3D Query | 流式时序传播 | Camera | 稀疏 Query + 长时序 |
| **Far3D** | 2023 | Adaptive Query | 时序累积 | Camera | 远距离检测 (>150m) |
| **SparseBEV** | 2024 | Adaptive Sampling | 自适应时序 | Camera | 稀疏化 + 自适应 |

---

## 5. BEV 的局限与演进

### 固有局限

1. **垂直信息丢失**：BEV 是将 3D 空间压缩到 X-Y 平面的 2D 投影。对于需要精确高度信息的场景（如立交桥上下层、低矮障碍物）表达能力不足。

2. **深度估计误差**：纯视觉方案中，深度预测的误差随距离增大而显著放大。在 50m 以外，单目深度估计的误差可达数米。

3. **计算开销**：构建稠密 BEV 特征（如 200×200 网格 × 256 通道）的计算量和显存占用巨大，限制了实时性。

4. **BEV 分辨率权衡**：分辨率高则计算量大，分辨率低则小目标检测困难。

### 演进方向：从 BEV 到 3D 占据网络 (Occupancy Networks)

为解决 BEV 的垂直信息丢失问题，**占据网络 (Occupancy Networks)** 将 BEV 的 2D 网格扩展为 **3D 体素网格**，每个体素预测其占据状态和语义类别：

```
BEV 表示:                    3D 占据表示:
┌──────────────┐             ┌──────────────┐
│  X-Y 平面     │             │  X-Y-Z 体素   │
│  (H × W × C)  │    →→→     │  (H × W × Z × C) │
│  高度信息压缩   │             │  保留高度信息   │
└──────────────┘             └──────────────┘

     BEV Grid                   Occupancy Grid
  (鸟瞰图，2D)                (体素化世界，3D)
```

这一演进代表了从"平面感知"到"体积感知"的范式跳跃，详见 → [31. 3D 占据网络](./31_Occupancy_Networks.md)。

---

## 6. 总结

```
BEV 感知的技术脉络：

2D 多相机感知 (独立处理，信息割裂)
        ↓
LSS (2020): 显式深度预测 → Lift → Splat → BEV
        ↓
BEVFormer (2022): 可学习 Query + Deformable Attention → BEV
        ↓
BEVFusion (2023): 相机 + LiDAR 在 BEV 空间统一融合
        ↓
StreamPETR / SparseBEV (2023-24): 稀疏化 + 流式时序 → 更高效
        ↓
Occupancy Networks: 从 2D BEV → 3D 体素 (保留高度信息)
```

> **一句话总结**：BEV 感知将多个相机/传感器的 2D 输入统一到鸟瞰视角的 3D 空间表示中，是端到端自动驾驶感知-预测-规划流水线的基石。它有两条技术路线——显式深度 (LSS) 和隐式注意力 (BEVFormer)——最终都收敛到 BEV 空间，为多传感器融合和多任务学习提供统一接口。

---

[⬅️ 返回 AD 目录](./README.md) | [⬅️ 28. 架构概述](./28_AD_Overview.md) | [30. UniAD ➡️](./30_UniAD.md) | [31. 占据网络 ➡️](./31_Occupancy_Networks.md)
