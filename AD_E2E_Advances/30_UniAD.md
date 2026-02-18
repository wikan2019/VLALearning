# 30. UniAD：规划导向的统一自动驾驶框架

> **Planning-oriented Autonomous Driving** — CVPR 2023 Best Paper  
> 论文：*Planning-oriented Autonomous Driving* (Li et al., 2023)  
> 代码：[github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

[⬅️ 返回 AD 目录](./README.md) | [⬆️ 29. BEV 感知](./29_BEV_Perception.md) | [➡️ 31. 占据网络](./31_Occupancy_Networks.md)

---

## 1. UniAD 的核心思想

传统自动驾驶采用**模块化流水线**——检测、跟踪、预测、规划各自独立训练和优化：

```
Camera → 检测器 → 跟踪器 → 预测器 → 规划器 → 控制
            ↑         ↑         ↑         ↑
         独立 Loss  独立 Loss  独立 Loss  独立 Loss
```

这种设计存在两个根本问题：

| 问题 | 说明 |
|------|------|
| **信息瓶颈** | 模块间通过固定格式的中间结果传递信息（如 3D bbox），丢失大量特征细节 |
| **目标不一致** | 各模块独立优化各自的指标，但检测最优 ≠ 规划最优 |

UniAD 提出 **"Planning-oriented"** 设计哲学：**所有感知/预测任务都服务于最终的规划目标**。具体来说：

- 将 Detection、Tracking、Mapping、Motion Prediction、Occupancy Prediction、Planning **六大任务统一在一个 Transformer 网络中**
- 任务之间通过 **query-based attention** 传递信息，而非有损的中间输出
- 梯度从 planning loss 端到端回传到所有感知模块，实现**全局联合优化**

> **核心洞察**：上游任务的价值不在于其自身指标，而在于对最终规划的贡献。

---

## 2. 架构详解

### 2.1 总体架构

```
                          UniAD 全流程架构
 ═══════════════════════════════════════════════════════════════

 多相机图像 (6 views)
      │
      ▼
 ┌─────────────┐
 │ BEV Encoder │  ← BEVFormer backbone (ResNet-101 + Transformer)
 │ (BEVFormer) │     多尺度 deformable attention
 └──────┬──────┘     输出: BEV Features B ∈ R^{H×W×C}
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
 ┌──────────────┐                          ┌──────────────┐
 │  TrackFormer │  ← 检测 + 跟踪           │   MapFormer  │ ← 在线地图分割
 │  (Det+Track) │     track queries Q_A     │  (HD Map)    │   map queries Q_M
 └──────┬───────┘                          └──────┬───────┘
        │  agent queries                          │  map queries
        │  (位置/速度/类别特征)                     │  (车道/道路边界/分隔线)
        ├─────────────┬───────────────────────────┘
        │             ▼
        │     ┌───────────────┐
        │     │  MotionFormer │  ← 交互式运动预测
        │     │  (Prediction) │     agent×agent + agent×map attention
        │     └───────┬───────┘
        │             │  motion queries (多模态未来轨迹)
        │             │
        ├─────────────┤
        ▼             ▼
 ┌──────────────┐  ┌──────────┐
 │   OccFormer  │  │ Planner  │  ← ego 轨迹规划
 │  (Occupancy) │  │          │     融合所有 query 信息
 └──────────────┘  └──────────┘
   未来占据网格        ego 未来轨迹 + 控制信号
```

### 2.2 各模块详解

#### BEV Encoder

基于 [BEVFormer](./29_BEV_Perception.md)，使用 **spatial cross-attention** 和 **temporal self-attention** 将多相机 2D 特征提升到统一的 BEV 空间：

```
输入: 6 个相机图像 I_t ∈ R^{6×3×H_img×W_img}
     + 历史 BEV 特征 B_{t-1}
输出: BEV 特征 B_t ∈ R^{200×200×256}
```

#### TrackFormer — 检测 + 跟踪

采用 DETR 风格的 query-based 检测，核心创新是 **track queries 的时序传递**：

- **Detection queries**：可学习的位置查询，用于发现新目标
- **Track queries**：从上一帧传递的查询，持续跟踪已有目标
- 输出 agent queries \( Q_A \in \mathbb{R}^{N_a \times C} \)，编码每个目标的类别、位置、速度、外观特征

#### MapFormer — 在线地图构建

使用 panoptic segmentation 思路对 BEV 特征做语义分割：

- 分割类别：road boundary、lane divider、pedestrian crossing
- 输出 map queries \( Q_M \in \mathbb{R}^{N_m \times C} \)，编码道路结构信息

#### MotionFormer — 交互式运动预测

这是 UniAD 最核心的模块，使用三种 attention 实现交互式预测：

```
                    MotionFormer 内部结构
 ══════════════════════════════════════════════════

 Q_A (agent queries)    Q_M (map queries)
      │                       │
      ▼                       │
 ┌─────────────────┐          │
 │ Agent-Agent Attn │  ← 车辆间交互建模    │
 └────────┬────────┘          │
          ▼                   ▼
 ┌─────────────────────────────┐
 │    Agent-Map Attn           │  ← 车辆与道路结构交互
 └────────────┬────────────────┘
              ▼
 ┌─────────────────┐
 │  Mode Attention  │  ← 多模态轨迹预测 (K=6 modes)
 └────────┬────────┘
          ▼
    Motion Queries Q_motion
    (每个 agent K 条未来轨迹 + 置信度)
```

预测目标：每个 agent 未来 T=12 步（6 秒）的 K=6 条多模态轨迹。

#### OccFormer — 未来占据预测

预测未来 T 步的稠密占据网格，捕捉无法用 bbox 表示的场景动态（如道路施工区域、异形障碍物）：

\[
\hat{O}_{t+1:t+T} = \text{OccFormer}(B_t, Q_A, Q_M)
\]

#### Planner — 规划模块

融合所有上游信息，输出 ego vehicle 的未来轨迹：

\[
\tau_{ego} = \text{Planner}(Q_{ego}, Q_A, Q_M, Q_{motion}, \hat{O})
\]

其中 \( Q_{ego} \) 是可学习的 ego query，通过 cross-attention 聚合所有上游 query 的信息。

---

## 3. 关键设计："Query-based 任务串联"

UniAD 的核心设计在于**所有模块之间通过 queries 和 attention 通信**，而非有损的中间输出：

```
传统模块化:
  检测 ──[bbox list]──→ 跟踪 ──[track list]──→ 预测 ──[traj list]──→ 规划
         有损压缩            有损压缩            有损压缩

UniAD query-based:
  TrackFormer ──[Q_A: 高维特征]──→ MotionFormer ──[Q_motion]──→ Planner
       ↑                              ↑
  BEV Features                   MapFormer [Q_M]
```

**为什么 query-based 设计更优？**

| 维度 | 模块化传递 | Query-based 传递 |
|------|-----------|------------------|
| 信息量 | 3D bbox (7 个数值) | 256 维特征向量 |
| 梯度流 | 被中间输出截断 | 端到端可微 |
| 容错性 | 上游错误直接传递 | Attention 可做软选择，降低错误影响 |
| 灵活性 | 固定接口格式 | 自适应特征聚合 |

Query 之间的信息流动关系：

```
Track Queries (Q_A)  ←─── BEV Features
       │
       ├──→ Map Queries (Q_M) ←─── BEV Features
       │          │
       ▼          ▼
  Motion Queries (Q_motion)
       │
       ▼
  Ego Planning Query (Q_ego) ←── Occupancy Features
       │
       ▼
  输出轨迹 τ_ego
```

---

## 4. 训练策略

### 4.1 多任务联合损失

UniAD 使用加权多任务损失进行端到端联合训练：

\[
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{det} + \lambda_2 \mathcal{L}_{track} + \lambda_3 \mathcal{L}_{map} + \lambda_4 \mathcal{L}_{motion} + \lambda_5 \mathcal{L}_{occ} + \lambda_6 \mathcal{L}_{plan}
\]

各任务损失的具体构成：

| 任务 | 损失函数 | 说明 |
|------|---------|------|
| Detection | Focal Loss + L1 | 分类 + 回归（DETR 风格） |
| Tracking | 匈牙利匹配 + L1 | 时序 query 关联 |
| Mapping | Dice Loss + CE | 语义分割 |
| Motion | minADE + Cls Loss | 最优模态回归 + 模态分类 |
| Occupancy | CE + Lovász Loss | 占据网格分类 |
| Planning | L2 + Collision Loss | 轨迹回归 + 碰撞惩罚 |

### 4.2 训练流程

```
阶段 1: 预训练 BEV Encoder (BEVFormer, 24 epochs on nuScenes)
            ↓
阶段 2: 联合训练全部模块 (20 epochs, 8×A100)
            ↓
阶段 3: 微调 Planner (可选)
```

关键训练细节：
- **损失平衡**：不同任务的损失量级差异大，需要仔细调节 λ 权重
- **梯度冲突**：多任务间可能存在梯度冲突，UniAD 通过 task-specific normalization 缓解
- **端到端梯度流**：planning loss 的梯度回传到 BEV encoder，驱动感知特征学习"对规划有用"的表示

---

## 5. 性能与影响

### 5.1 nuScenes 基准结果

| 方法 | NDS ↑ | AMOTA ↑ | Map mIoU ↑ | minADE ↓ | minFDE ↓ | Collision ↓ | L2 (plan) ↓ |
|------|-------|---------|------------|----------|----------|-------------|-------------|
| 独立模块组合 | 0.386 | 0.359 | 0.310 | 1.02 | 1.56 | 0.71% | 1.02 |
| ST-P3 | — | — | — | — | — | 0.87% | 1.33 |
| **UniAD (Base)** | 0.498 | 0.390 | 0.332 | 0.71 | 1.02 | **0.31%** | **0.69** |

关键结论：
- **规划碰撞率 0.31%**：当时最优，显著优于模块化方案的 0.71%
- **运动预测 minADE 0.71m**：证明统一框架在子任务上也有竞争力
- **全任务统一提升**：不是牺牲子任务来换取规划性能，而是全面提升

### 5.2 消融实验

UniAD 做了系统的消融实验验证每个模块的必要性：

| 配置 | Collision ↓ | L2 (plan) ↓ |
|------|-------------|-------------|
| Planner only (无上游) | 0.96% | 1.15 |
| + TrackFormer | 0.68% | 0.91 |
| + MapFormer | 0.53% | 0.82 |
| + MotionFormer | 0.38% | 0.73 |
| + OccFormer (完整) | **0.31%** | **0.69** |

每个模块都为规划带来了可量化的增益，验证了 "planning-oriented" 的设计理念。

### 5.3 学术影响

UniAD 作为 CVPR 2023 Best Paper，对后续研究产生了深远影响：

```
UniAD (2023)
  ├──→ VAD (2024): 向量化场景表示 + 端到端规划
  ├──→ GenAD (2024): 生成式轨迹预测 + 规划
  ├──→ DriveTransformer (2025): 更高效的 D-Former 架构
  ├──→ UniUGP (2025): 统一生成式框架
  └──→ EMMA (Waymo, 2024): 多模态大模型驾驶
```

---

## 6. UniAD 2.0 (2025)

OpenDriveLab 于 2025 年发布了 [UniAD 2.0](https://github.com/OpenDriveLab/UniAD/tree/v2.0)，主要更新：

| 维度 | UniAD 1.0 | UniAD 2.0 |
|------|-----------|-----------|
| 框架 | MMDetection3D v0.x | OpenDriveLab 现代化框架 |
| 评估 | nuScenes 开环 | + nuPlan 闭环 + NAVSIM |
| 训练 | 单阶段联合训练 | 分阶段渐进训练 |
| 速度 | ~2 FPS | 优化后 ~5 FPS |
| 代码 | 研究原型 | 工程化重构 |

UniAD 2.0 的核心意义：
- **闭环评估**：从开环 nuScenes 扩展到闭环 nuPlan 和 NAVSIM，结果更具参考价值
- **工程可用性**：代码重构后更易复现和二次开发
- **与后续工作对齐**：统一了与 VAD、GenAD 等后续工作的评估协议

---

## 7. 总结与展望

UniAD 的核心贡献可以总结为三点：

1. **理念贡献**：首次系统论证 "以规划为目标的统一框架" 优于独立优化的模块化流水线
2. **架构贡献**：提出 query-based 任务串联机制，实现端到端可微的多任务协作
3. **实验贡献**：在全部子任务和最终规划上均取得当时最优，证明统一 > 分离

**局限性与后续改进方向**：

| 局限 | 后续改进 |
|------|---------|
| 仅支持开环评估 | UniAD 2.0 支持闭环；VAD 系列改进 |
| 推理速度慢 (~2 FPS) | DriveTransformer 实现 45 FPS |
| 缺乏语言理解能力 | EMMA、DriveVLM 引入 VLM |
| 固定预测时域 | GenAD 使用生成式方法灵活预测 |

> **一句话总结**：UniAD 用一个 Transformer 证明了"一体化"比"各自为战"更强，奠定了端到端自动驾驶的架构范式。

---

[⬅️ 返回 AD 目录](./README.md) | [⬆️ 29. BEV 感知](./29_BEV_Perception.md) | [➡️ 31. 占据网络](./31_Occupancy_Networks.md)
