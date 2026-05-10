# 22. π0 系列：通用机器人基座模型（总览与导航）

## 1. Physical Intelligence 是谁？

**Physical Intelligence (π)** 是一家成立于 2024 年的机器人 AI 公司，由一群来自 Google DeepMind、Stanford、Berkeley 的顶级研究者创办。他们的目标极其明确：

> **"构建一个能控制任何机器人、完成任何物理任务的通用基座模型。"**

这就是 **π0 (Pi-Zero)**——目前最接近"通用机器人大脑"的模型。

---

## 2. 系列文章导航

π0 系列内容丰富，已拆分为独立子篇，每篇聚焦一个阶段的技术突破：

| 子篇 | 版本 | 核心主题 | 链接 |
| :--- | :--- | :--- | :--- |
| **22a** | π0 (2024.10) | 首个通用 VLA：Flow Matching 动作头、VLM+扩散异构架构、三阶段训练范式 | [22a_Pi0_Base_Model.md](./22a_Pi0_Base_Model.md) |
| **22b** | π0-FAST (2025.01) | 频域动作编码：DCT+VQ 压缩 17.5x、自回归统一架构、5 倍训练加速 | [22b_Pi0_FAST.md](./22b_Pi0_FAST.md) |
| **22c** | π0.5 (2025.04) | 开放世界泛化：知识隔离、异构数据混合训练、高层-低层决策解耦 | [22c_Pi0_5_Open_World.md](./22c_Pi0_5_Open_World.md) |
| **22d** | π0.6 / π\*0.6 (2025.11) | 从经验中学习：Gemma3 4B 升级、Recap RL、价值函数+优势条件化策略 | [22d_Pi0_6_Learning_From_Experience.md](./22d_Pi0_6_Learning_From_Experience.md) |
| **22e** | π0.7 (2026.04) | 组合泛化涌现：多模态 Prompt 框架、跨本体迁移、单模型≥多专家 | [22e_Pi0_7_Steerable_Model.md](./22e_Pi0_7_Steerable_Model.md) |

---

## 3. π0 系列时间线

```
2024.10  π0 发布        ← 首个通用机器人基座模型，Flow Matching 动作头，10000+ 小时数据
2025.01  FAST 发布      ← 频域动作编码，自回归替代扩散，5x 加速
2025.01  π0-FAST 发布   ← π0 的高效自回归版本
2025.04  π0.5 发布      ← 首次开放世界泛化，知识隔离 + 高低层解耦
2025.H1  OpenPi 开源    ← 社区可以复现和扩展
2025.11  π0.6 发布      ← Gemma3 4B 骨干，开箱即用性能质变
2025.11  π*0.6 发布     ← 🔑 Recap RL，从自身经验中学习，全天连续运行
2026.04  π0.7 发布      ← 🔑 组合泛化涌现，单模型 ≥ 多个专家，可操控通用模型
```

## 4. π0 系列内部演进对比

| 特性 | π0 | π0-FAST | π0.5 | π0.6 / π*0.6 | π0.7 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **发布时间** | 2024.10 | 2025.01 | 2025.04 | 2025.11 | 2026.04 |
| **VLM 骨干** | PaliGemma 3B | PaliGemma 3B | PaliGemma 3B | SigLIP + Gemma3 4B | SigLIP + Gemma3 4B+ |
| **动作生成** | Flow Matching | FAST 自回归 | Flow Matching + FAST | Flow Matching + FAST | Flow Matching + FAST |
| **核心突破** | 首个通用 VLA | 5x 推理加速 | 开放世界泛化 | RL 从经验学习 | 组合泛化涌现 |
| **需要微调?** | Stage 3 可选 | 可选 | 部分任务需要 | 无需任务微调 | 无需任务微调 |
| **RL 训练** | ❌ | ❌ | ❌ | ✅ (Recap) | 蒸馏 Recap 经验 |
| **泛化方式** | 多任务 | 多任务 | 零样本新环境 | 零样本 + RL 改进 | 组合泛化新任务 |
| **长程任务** | 有限 | 有限 | 10+ 步 | 全天连续运行 | 多任务自主链式执行 |

## 5. π0 vs OpenVLA vs RT-2

| 特性 | RT-2-X | OpenVLA | π0 系列 (最新: π0.7) |
| :--- | :--- | :--- | :--- |
| **参数量** | 55B | 7B | ~5B (VLM) + 860M (Action Expert) |
| **动作头** | 自回归 Binning | 自回归 Binning | **Flow Matching + FAST** |
| **数据规模** | ~数百小时 | ~2000 小时 | **10,000+ 小时** |
| **操作精度** | 中 | 中 | **高 (扩散 + RL)** |
| **泛化能力** | 有限 | 中 | **组合泛化 (π0.7)** |
| **RL 改进** | ❌ | ❌ | **✅ (Recap)** |
| **双臂支持** | ❌ | ❌ | **✅** |
| **灵巧操作** | 有限 | 有限 | **折叠、Espresso、组装等** |
| **跨本体迁移** | 有限 | 有限 | **✅ (零样本迁移 UR5e)** |
| **开源** | ❌ | ✅ | ✅ (部分) |

> **一句话总结**：π0 系列是目前最接近"通用机器人大脑"的模型——π0 用扩散策略开创了 VLA 新范式，π0-FAST 用频域编码将速度提升 5 倍，π0.5 首次实现开放世界泛化，π0.6/π\*0.6 通过强化学习让机器人从自身经验中持续改进达到全天运行水平，π0.7 则展现出组合泛化的涌现能力——一个模型超越多个专家。从 2024 到 2026，π0 系列完整走过了"能动→快速→泛化→鲁棒→通用"的演进路径，代表了 VLA 从实验室走向真实世界的完整技术路线。

---

## 6. π0 系列开源

| 模型 | 开源状态 | 链接 |
| :--- | :--- | :--- |
| π0 | ✅ 权重 + 代码 | github.com/Physical-Intelligence/openpi |
| π0-FAST | ✅ 权重 + 代码 | 同上 |
| π0.5 | ✅ 权重 + 代码 (Knowledge Insulation 版本) | 同上 |
| π0.6 | ✅ 模型卡片公开 | 同上 |
| π*0.6 | 部分公开 | — |
| π0.7 | 未开源 | — |

## 7. π0 系列机器人平台与官方视频

### (a) 机器人硬件平台

π0 不是一个特定的机器人硬件，而是一个**通用的"机器人大脑"（软件模型）**，可以控制多种不同的机器人平台。π0 训练和部署使用了 **8 种机器人平台**：

| # | 机器人平台 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | **UR5e** | 单臂工业机械臂 | Universal Robots 的经典 6 轴协作机械臂 |
| 2 | **Bimanual UR5e** | 双臂 UR5e + Robotiq 夹爪 | 两台 UR5e 组成双臂系统，用于折叠衣物等双手任务 |
| 3 | **Franka Emika Panda** | 单臂研究型机械臂 | 7 轴力控机械臂，学术界最常用 |
| 4 | **Bimanual Trossen** | 双臂 Trossen 机械臂 | 轻量级双臂系统 |
| 5 | **Bimanual ARX** | 双臂 ARX 机械臂 | 用于折叠衣物等精细操作 |
| 6 | **Mobile Trossen** | Trossen 臂 + 移动底盘 | 可在家庭环境中移动的机器人 |
| 7 | **Mobile ARX** | ARX 臂 + 移动底盘 | π0.5 在家庭中使用的主力平台 |
| 8 | **Fibocom** | 移动机器人 | 用于数据采集 |

**典型外观**：
*   **桌面型（最常见展示形态）**：两个机械臂并排固定在桌面上，每臂约 50-80cm，末端装有平行夹爪，2-4 个摄像头（基座+腕部视角）。
*   **移动型（π0.5 家庭部署形态）**：轮式移动底盘（约 50cm 高）+ 顶部一/两个机械臂，整体约 1-1.5m 高，可在房间间自由移动。

### (b) 官方视频与博客

| 版本 | 链接 | 时长 | 内容亮点 |
| :--- | :--- | :--- | :--- |
| **π0** | https://www.pi.website/blog/pi0 | 3:36 | 折叠衣物、清理桌面、组装纸箱、装袋杂货、烤面包等 |
| **π0.5** | https://www.pi.website/blog/pi05 | 3:18 | 在全新家庭中清理厨房、整理卧室、用海绵擦拭溅洒 |
| **π\*0.6** | https://www.pi.website/blog/pistar06 | — | 连续 18 小时制作 Espresso、真实工厂包装 59 个纸箱 |
| **π0.7** | https://www.pi.website/blog/pi07 | 4:02 | 操作空气炸锅、削蔬菜、擦玻璃门、跨平台零样本折叠衣物 |

### (c) 论文 PDF

| 版本 | 下载链接 |
| :--- | :--- |
| π0 | https://www.pi.website/download/pi0.pdf |
| π0.5 | https://www.pi.website/download/pi05.pdf |
| π0.7 | https://www.pi.website/download/pi07.pdf |

### (d) 开源代码

| 项目 | 链接 |
| :--- | :--- |
| OpenPi (π0 / π0-FAST / π0.5) | https://github.com/Physical-Intelligence/openpi |

---

## 8. 主要参考文献

### π0 系列核心论文

1.  **[π0]** Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. R., Tanner, J., Vuong, Q., Walling, A., Wang, H., & Zhilinsky, O. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164.

2.  **[FAST / π0-FAST]** Pertsch, K., Nasiriany, S., Luo, J., Shi, L. R., & Levine, S. (2025). *Fast Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747.

3.  **[π0.5]** Physical Intelligence, Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Ganeshan, A., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Kirmani, S., Lee, K., Levine, S., Li-Bell, A., Lin, J., Luo, J., Mothukuri, M., Nair, S., Nasiriany, S., Pertsch, K., Shi, L. R., Tanner, J., Vuong, Q., Walling, A., Wang, H., & Zhilinsky, O. (2025). *π0.5: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054.

4.  **[Knowledge Insulation]** Driess, D., Springenberg, J. T., Ichter, B., Yu, L., Li-Bell, A., Pertsch, K., Ren, A. Z., Walke, H., Vuong, Q., Shi, L. R., et al. (2025). *Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*. NeurIPS 2025 (Spotlight).

5.  **[π0.6 Model Card]** Physical Intelligence. (2025). *π0.6 Model Card*. November 2025.

6.  **[π\*0.6 / Recap]** Physical Intelligence team. (2025). *π\*0.6: a VLA That Learns From Experience*. November 2025. Blog: pi.website/blog/pistar06.

7.  **[π0.7]** Physical Intelligence team. (2026). *π0.7: a Steerable Model with Emergent Capabilities*. April 2026. Blog: pi.website/blog/pi07.

### 基础架构与前置工作

8.  **[PaliGemma]** Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., Unterthiner, T., Keysers, D., Koppula, S., Xiong, F., Houlsby, N., Gritsenko, A. A., Garg, S., Minderer, M., Mustafa, B., & Zhai, X. (2024). *PaliGemma: A versatile 3B VLM for transfer*. arXiv:2407.07726.

9.  **[Gemma 3]** Gemma Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., et al. (2025). *Gemma 3 Technical Report*. arXiv:2503.19786.

10.  **[Flow Matching]** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. (2023). *Flow Matching for Generative Modeling*. ICLR 2023.

11.  **[Diffusion Policy]** Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS 2023.

12.  **[Action Chunking / ACT]** Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS 2023.

### 对比工作

13.  **[RT-2]** Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., ... & Zitkovich, B. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv:2307.15818.

14.  **[OpenVLA]** Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., ... & Finn, C. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246.

15. **[Open X-Embodiment]** Open X-Embodiment Collaboration. (2024). *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. ICRA 2024.

### 动作表示相关

16. **[DCT in Signal Processing]** Ahmed, N., Natarajan, T., & Rao, K. R. (1974). *Discrete Cosine Transform*. IEEE Transactions on Computers, C-23(1), 90–93.

17. **[VQ-VAE]** van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NeurIPS 2017.

### 强化学习相关

18. **[DAgger]** Ross, S., Gordon, G., & Bagnell, D. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011. (复合误差问题的经典分析)
