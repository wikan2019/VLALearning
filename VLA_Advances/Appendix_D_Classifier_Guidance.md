# Appendix D. Classifier Guidance 与 Classifier-Free Guidance

> 扩散模型如何从「能生成」走到「可控生成」：两条 guidance 路线分别要解决什么、怎么解

[⬅️ 返回 VLA 索引](./README.md) | [Appendix A. Flow Matching](./Appendix_A_Flow_Matching.md)

---

## 1. 共同背景：无条件扩散「能画」，但「画不准」

标准扩散 / 去噪模型学的是噪声预测（或等价的 score）：

$$\epsilon_\theta(x_t, t) \approx \epsilon, \qquad \nabla_{x_t}\log p_t(x_t) \approx -\frac{1}{\sigma_t}\epsilon_\theta(x_t,t)$$

无条件采样时，每一步只跟着数据分布的 score 走，结果是：

- **多样性好**：能覆盖训练集里各种模式
- **可控性弱**：无法指定「生成一只猫 / 按这句话画图 / 生成某一类动作」

把条件 \(c\)（类别、文本、机器人观测+指令等）塞进网络，得到 \(\epsilon_\theta(x_t,t,c)\)，已经是**条件扩散**。但实践中还会碰到：

1. 条件信号在去噪过程中往往偏「软」——模型既要匹配数据分布，又要服从条件，两者折中后**对条件的忠诚度不够**
2. 希望有一个**可调旋钮**（guidance scale），运行时加强或减弱「听条件」的程度，而不必为每个强度重训模型

**Guidance** 就是在采样时改写每一步的更新方向，把概率质量推向「更符合条件 \(c\)」的区域。下面两条路线用不同方式构造这个推力。

---

## 2. Classifier Guidance：用外部分类器推一把

### 2.1 要解决什么问题？

| 痛点 | 说明 |
| :--- | :--- |
| 条件不够「硬」 | 仅靠条件去噪网络 \(\epsilon_\theta(x_t,t,c)\)，采样结果经常偏离目标类/文本 |
| 希望可调节强度 | 需要一个采样期超参，加大后更「像」目标条件，减小则更接近无条件多样性 |
| 已有无条件/弱条件模型 | 希望**不必重训**整套扩散骨干，只加一个辅助模块就能做可控采样 |

核心问题一句话：**如何在采样时，把扩散轨迹显式地推向 \(p(x\mid c)\)，且推力大小可控？**

### 2.2 解决思路（数学）

目标是按条件分布采样。Bayes 给出：

$$\nabla_{x_t}\log p(x_t\mid c) = \underbrace{\nabla_{x_t}\log p(x_t)}_{\text{无条件 score（扩散模型）}} + \underbrace{\nabla_{x_t}\log p(c\mid x_t)}_{\text{分类器对 }x_t\text{ 的梯度}}$$

因此：

1. **训练一个噪声感知分类器** \(p_\phi(c\mid x_t,t)\)：在加噪图像 \(x_t\) 上预测类别/条件（必须见过各噪声水平 \(t\)）
2. **采样时**把分类器梯度加到扩散 score 上（或等价地改噪声预测）：

$$\tilde{\epsilon}(x_t,t) = \epsilon_\theta(x_t,t) - s\cdot\sigma_t\,\nabla_{x_t}\log p_\phi(c\mid x_t,t)$$

其中 \(s\) 为 **guidance scale**：

- \(s=0\)：退化为无条件（或原条件）采样
- \(s\) 增大：更强地朝「分类器认为属于 \(c\)」的方向挪动 \(x_t\)

（若骨干本身已是条件模型，还可与条件预测组合；经典形式常建立在无条件扩散 + 外部分类器上。）

### 2.3 算法直觉（伪代码）

```
训练阶段:
  1. 训扩散模型 ε_θ(x_t, t)          # 学数据分布
  2. 训分类器 p_φ(c | x_t, t)         # 在噪声图上认类

采样阶段 (给定目标条件 c, scale s):
  x_T ~ N(0, I)
  for t = T … 1:
      ε = ε_θ(x_t, t)
      g = ∇_{x_t} log p_φ(c | x_t, t)   # 分类器反传对 x 的梯度
      ε̃ = ε - s · σ_t · g              # 引导后的噪声预测
      x_{t-1} = denoising_step(x_t, ε̃, t)
  return x_0
```

### 2.4 代价与局限

| 问题 | 表现 |
| :--- | :--- |
| 要训第二个模型 | 噪声感知分类器成本高，且要与扩散的噪声日程对齐 |
| 高 \(s\) 易出伪影 | 梯度上升类似对抗扰动，图像可出现过饱和、不自然纹理 |
| 条件形态受限 | 天然适合**离散类别**；开放文本、复杂多模态条件更难训「分类器」 |
| 两套模型耦合 | 换骨干或噪声日程，分类器往往要跟着重训 |

**代表工作**：Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (NeurIPS 2021) —— 用 classifier guidance 把扩散推到当时 ImageNet 生成 SOTA。

---

## 3. Classifier-Free Guidance (CFG)：不训分类器，用「有条件 − 无条件」代替

### 3.1 要解决什么问题？

Classifier Guidance 证明了「采样时加推力」有效，但工程与效果上仍难受：

| Classifier Guidance 留下的问题 | 期望 |
| :--- | :--- |
| 必须另训噪声感知分类器 | **不要**额外分类器 |
| 高 guidance 易对抗伪影 | 推力来自生成模型自身，更稳 |
| 文本等开放条件难做成分类器 | 条件直接进同一个去噪网络即可 |
| 两模型维护成本高 | **一个网络**同时提供有条件与无条件预测 |

核心问题一句话：**能否在不使用外部分类器的情况下，仍得到可调强度的条件引导？**

### 3.2 解决思路

关键观察：分类器梯度 \(\nabla_{x}\log p(c\mid x)\) 可用「有条件 score 与无条件 score 之差」来近似表达（同一扩散框架下的恒等变形）。因此不必显式训 \(p_\phi(c\mid x)\)，只要网络能同时给出：

- **有条件**噪声预测 \(\epsilon_\theta(x_t,t,c)\)
- **无条件**噪声预测 \(\epsilon_\theta(x_t,t,\varnothing)\)（条件置空 / dropout）

采样时做线性外推（这就是 CFG）：

$$\tilde{\epsilon}(x_t,t,c) = \epsilon_\theta(x_t,t,\varnothing) + w\cdot\bigl(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\bigr)$$

等价写法：

$$\tilde{\epsilon} = (1-w)\,\epsilon_{\text{uncond}} + w\,\epsilon_{\text{cond}}$$

其中 \(w\) 为 **CFG scale**（常取 \(w>1\)，即沿「条件方向」外推）：

- \(w=0\)：纯无条件
- \(w=1\)：普通条件采样（无额外引导）
- \(w>1\)：加强条件忠诚度（更贴 prompt，多样性通常下降）

### 3.3 训练方法：条件随机丢弃

同一个网络 \(\epsilon_\theta(x_t,t,c)\)：

1. 以概率 \(p_{\text{uncond}}\)（常见 10%–20%）把 \(c\) 换成空条件 \(\varnothing\)（文本用空串 / 可学习 null embedding，类别用特殊 null class）
2. 其余样本正常带条件训练

这样一次训练同时学会「听条件」和「不听条件」，推理时各跑一次（或 batch 内拼成两条），再按上面公式混合。

### 3.4 算法直觉（伪代码）

```
训练阶段 (同一个 ε_θ):
  for each batch (x_0, c):
      t ~ U({1..T});  x_t = add_noise(x_0, t)
      with prob p_uncond: c ← ∅          # 条件 dropout
      loss = ||ε_θ(x_t, t, c) - ε||²

采样阶段 (给定 c, scale w):
  x_T ~ N(0, I)
  for t = T … 1:
      ε_c  = ε_θ(x_t, t, c)             # 有条件
      ε_u  = ε_θ(x_t, t, ∅)             # 无条件
      ε̃   = ε_u + w · (ε_c - ε_u)      # CFG
      x_{t-1} = denoising_step(x_t, ε̃, t)
  return x_0
```

### 3.5 代价与局限

| 点 | 说明 |
| :--- | :--- |
| 推理约 2× 前向 | 每步要 cond + uncond（可用 batch 并行；部分蒸馏/引导蒸馏可减负） |
| \(w\) 过大 | 过饱和、细节崩坏、多样性塌缩（与 CG 高 \(s\) 类似，但通常更可控） |
| 必须联合训 null 条件 | 纯条件、从未见过 \(\varnothing\) 的旧模型，不能直接套标准 CFG |

**代表工作**：Ho & Salimans, *Classifier-Free Diffusion Guidance* (NeurIPS 2022 Workshop / 后续广泛引用)。Stable Diffusion、Imagen、大量文生图 / 视频模型默认使用 CFG。

---

## 4. 对照一览

| 维度 | Classifier Guidance | Classifier-Free Guidance |
| :--- | :--- | :--- |
| **要解决的核心问题** | 用可调推力把采样推向目标条件 | **同样要可调推力**，但去掉外部分类器 |
| **推力从哪来** | \(\nabla_x\log p_\phi(c\mid x_t,t)\) | \(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}}\) |
| **额外模型** | 需要噪声感知分类器 | 不需要；同一去噪网络 |
| **训练改动** | 另训分类器；扩散骨干可不动 | 训练时随机 dropout 条件 |
| **推理开销** | 每步分类器反传（梯度） | 每步两次前向（cond + uncond） |
| **条件类型** | 类别最自然 | 文本、多模态条件更友好 |
| **高 scale 风险** | 易对抗伪影 | 过饱和 / 多样性下降 |
| **产业现状** | 早期 ImageNet SOTA 关键手段 | **主流默认**（SD、Flux 系、多数条件生成） |

关系可以记成：

```
条件扩散:     网络「看见」c，但服从程度固定
      │
      ├─ Classifier Guidance ──► 外部分类器梯度当「方向盘」
      │
      └─ Classifier-Free Guidance ──► 用 (有条件 − 无条件) 当「方向盘」
                                      （等价地近似分类器项，却不训分类器）
```

---

## 5. 和 VLA / Flow Matching 的关系（简要）

- **Diffusion Policy、部分动作扩散头**：可用 CFG 式混合，加强「服从语言/目标」的程度（例如对任务条件做 null dropout）。
- **π0 等 Flow Matching 动作头**：guidance 思想可平移到速度场 \(v_\theta\)——对条件/无条件速度场做同样线性组合；细节与调度因 ODE/FM 路径而异，见 [Appendix A](./Appendix_A_Flow_Matching.md)。
- 选型直觉：**新做条件生成默认 CFG**；只有在已有强分类器、或不想改生成模型训练协议时，才更常考虑经典 Classifier Guidance。

---

## 6. 一句话总结

| 方法 | 要解决的问题 | 解决方法 |
| :--- | :--- | :--- |
| **Classifier Guidance** | 条件采样忠诚度不够，需要可调、显式的条件推力 | 训噪声感知分类器，采样时把 \(\nabla_x\log p(c\mid x_t)\) 加进 score / 噪声预测 |
| **Classifier-Free Guidance** | 仍要可调推力，但不要外部分类器及其伪影与维护成本 | 训练时 dropout 条件，推理时用 \(\epsilon_u + w(\epsilon_c-\epsilon_u)\) 外推 |

---

## 参考

1. Dhariwal & Nichol. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.
2. Ho & Salimans. *Classifier-Free Diffusion Guidance*. 2022.
3. 本仓库：[Appendix A. Flow Matching](./Appendix_A_Flow_Matching.md)、[20. 动作表示](./20_Action_Tokenization.md)
