# Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models

> **论文信息**
> - **标题**: Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models
> - **中文译名**: 定位、引导与改进：大语言模型中可落地的机制可解释性实用综述
> - **作者**: Hengyuan Zhang*, Zhihao Zhang*, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Feijiang Han, Qibo Xue, Zeping Yu, Chenming Shang, Xiao Liang, Jing Xiong, Hui Shen, Chaofan Tao, Zhengwu Liu, Senjie Jin, Zhiheng Xi, Dongdong Zhang, Sophia Ananiadou, Tao Gui, Ruobing Xie, Hayden Kwok-Hay So, Hinrich Schütze, Xuanjing Huang, Qi Zhang†, Ngai Wong†
> - **机构**: 香港大学、复旦大学、LMU Munich、清华大学、达姆施塔特工业大学、柏林工业大学、德累斯顿工业大学、香港中文大学、宾夕法尼亚大学、南京大学、曼彻斯特大学、达特茅斯学院、UCLA、密歇根大学、Microsoft、Tencent 等
> - **发表**: arXiv:2601.14004，2026 年 4 月 13 日（持续更新综述）
> - **关键词**: Actionable Interpretability、LLM、Localizing / Steering、Model Improvement、Mechanistic Interpretability

---

## 论文链接

| 资源 | 链接 |
|------|------|
| **arXiv 摘要页** | https://arxiv.org/abs/2601.14004 |
| **PDF** | https://arxiv.org/pdf/2601.14004 |
| **GitHub（持续更新书单）** | https://github.com/rattlesnakey/Awesome-Actionable-MI-Survey |

联系邮箱（原文）：hengyuan.zhang88@gmail.com、zhangzhihao19@fudan.edu.cn、qz@fudan.edu.cn、nwong@eee.hku.hk

---

## 描述（Abstract）

**机制可解释性（Mechanistic Interpretability, MI）**已成为揭开大语言模型（LLM）决策黑箱的重要路径。然而，现有综述大多将 MI 视为一种**观察性科学**：侧重总结分析洞见，却缺少一套可落地干预的系统框架。为填补这一空白，我们提出一份围绕 **「定位—引导—改进」（Locate, Steer, and Improve）** 流水线组织的实用综述。我们依据具体的**可解释对象（Interpretable Objects）**，对**定位（诊断）**与**引导（干预）**方法进行形式化分类，从而建立一套严谨的干预协议。进一步地，我们说明该框架如何在**对齐、能力与效率**上带来切实提升，使 MI 真正成为可用于模型优化的可操作方法。鉴于可落地机制可解释性发展迅速，我们将持续更新本综述，以反映该领域的最新进展。

原文：

> Mechanistic Interpretability (MI) has emerged as a vital approach to demystify the opaque decision-making of Large Language Models (LLMs). However, existing reviews primarily treat MI as an observational science, summarizing analytical insights while lacking a systematic framework for actionable intervention. To bridge this gap, we present a practical survey structured around the pipeline: “Locate, Steer, and Improve.” We formally categorize Localizing (diagnosis) and Steering (intervention) methods based on specific Interpretable Objects to establish a rigorous intervention protocol. Furthermore, we demonstrate how this framework enables tangible improvements in Alignment, Capability, and Efficiency, effectively operationalizing MI as an actionable methodology for model optimization. With actionable mechanistic interpretability evolving at a fast pace, we pledge to keep this survey up to date, ensuring it reflects the cutting-edge advances in this area.

---

## 简介（Introduction）

大语言模型在复杂推理、多语言与专业领域上已取得显著成功，但其内部决策过程仍高度不透明，常被视作「黑箱」。缺乏可解释性会带来安全风险，也限制了我们高效调试、控制和优化模型行为的能力。因此，**机制可解释性（MI）**成为关键方向：它不同于传统行为分析，目标是**反向工程**神经网络，把复杂计算拆解成可理解的组件与因果机制。

现有工作大致分成两类，但都还不够「能动手改模型」：

| 路线 | 侧重点 | 缺口 |
|------|--------|------|
| **理论 / 基础 MI** | 解剖 Transformer、找出基本单元，追求科学理解 | 把微观洞见翻译成实际模型改进的路径探索不足，MI 仍是观察性科学 |
| **应用向综述** | 讨论 MI 如何服务下游任务或特定领域 | 诊断工具与干预技术边界模糊；应用覆盖不全、方法叙述偏抽象，难落到具体问题上 |

作者进一步指出：LLM 的快速进步主要靠**外部 scaling**（更大模型、更多数据、更多算力），这条路正遇到瓶颈。后续提升需要转向对内部机制的更深理解——从「改外面」转向「从内部外科式改进」。目前仍缺少一份统一指南，系统分类这些内部方法，并给出**主动改进模型**的具体流水线。

为此，本文提出 **「Locate, Steer, and Improve」** 流水线，把 MI 从被动观察变成可操作的干预学科。三项主要贡献：

1. **流水线驱动的严谨框架**  
   先定义 LLM 内的核心**可解释对象**（如神经元、注意力头、残差流）。再按工作流把方法分成两阶段：  
   - **Localizing（定位 / 诊断）**：找出造成特定行为的因果组件  
   - **Steering（引导 / 干预）**：主动操纵这些组件以改变输出  
   对每种技术给出方法形式化、适用对象与范围。

2. **面向应用的完整范式**  
   围绕三大主题组织 MI 应用：**改进对齐（Alignment）、改进能力（Capability）、改进效率（Efficiency）**，覆盖安全、多语言、高效训练等八类场景；不只罗列论文，还总结各场景的代表性用法，便于迁移到新问题。

3. **洞见、资源与未来方向**  
   讨论可落地 MI 的挑战与前景，并整理 **200+** 篇论文（原文 Table 2），按对应的定位 / 引导方法打标签，作为社区可检索的参考。仓库会持续更新。

全文结构：

```
§2  定义核心可解释对象（Interpretable Objects）
        ↓
§3  Locate：定位 / 诊断方法
        ↓
§4  Steer：引导 / 干预方法
        ↓
§5  Improve：对齐 · 能力 · 效率上的切实改进
```
