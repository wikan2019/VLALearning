# VLM 技术进展学习文档索引

> 🎨 Vision-Language Models (VLM) 截止 2026 年 2 月的 16 项关键技术演进

[⬅️ 返回主目录](../README.md) | [📖 LLM 基础](../LLM_Advances/README.md)

---

## 📑 目录概览

本文档汇总了视觉语言模型领域的核心技术演进，分为两大部分：

- **Part 3**: 多模态架构与训练 (文档 11-16) - 核心主线
- **补充专题**: 视觉编码、接地、视频、幻觉、统一模型 (S1-S5) - 重要扩展

---

## Part 3: 多模态架构与训练 (VLM Architecture & Training)

### 11. [Connector: Q-Former → MLP](./11_Connector.md)
*   **演进点**：模态连接器的简化之路
*   **核心价值**：从复杂的 Query Transformer 回归到简单的 **MLP Projection**，证明了在 LLM 足够强的情况下，保留原始视觉特征比压缩特征更好
*   **代表模型**：LLaVA, GPT-4V
*   **关联阅读**：→ [S1. 视觉编码器演进](./VLM_S1_Vision_Encoder_Evolution.md)（Connector 的输入）

### 12. [Resolution: Fixed Res → AnyRes](./12_Resolution_AnyRes.md)
*   **演进点**：从固定分辨率 (Resize) 到动态高分辨率切片 (Dynamic Tiling)
*   **核心价值**：通过 **AnyRes (Crop & Fuse)** 策略，让 VLM 能够看清高清图片中的细粒度物体、OCR 文字和图表，打破了 ViT 的分辨率限制
*   **代表模型**：LLaVA-NeXT, InternVL
*   **关联阅读**：→ [13. 纯解码器架构](./13_Decoder_Only_VLM.md)（另一种分辨率解决方案）

### 13. [Architecture: V-Encoder → Pure Decoder](./13_Decoder_Only_VLM.md)
*   **演进点**：从双塔结构 (ViT + LLM) 到纯解码器架构 (Pure Decoder)
*   **核心价值**：移除 Vision Encoder，直接将像素片段 (Patch) 线性投影进入 LLM。支持原生任意分辨率，架构极简，利于端侧部署和文档理解
*   **代表模型**：Fuyu-8B, Molmo, Nougat
*   **关联阅读**：→ [12. AnyRes](./12_Resolution_AnyRes.md)（对比两种分辨率方案）

### 14. [Training: Img-Txt Align → Visual Instruction Tuning](./14_Visual_Instruction_Tuning.md)
*   **演进点**：从对齐预训练 (CLIP) 到视觉指令微调 (Visual SFT)
*   **核心价值**：利用纯文本 GPT-4 生成多模态对话数据，将 VLM 从"看图说话"进化为"听懂指令"的 **Visual Chatbot**。含 RLHF 详解
*   **代表模型**：LLaVA, InstructBLIP
*   **关联阅读**：→ [LLM/10. DPO](../LLM_Advances/10_DPO.md)（对齐方法的基础）

### 15. [Reasoning: VQA → Visual CoT](./15_Visual_CoT.md)
*   **演进点**：从直接回答 (VQA) 到视觉思维链 (Visual Chain-of-Thought)
*   **核心价值**：强制模型输出 `观察 -> 分析 -> 结论` 的推理过程，显著提升了在几何、逻辑、科学问题上的准确率，减少了通过 Shortcut 猜答案的现象
*   **代表模型**：LLaVA-1.5, GPT-4o, Gemini 1.5
*   **关联阅读**：→ [S4. 多模态幻觉](./VLM_S4_Hallucination.md)（推理错误的根源之一）

### 16. [SOTA Models: Qwen3 vs Gemini 3 vs GPT-5.2](./16_SOTA_Models_Review.md)
*   **演进点**：当前最强 VLM 模型架构深度解析 + 后续模型全面对比
*   **核心价值**：
    *   **Qwen2-VL → Qwen3**: 动态分辨率 + 开源全模态 + 音频 SOTA
    *   **Gemini 1.5 → 3 Pro**: 1M 上下文 + 帧级视频推理，感知之王
    *   **GPT-4o → 5.2**: 端到端语音 + 自验证机制，推理之王
*   **代表模型**：Qwen3, Gemini 3 Pro, GPT-5.2
*   **关联阅读**：→ [S3. 视频理解](./VLM_S3_Video_Understanding.md)（长视频处理技术）

---

## 补充专题 (Supplementary Topics)

以下 5 篇文档补充了 VLM 进展中同样重要但未被上述主线覆盖的关键技术：

### S1. [视觉编码器演进：CLIP → SigLIP → DINOv2 → SigLIP 2](./VLM_S1_Vision_Encoder_Evolution.md)
*   **演进点**：VLM 的"眼睛"从单一对比学习走向多目标联合训练
*   **核心价值**：CLIP 提供语义，DINOv2 提供空间感知，**双编码器融合**成为 VLA 标配。SigLIP 2 (400M) 以多目标训练打败 InternViT (6B)，证明**训练方法 > 模型规模**
*   **代表模型**：CLIP, SigLIP 2, DINOv2, Prismatic VLMs
*   **关联阅读**：→ [11. Connector](./11_Connector.md)（编码器输出如何连接到 LLM）

### S2. [视觉接地：从"看图说话"到"指哪说哪"](./VLM_S2_Visual_Grounding.md)
*   **演进点**：VLM 从纯文本输出 → 输出空间坐标（边界框/点坐标）
*   **核心价值**：Grounding 是 **VLM → VLA 的桥梁**——机器人操作、UI 自动化、医学影像都需要精确定位
*   **代表模型**：Kosmos-2, Ferret, CogAgent, Set-of-Mark
*   **关联阅读**：→ [VLA/19. VLA 架构](../VLA_Advances/19_VLA_Architecture.md)（Grounding 在机器人中的应用）

### S3. [视频理解：从单张图片到时间流](./VLM_S3_Video_Understanding.md)
*   **演进点**：从单图理解 → 多帧分析 → 数小时长视频理解
*   **核心价值**：帧采样、Token 压缩、M-RoPE 时间位置编码让 VLM 理解动态世界。**视频理解是 VLM 静态感知到 VLA 动态行动的关键桥梁**
*   **代表模型**：Video-LLaVA, Qwen2-VL, Gemini 3 Pro
*   **关联阅读**：→ [16. SOTA 模型](./16_SOTA_Models_Review.md), → [VLA/24. 世界模型](../VLA_Advances/24_World_Models_Planning.md)

### S4. [多模态幻觉：VLM 最致命的缺陷](./VLM_S4_Hallucination.md)
*   **演进点**：识别和对抗 VLM 特有的"编造不存在内容"问题
*   **核心价值**：对比解码 (VCD/ICD/LCD) 提供**无需重训的即插即用**方案；RLHF/DPO 从训练层面根治。在 VLA 中，幻觉从"说错话"升级为"做错事"
*   **评测基准**：POPE, CHAIR, MMHal-Bench
*   **关联阅读**：→ [14. Visual Instruction Tuning](./14_Visual_Instruction_Tuning.md)（RLHF 对抗幻觉）

### S5. [统一视觉模型：从"只看"到"又看又画"](./VLM_S5_Unified_Vision.md)
*   **演进点**：VLM 从纯理解 → 理解+生成的统一模型
*   **核心价值**：通过视觉 Tokenization (VQ-VAE) 将图像纳入自回归框架，**一个模型同时理解和生成图像**。为 VLA 的世界模型和视觉目标生成奠定基础
*   **代表模型**：Chameleon, Emu2, Show-o, Janus
*   **关联阅读**：→ [VLA/24. 世界模型](../VLA_Advances/24_World_Models_Planning.md)（视频生成在规划中的应用）

---

## 🔗 技术关联图

```
视觉输入层:
S1. Vision Encoder (CLIP/DINOv2) 
   ↓
11. Connector (MLP)
   ↓
12. AnyRes / 13. Pure Decoder (分辨率处理)

训练与推理层:
14. Visual Instruction Tuning (SFT + RLHF)
   ↓
15. Visual CoT (推理能力)
   ↓
S4. 对抗幻觉

扩展能力层:
S2. Visual Grounding (空间定位) → VLA 动作输出
S3. Video Understanding (时间理解) → VLA 动态感知
S5. Unified Vision (生成能力) → 世界模型
```

---

## 📚 学习建议

### 🎯 新手路径
1. **基础理解**：先学习 [LLM 基础](../LLM_Advances/README.md)
2. **视觉基础**：S1 → 11 → 12（理解 VLM 如何"看"）
3. **对话能力**：14 → 15（理解 VLM 如何"说"）
4. **前沿模型**：16（了解当前 SOTA）

### 🚀 进阶路径
1. **架构深度**：11 → 12 → 13（对比三种架构方案）
2. **训练方法**：14 → S4（从训练到对抗幻觉）
3. **能力扩展**：S2 → S3 → S5（空间、时间、生成）

### 💡 实践路径
- **部署 VLM**：13 → 12 → 11（选择合适的架构）
- **提升性能**：S1 → 14 → S4（从编码器到对齐）
- **特定场景**：
  - 机器人控制 → S2（Grounding）
  - 视频分析 → S3（Video Understanding）
  - 内容生成 → S5（Unified Vision）

---

## 🔄 与其他技术栈的关系

- **← LLM 技术栈**：VLM 基于 LLM 构建，继承了所有 LLM 的优化技术
  - [返回 LLM 文档目录](../LLM_Advances/README.md)

- **→ VLA 技术栈**：VLA 在 VLM 基础上增加动作输出，实现具身智能
  - [进入 VLA 文档目录](../VLA_Advances/README.md)

---

## 📝 文档说明

- **核心主线（11-16）**：VLM 架构演进的必读内容
- **补充专题（S1-S5）**：深入理解特定技术方向
- 所有文档包含：原理解析、数学公式、代码示例、应用案例
- 标注了跨文档的交叉引用，便于串联学习

---

[⬅️ 返回主目录](../README.md) | [📖 LLM 基础](../LLM_Advances/README.md) | [➡️ 前往 VLA 文档](../VLA_Advances/README.md)
