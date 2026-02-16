# VLM 技术进展学习文档索引 (Index of VLM Advances)

本文档汇总了截止 2026 年 2 月 Vision-Language Models (VLM) 领域的 5 项关键技术演进。

---

## Part 3: 多模态架构与训练 (VLM Architecture & Training)

### 11. [Connector: Q-Former → MLP](./11_Connector.md)
*   **演进点**：模态连接器的简化之路。
*   **核心价值**：从复杂的 Query Transformer 回归到简单的 **MLP Projection**，证明了在 LLM 足够强的情况下，保留原始视觉特征比压缩特征更好。
*   **代表模型**：LLaVA, GPT-4V.

### 12. [Resolution: Fixed Res → AnyRes](./12_Resolution_AnyRes.md)
*   **演进点**：从固定分辨率 (Resize) 到 动态高分辨率切片 (Dynamic Tiling)。
*   **核心价值**：通过 **AnyRes (Crop & Fuse)** 策略，让 VLM 能够看清高清图片中的细粒度物体、OCR 文字和图表，打破了 ViT 的分辨率限制。
*   **代表模型**：LLaVA-NeXT, InternVL.

### 13. [Architecture: V-Encoder → Pure Decoder](./13_Decoder_Only_VLM.md)
*   **演进点**：从双塔结构 (ViT + LLM) 到 纯解码器架构 (Pure Decoder)。
*   **核心价值**：移除 Vision Encoder，直接将像素片段 (Patch) 线性投影进入 LLM。支持原生任意分辨率，架构极简，利于端侧部署和文档理解。
*   **代表模型**：Fuyu-8B, Molmo, Nougat.

### 14. [Training: Img-Txt Align → Visual Instruction Tuning](./14_Visual_Instruction_Tuning.md)
*   **演进点**：从对齐预训练 (CLIP) 到 视觉指令微调 (Visual SFT)。
*   **核心价值**：利用纯文本 GPT-4 生成多模态对话数据，将 VLM 从“看图说话”进化为“听懂指令”的 **Visual Chatbot**。
*   **代表模型**：LLaVA, InstructBLIP.

### 15. [Reasoning: VQA → Visual CoT](./15_Visual_CoT.md)
*   **演进点**：从直接回答 (VQA) 到 视觉思维链 (Visual Chain-of-Thought)。
*   **核心价值**：强制模型输出 `观察 -> 分析 -> 结论` 的推理过程，显著提升了在几何、逻辑、科学问题上的准确率，减少了通过 Shortcut 猜答案的现象。
*   **代表模型**：LLaVA-1.5, GPT-4o, Gemini 1.5.

### 16. [SOTA Models: Qwen2-VL vs Gemini 1.5 vs GPT-4o](./16_SOTA_Models_Review.md)
*   **演进点**：当前最强 VLM 模型架构深度解析。
*   **核心价值**：
    *   **Qwen2-VL**: Naive Dynamic Resolution (M-RoPE) 实现极致 OCR。
    *   **Gemini 1.5**: MoE + Native Multimodality 实现百万级长视频理解。
    *   **GPT-4o**: Unified Omni Model 实现低延迟实时语音交互。
*   **代表模型**：Qwen2-VL, Gemini 1.5 Pro, GPT-4o.
