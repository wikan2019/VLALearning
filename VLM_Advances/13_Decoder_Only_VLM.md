# 13. Architecture: V-Encoder -> Pure Decoder (Fuyu / Molmo)

目前的 VLM 主流架构是 **Dual Tower (双塔)** 结构：一个 Vision Encoder (ViT) 负责看图，一个 LLM 负责推理。但 **Pure Decoder (纯解码器)** 架构正在挑战这一范式。

## 1. 主流范式：Vision Encoder + LLM
*   **各司其职**：ViT（如 CLIP, SigLIP）在大规模图文对上预训练，学习很好的视觉表征；LLM 在文本上预训练。
*   **中间商**：通过 Connector (MLP/Q-Former) 强行把两者对齐。
*   **缺点**：
    *   **分辨率限制**：ViT 通常限制了特定分辨率（如 336）。
    *   **信息瓶颈**：ViT 的输出是高度抽象的 Semantic Embedding，可能丢弃了颜色、位置等底层物理信息。

## 2. 新范式：Pure Decoder (Fuyu Architecture)
**Adept AI 提出的 Fuyu-8B** 和 AI2 提出的 **Molmo** 采用了激进的简化方案：**Remove the Vision Encoder.**

### 核心机制
1.  **直接吃像素**：
    *   将图片切成小 Patch（例如 $14 \times 14$）。
    *   直接将 Patch 中的像素值展平 (Flatten) 成一个向量。
    *   通过一个简单的 Linear Projection 层将像素向量映射到 LLM 的 Embedding 维度。
2.  **像读文本一样读图**：
    *   将图片的 Patch 序列直接视为一种“外语”词汇，和 Text Tokens 混在一起输入给 Transformer Decoder。
    *   模型在训练时，通过 Next Token Prediction 既学习语言，也学习图像模式。

## 3. 优势
1.  **无限分辨率 (Arbitrary Resolution)**：
    *   因为没有 ViT 的 Positional Embedding 限制，理论上可以输入任意大小、任意比例的图片。不需要 Resize，不需要 Padding。
2.  **底层信息保留**：
    *   模型直接接触 Raw Pixels，对于主要依赖几何形状和细节的任务（如 UI 理解、图表分析、OCR）表现极佳。
3.  **架构极简**：
    *   整个模型就一个 Transformer，部署维护极其方便。

## 4. 劣势
*   **训练成本高**：因为没有现成的强力 Vision Encoder (CLiP) 可以白嫖，模型必须从头开始学习“如何看图”。这需要海量的高质量图文数据和算力。
*   **语义理解稍弱**：在纯粹的语义识别（如“这是一只澳洲牧羊犬还是一只边境牧羊犬”）上，可能不如在几十亿图文对上训练过的 CLIP 强。

## 5. 总结

| 特性 | Encoder-Decoder (LLaVA) | Pure Decoder (Fuyu) |
| :--- | :--- | :--- |
| **视觉处理** | Pre-trained ViT | **Linear Projection of Pixels** |
| **分辨率** | 受限 (需 AnyRes 补丁) | **原生支持任意分辨率** |
| **训练效率** | 快 (站在 CLIP 肩膀上) | 慢 (从零学视觉) |
| **擅长领域** | 通用物体识别、描述 | **以及 OCR, UI, 图表, 试卷** |
| **代表作** | GPT-4V, Gemini, LLaVA | Fuyu, Molmo, Nougat |

**趋势**：虽然 Encoder-Decoder 目前仍是主流，但随着多模态原生预训练（Native Multimodal Pretraining）的兴起，Pure Decoder 正在变得越来越重要（如 GPT-4o 的原生音频视频能力可能部分借鉴了此思路）。
