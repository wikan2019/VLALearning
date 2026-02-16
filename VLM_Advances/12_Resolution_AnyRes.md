# 12. Resolution: Fixed Res -> AnyRes (Dynamic Tiling)

分辨率 (Resolution) 是限制 VLM 识别细粒度物体（如小文字、密集图表）的最大瓶颈。传统的 Vision Encoder（如 CLIP-ViT）通常在大规模低分辨率数据集（如 224x224 或 336x336）上预训练，导致直接拉伸高分辨率图片会严重降低性能。

## 1. 传统方案：直接缩放 (Resize / Pad)
大多数早期 VLM（如 LLaVA v1）的做法非常简单粗暴：
不管输入图片是 1080p 还是 4K，一律 **Resize** 到 336x336。
*   **后果**：
    *   **细节丢失**：原本清晰的文档文字变成马赛克，OCR 能力归零。
    *   **长宽比失真**：扁长的图片被强行压成正方形，物体变形。

## 2. 进阶方案：AnyRes / Dynamic Tiling (LLaVA-NeXT, InternVL)
为了解决这个问题，业界提出了 **AnyRes (Any Resolution)** 策略，核心思想是 **“切片”(Crop & Fuse)**。

### 核心流程
假设输入是一张 $1000 \times 600$ 的高分图，模型支持的基准分辨率是 $336 \times 336$。

1.  **动态网格划分 (Dynamic Grid)**：
    *   根据图片的长宽比，选择最合适的网格配置（例如 $2 \times 3$ 或 $3 \times 1$）。
    *   将大图切成 $N$ 个 $336 \times 336$ 的局部小图 (Local Patches)。

2.  **全局缩略图 (Global Thumbnail)**：
    *   将原图 Resize 到 $336 \times 336$，作为能够看到整体全貌的 Global Image。
    *   **Global 提供宏观语义，Local 提供微观细节。**

3.  **Token 拼接**：
    *   将 Global Tokens 和所有 Local Tokens 拼接成一个长序列喂给 LLM。
    *   例如：`[Global Tokens] [Separator] [Local-1 Tokens] [local-2 Tokens] ...`
    *   为了让 LLM 知道这些 Token 的空间关系，通常会插入特殊的 `\n` 或坐标 Token。

### 2.1 关键问题：它们经过同一个 Vision Encoder 吗？
**答案是肯定的：YES。**

*   **共享权重 (Shared Weights)**：无论是整张图的缩略图 (Global Thumbnail)，还是切分出来的局部图 (Local Patches)，它们都经过**同一个、共享参数的 Vision Encoder** (例如 CLIP-ViT-L)。
*   **Batch 处理**：在代码实现上，这些切片通常会被堆叠成一个 Batch (Batch Size = $N+1$)，一次性并行送入 ViT 提取特征。
*   **为什么这样做？**
    *   **复用预训练知识**：ViT 在预训练时就是在这个分辨率（如 336）上学习的。如果我们强行搞一个“大分辨率 ViT”，就需要从头训练，成本太高。将大图切成小图，让 ViT 觉得自己在看很多张经过预训练的小图，就能直接复用强大的 CLIP 权重。
    *   **位置编码插值**：虽然走的同一个 Encoder，但 LLM 需要知道 Patch 之间的相对位置。通常会在 Connector 阶段或进入 LLM 之前，给不同位置的 Tile 添加特殊的 **分隔符 (Separators)** 或 **2D 位置编码**。

## 3. 典型代表
*   **LLaVA-NeXT (LLaVA-1.6)**: 支持任意分辨率切片，极大地提升了 OCR 和图表理解能力。
*   **InternVL 1.5**: 采用了类似的策略，最高支持 4K+ 分辨率输入，性能逼近 GPT-4V。

## 4. 优势与代价
*   **优势**：
    *   **零样本提升**：不需要重新训练 Vision Encoder，就能处理高清图。
    *   **OCR 质变**：从小图看不请到能读几千字的小说截图。
*   **代价**：
    *   **Context 爆炸**：一张 1080p 图片切成 6 块，Token 数量直接翻 7 倍（1 Global + 6 Local）。这显著增加了推理的显存占用和延迟。
    *   所以，**FlashAttention** 等长文本优化技术对于高清 VLM 也是必不可少的。

## 5. 总结
**AnyRes (Dynamic Tiling)** 是 VLM 迈向实用的关键一步，它解决了“看不清”的问题，让 VLM 真正具备了阅读文档和分析复杂场景的能力。
