# 14. Training: Img-Txt Align -> Visual Instruction Tuning

VLM 的训练范式经历了从“对齐”到“指令微调”的巨大转变。这一转变让 VLM 从一个只会分类或简短描述的工具，变成了真正的**多模态聊天机器人 (Visual Chatbot)**。

## 1. 阶段一：图文对齐 (Image-Text Alignment)
以 **CLIP (OpenAI)** 为代表。
*   **数据**：海量的 `(Image, Text)` 对（如 LAION-400M）。
*   **目标**：**对比学习 (Contrastive Learning)**。拉近匹配的图文对的特征距离，推远不匹配的。
*   **能力**：极强的 Zero-Shot 分类和检索能力。
*   **局限**：**无法生成连续文本**。它只能告诉你图片和那句话“像不像”，不能回答“图片里有什么”。

## 2. 阶段二：生成式预训练 (Captioning)
以 **BLIP / SimVLM** 为代表。
*   **数据**：还是图文对，但质量更高（如 COCO Captions）。
*   **目标**：**Language Modeling (LM)**。给定图片，预测对应的文本描述。
*   **能力**：Image Captioning（看图说话）、VQA（简短问答）。
*   **局限**：**只会陈述事实，不懂指令**。如果你问它“这张图片好笑在哪里？”，它可能只会回答“一只猫在跳舞”，而无法解释笑点。

## 3. 阶段三：视觉指令微调 (Visual Instruction Tuning)
**LLaVA (Large Language and Vision Assistant)** 的出现定义了现代 VLM 的训练范式。
核心思想：**将视觉任务转化为多轮对话格式**。

### (a) 数据构造 (The Magic of GPT-4)
LLaVA 没有大量人工标注的对话数据。它利用纯文本的 GPT-4：
1.  输入图片的详细描述（Captions）和物体边框（Bounding Boxes）。
2.  让 GPT-4 **脑补**出一段关于这张图的对话。
    *   *User*: "这张图里有什么异常吗？"
    *   *Assistant*: "异常的是这只猫像人一样站着..."
3.  这就生成了 158k 条高质量的**多模态对话数据**。

### (b) 训练过程 (SFT)
将 VLM 在这些生成的数据上进行 **Supervised Fine-Tuning (SFT)**。
*   **输入**：`<Image> User: ... \n Assistant:`
*   **输出**：预测 Assistant 的回答。

**结果**：模型不仅学会了看图，更学会了**遵从人类指令 (Follow Instructions)**，能够进行复杂的推理、解释代码、编写故事。

## 4. 阶段四：VLM 的 RLHF
最新的研究（如 LLaVA-RLHF, Silkie）开始将 RLHF / DPO 引入 VLM。
*   **目标**：减少幻觉（Hallucination），比如模型不应该描述图片里不存在的物体。通过人类反馈，让模型更诚实 (Honest) 和有帮助 (Helpful)。

## 5. 总结

| 特性 | CLIP (对齐) | LLaVA (指令微调) |
| :--- | :--- | :--- |
| **核心逻辑** | `Match(Img, Txt)` | `Chat(Img, User_Instruction)` |
| **数据来源** | 爬虫 (LAION) | GPT-4 生成 / 人工改写 |
| **能力边界** | 分类, 检索 | **复杂对话, 推理, 代码** |
| **训练目标** | 对比损失 (InfoNCE) | 交叉熵 (Next Token Prediction) |
| **地位** | 视觉编码器的基石 | **多模态助手的基石** |
