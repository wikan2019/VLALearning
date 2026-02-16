## 2026-02-16

### 1. 我要学习大语言模型最近5年的进展。主要有以下要点, 请为每个要点写一篇 详细的MarkDown 文档方便我来学习
1. layerNorm -> RMSNorm
2. APE -> RoPE
3. FFN->Gated FFN
4. Attention -> QK-Norm + 稳定性设计
5. 引入 Register/Memory Token


请为我解释为什么 RMS Norm 其数值稳定性在深层网络中表现也十分优异，有助于缓解梯度消失/爆炸问题？ 并且RMS Norm 跟 PreNorm 和 PostNorm 有什么关系？ 把这一部分的内容也放入第一篇LLM 技术进展文档里

请为我详细解释 第三篇 Gated FFN & SwiGLU 中 SwiGLU 的计算公公式每一项的详细含义与实现方法，并为我逐点详细介绍 SwiGLU为什么有效？提供更丰富的讲解和案例。把这一部分的内容也补充进入第三篇LLM 技术进展文档里
