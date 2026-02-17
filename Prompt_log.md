## 2026-02-16

# LLM 技术进展学习文档

### 1. 我要学习大语言模型最近5年的进展。主要有以下要点, 请为每个要点写一篇 详细的MarkDown 文档方便我来学习
1. layerNorm -> RMSNorm
2. APE -> RoPE
3. FFN->Gated FFN
4. Attention -> QK-Norm + 稳定性设计
5. 引入 Register/Memory Token


请为我解释为什么 RMS Norm 其数值稳定性在深层网络中表现也十分优异，有助于缓解梯度消失/爆炸问题？ 并且RMS Norm 跟 PreNorm 和 PostNorm 有什么关系？ 把这一部分的内容也放入第一篇LLM 技术进展文档里

请为我详细解释 第三篇 Gated FFN & SwiGLU 中 SwiGLU 的计算公公式每一项的详细含义与实现方法，并为我逐点详细介绍 SwiGLU为什么有效？提供更丰富的讲解和案例。把这一部分的内容也补充进入第三篇LLM 技术进展文档里

请解释Memory Tokens (LLMs) 的具体实现方式，补充Recurrent Memory Transformer (RMT)： 和 StreamLLM (Attention Sinks) 的具体实现方式，并列举出应用实例。 把这一部分的内容也补充进入第五篇LLM 技术进展文档里

请解释详细第五篇技术文档里： LLM 在流式多轮对话中，显存（KV Cache）会爆满，但简单的“滑动窗口”会导致模型崩盘 的原因。以及解释 直接丢弃最早的 token（滑动窗口），LLM 的 Perplexity 会瞬间暴涨的原因。 在StreamLLM  里 的Rolling Cache 方法里，当新 token 进来时，丢弃 Rolling Cache 中最旧的，但在计算 Attention 时，位置编码（Positional Encoding）需要进行特殊处理（例如相对位置不变），请详细解释。 把这一部分的内容也补充进入第五篇LLM 技术进展文档里


出了这里的五篇LLM技术进展之外，截止到今天还有哪些LLM重要进展，请列出请补充在一个MarkDown 文档里面， 并对每一个要点，类似这五篇LLM技术文档，生成一个MarkDown 文档详细介绍。

请详细介绍第8篇文章中的技术。 为什么 FlashAttention 不是近似计算，他是怎么加速计算的，为什么他减少了显存开支，使context 上下文推理成为了可能。把这一部分内容也补充进入第8篇 LLM技术进展文档里面。

第8章技术文档里，softmax([x 
1
​
 ,x 
2
​
 ])=α⋅softmax(x 
1
​
 )+(1−α)⋅softmax(x 
2
​
 )

 里的 α 是怎么来的？请补充计算公式。，为了节省显存，FlashAttention 不存储 
N
×
N
N×N 的 Attention Map。 在反向传播时，它利用保留的 Output 和 LogSumExp 统计量，重新快速计算一遍 Attention Score。  这里的反向传播梯度计算是怎么实现的？请补充数学公式。如果flash attention并不损失计算精度，那这种操作应该已经成为业界的标准做法，能否举出反例，说明flash attention 无法适用的地方。 请详细解释。 把这一部分的内容也补充进入第8篇LLM 技术进展文档里


请详细解释 第9章， 3. 挑战与解决方案中。 这些挑战都是怎么解决的。请详细介绍第二段关键组件里， 路由网络是怎么做的， 怎么来选择 专家的， 如果是通过softmax来选专家，是所有专家都会选上吗？ 只是每个专家对应的权重不同。把这一部分内容也补充进入第9篇 LLM技术进展文档里面。


请详细介绍第10章。 DPO 与sft 的区别在哪里？ 为什么DPO 比 PPO 稳定？为什么DPO不需要reward model？DPO 的效果怎么样？ 在最近的chat-gpt 5.2 和 cluade code 4.6 中有用到DPO， 如果没有用到，那为什么不用？ 把这一部分内容也补充进入第10篇 LLM技术进展文档里面。


为LLM Advances 的技术文档生成一篇索引目录的MarkDown文档。


# VLM 技术学习文档

类似LLM 技术学习文档，请列举出 截止到2026年 2月份 VLM 技术进展的要点， 为每个要点写一篇 详细的MarkDown 文档方便我来学习，放在另一个子文件夹里面

第11章， connector 从 简单到复杂再到简单的发展原因是什么？ 为什么一开始这种简单的架构没有发挥出优势？ 请介绍现在的主要VLM模型比如qwen， gemini， gpt-5v 等等， 它们的 connector 是怎么设计的？ 把这一部分的内容也补充进入第11篇VLM 技术学习文档里

请再补充第16篇， 介绍 现在的主要VLM模型比如qwen， gemini， gpt-5v 的模型结构，并做详细解析，解析每个模型的优缺点。

12 篇讲到的 AnyRes (Dynamic Tiling) 把一个大图片切为多个固定尺寸的小图片，这多个小图片，是经过同一个 visual encoder吗？ 请把这一部分内容也补充进去第12篇。


关于第14篇， 现在的VLM 模型中， 主要在sft 阶段 使VLM 模型具备，看图聊天的能力吗？ 其他阶段的训练，问什么不训练 看图聊天的能力？请补充在第14 篇里 

请介绍第14篇，里第3部分，介绍了RLHF, 请介绍RLHF 的主要做作用是什么？请补充在第14 篇里 

在第16篇里，请告诉我，gemini 的后续模型 gemini 3 以及 chat gpt 的后续模型 gpt-5.2 是否仍然具备，图像，视频，语音，文字的多模态能力，比较gemini 3 和 gpt 5.2 的优缺点， 请补充在第16篇里面。

在第16篇里，在多模态对比表格里，请加上， qwen3 的对比列

请review 下 VLM 进展相关文档，检查是否有重要进展没有类出来的，请另外补充markdown 文档


# VLA 技术学习文档

类似LLM 和VLM 技术学习文档，请列举出 截止到2026年 2月份 VLA 技术进展的要点， 为每个要点写一篇 详细的MarkDown 文档方便我来学习，放在另一个子文件夹里面