# RL on Flow Matching and Diffusion

本目录整理三篇把强化学习接到扩散 / Flow Matching 生成模型上的论文：从扩散上的 online 政策梯度（DDPO），到视频 Flow 的 offline 对齐（VideoAlign / Flow-DPO），再到图像 Flow 的 online GRPO（Flow-GRPO）。

## 论文链接

| 顺序 | 论文 | arXiv | 代码 / 主页 | 笔记 |
|------|------|-------|-------------|------|
| 1 | Training Diffusion Models with Reinforcement Learning (**DDPO**) | [2305.13301](https://arxiv.org/abs/2305.13301) · [pdf](https://arxiv.org/pdf/2305.13301) | [主页](http://rl-diffusion.github.io) | [DDPO_Training_Diffusion_Models_with_RL.md](DDPO_Training_Diffusion_Models_with_RL.md) |
| 2 | Improving Video Generation with Human Feedback (**VideoAlign**) | [2501.13918](https://arxiv.org/abs/2501.13918) · [pdf](https://arxiv.org/pdf/2501.13918) | [主页](https://gongyeliu.github.io/videoalign/) | [VideoAlign_Improving_Video_Generation_with_Human_Feedback.md](VideoAlign_Improving_Video_Generation_with_Human_Feedback.md) |
| 3 | Flow-GRPO: Training Flow Matching Models via Online RL | [2505.05470](https://arxiv.org/abs/2505.05470) · [pdf](https://arxiv.org/pdf/2505.05470) | [代码](https://github.com/yifan123/flow_grpo) | [Flow_GRPO_Training_Flow_Matching_via_Online_RL.md](Flow_GRPO_Training_Flow_Matching_via_Online_RL.md) |

建议阅读顺序与上表一致：先看 DDPO 如何把去噪写成 MDP，再看 VideoAlign 在 Flow 上做 offline DPO / RWR，最后看 Flow-GRPO 如何把 ODE 改成 SDE 才能跑 online GRPO。

## 三篇如何串起来

```
似然 / 数据匹配                    下游目标（审美、对齐、可验证任务）
        │                                      ▲
        │  预训练：DDPM / Flow Matching         │
        ▼                                      │
   去噪轨迹 {x_T → ... → x_0}  ───────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   扩散（高斯逐步转移）      Flow / Rectified Flow（默认 ODE，确定性）
        │                       │
     DDPO                    需要先解决「没有逐步密度」
   online PG                    │
   π = N(μ_θ, σ_t² I)     ┌─────┴──────┐
                          │            │
                     VideoAlign    Flow-GRPO
                     offline       online
                     Flow-DPO      ODE→SDE + GRPO
                     Flow-RWR      Denoising Reduction
                     Flow-NRG
```

| 维度 | DDPO (2023) | VideoAlign (2025.01) | Flow-GRPO (2025.05) |
|------|-------------|----------------------|---------------------|
| 生成范式 | 扩散（DDPM 式高斯反向） | Rectified Flow 视频 | Rectified Flow 图像 |
| RL 类型 | **Online on-policy** 政策梯度 | **Offline** DPO / RWR + 推理期引导 | **Online on-policy** GRPO |
| 策略密度从哪来 | 反向过程本就是 \(\mathcal{N}(\mu_\theta,\sigma_t^2 I)\) | 不需要逐步 \(\log\pi\)（DPO 在速度场上做偏好） | 把 ODE 改成边缘相同的 SDE，逐步高斯 |
| 要不要改网络结构 | 否，只改优化目标 | 否；奖励模型另训 | 否，只改采样器 |
| 奖励 | 可压缩性、美学、VLM 对齐 | VideoReward（VQ / MQ / TA） | GenEval 规则、OCR、PickScore |
| 相对谁 | 相对 RWR：逐步 PG 更强 | 相对 SFT / RWR：恒定 \(\beta\) 的 Flow-DPO 更强 | 相对 SFT / Flow-DPO：online GRPO 明显更强 |
| 奖励黑客 | 明确存在（过优化、风格崩） | 时变 \(\beta_t\) 会黑；恒定 \(\beta\) 可压 | KL 约束可压质量和多样性崩 |

## 和本仓库其它笔记的关系

- 交通仿真侧的 GRPO 微调见 [RIFT](../RIFT_Closed_Loop_RL_Fine_Tuning_2025.md)。
- Flow Matching 条件注入见 [AdaLN](../AdaLN_Conditioning_Paper_Guide.md)、[ControlNet × Flow Matching](../ControlNet_ControlNeXt_FlowMatching_LaneChange_Style.md)。
- 本目录关注的是：**生成器本身当策略，用 RL 对齐下游奖励**，而不是规划器里的条件注入。
