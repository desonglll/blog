---
title: DeepSeek V3 671B 大模型5 大亮点总结
date: 2025-01-19 22:12:45
tags:
---
# DeepSeek V3 671B 大模型5 大亮点总结

原创 小九九爸爸 [Python 智能研习社](javascript:void(0);) *2024年12月27日 08:47* *湖北* *标题已修改*

导读



昨天DeepSeek 发布了新一代大模型DeepSeek-V3，拥有671B 参数的混合专家（MoE）大语言模型，推理时激活37B 亿参数，在多项评估中超越了其他开源模型，并接近领先的闭源模型，且训练过程稳定高效。



DeepSeek V3



![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/UWiagj1op6rDEEDxp1UiauEB8PYFnFfwLdh8kGyH4oCUuv0icZSrs0lMicDF43FS4KPicHGI59ZUku91FPLRasn4Fjw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

核心看点：

1. 模型架构：整体上依然基于Transformer 架构，同时还继承了在DeepSeek-V2 中经受考验的Multi-head Latent Attention (MLA)和DeepSeekMoE 模块，此外还引入了无辅助损失的负载均衡策略和Multi-Token Prediction。该模型在 14.8 万亿个高质量 token 上进行预训练，并通过监督微调和强化学习进一步提升性能。
2. 无辅助损失负载均衡：DeepSeek-V3 提出了一种新颖的无辅助损失负载均衡策略，通过为每个专家引入一个可学习的偏置项（bias term），并将其加到token-to-expert 的亲和度得分上，以动态调整路由决策，从而实现负载均衡。这种方法避免了使用辅助损失，并在训练过程中动态调整偏置项，以保持负载均衡，同时避免了性能下降，还不会丢弃token。
3. 多Token 预测 (MTP) ：DeepSeek-V3 采用了一种新颖的多 Token 预测训练目标，通过引入多个 MTP 模块，每个模块负责预测一个额外Token，与传统并行预测多个 Token 的方法不同，DeepSeek-V3 采用顺序预测的方式，并保留每个预测深度的完整因果链。这样可以增强训练信号，提高数据效率，并可能使模型更好地预先规划其表示，以更好地预测未来的Token。每个模块包含共享的Embedding和输出Head，外加一个投影矩阵和Transformer block。在选取1个MTP 模块来预测未来的2个 token，从结果来看第二个token的接受度在85%-90%，非常高，而且在多个topic 上都表现出了一致、可靠的高接受率，极大地提升了推理解码速度，TPS 提升了1.8倍。
4. 知识蒸馏：DeepSeek-V3 还将DeepSeek R1 系列模型中的长链推理（CoT）能力蒸馏到自身，显著提升了其推理性能，同时保持了对输出风格和长度的控制。
5. 极致的大模型训练工程优化：DeepSeek-V3 采用了 FP8 混合精度训练框架以及相关的量化和乘法精度提升策略，并设计了DualPipe 训练框架，通过计算和通信的重叠编排、高效的跨节点通信策略来提升整体的训练效率并降低成本；还通过在反向传播时重新计算RMSNorm、MLA Up-Projection，CPU上维护EMA参数、共享Embedding和输出头等策略优化内存使用，这些工程优化对于大模型训练非常重要。



总结



在蒸馏 DeepSeek-R1  的过程中，作者还发现了一个有意思的tradeoff，蒸馏有助于模型性能，但同时会导致输出变长，需要做好平衡。此外，从推理模型进行知识蒸馏也是一个有前景的后训练优化方向，其有效性也显示出长CoT 蒸馏可以帮助其他需要复杂推理的认知任务提升性能表现。



**引用：**

1. DeepSeek-V3 Technical Report：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf