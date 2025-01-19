# 仅需一万块钱！清华团队靠强化学习让 7B模型数学打败GPT-4o

关注前沿科技 [量子位](javascript:void(0);) *2025年01月06日 12:29* *北京*

##### PRIME团队 投稿 量子位 | 公众号 QbiAI

OpenAI o1和o3模型的发布证明了强化学习能够让大模型拥有像人一样的快速迭代试错、深度思考的高阶推理能力，在基于模仿学习的Scaling Law逐渐受到质疑的今天，基于探索的强化学习有望带来新的Scaling Law。

近日，清华大学NLP实验室、上海AI Lab、清华大学电子系、OpenBMB社区等团队提出一种新的结合过程奖励的强化学习方法——**PRIME**（Process Reinforcement through IMplicit REwards）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylp9cibP5wzF4tKpOoBtG9rl38y6X1rLrOlq2biaS3UxBPHcUbzSXyxaWjg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

采用PRIME方法，研究人员不依赖任何蒸馏数据和模仿学习，仅用8张A100，花费一万块钱左右，不到10天时间，就能高效训练出一个数学能力超过 GPT-4o、Llama-3.1-70B的7B模型 Eurus-2-7B-PRIME。

具体而言，研究人员利用Qwen2.5-Math-7B-Base作为基座模型，训练出了新模型Eurus-2-7B-PRIME，并在美国IMO选拔考试AIME 2024上的准确率达到26.7%，大幅超越GPT-4o，Llama3.1-70B和Qwen2.5-Math-7B-Instruct，且仅使用了Qwen Math数据的 1/10。其中，强化学习方法PRIME为模型带来了16.7%的绝对提升，远超已知的任何开源方案。

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpLdvH4dB2ttu08jay7rN3tBlLzMafKTD9iaOcRvFhiclzHcm9eXZcDtww/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylp3WWlER5uXXan5qtukNFpXbV9TlPwfExkYC6gC13FGIotrOeqO8HurQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

该项目一经开源就在海外AI社区爆火，短短几天Github取得近300star。

未来，基于PRIME方法和更强的基座模型有潜力训练出接近OpenAI o1的模型。

## PRIME方法介绍

长久以来，开源社区严重依赖数据驱动的模仿学习来增强模型推理能力，但这种方法的局限也显而易见——更强的推理能力需要更高质量的数据，但高质量数据总是稀缺，使得模仿和蒸馏难以持续。

虽然OpenAI o1和o3的成功证明了强化学习有着更高的上限，但强化学习有着两个关键挑战：（1）如何获得精准且可扩展的密集奖励；（2）如何设计可以充分利用这些奖励的强化学习算法。

PRIME算法从隐式过程奖励（implicit process reward）的思想出发解决这两个问题。隐式过程奖励模型可以仅在输出奖励模型（outcome reward model, ORM）的数据，即答案的最终对错上进行训练，而隐式地建模过程奖励，最终自动训练出一个过程奖励模型，这整个过程都有严格的理论保证。

详细推导见：https://huggingface.co/papers/2412.01981

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpz7VCPb6zreq4oQ87QMDficiciaaKslwTk6pic31feEhQruFia7gJeyicJNVw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

基于隐式过程奖励模型的这种性质，研究人员指出将其应用于强化学习有三大优势：

- **过程奖励：**隐式过程奖励模型能够为每个 token 提供价值估计，在提供过程奖励的同时无需训练额外的价值模型（value model）
- **可扩展性：**隐式过程奖励模型只需结果标签即可在线更新。所以，我们可以结合策略模型采样与结果验证器来直接更新PRM，有效缓解分布偏移与可扩展性问题。
- **简洁性：**隐式过程奖励模型本质上就是一种语言模型。在实践中，研究人员发现可以直接用初始的策略模型初始化PRM。

隐式过程奖励解决了PRM在大模型强化学习中怎么用，怎么训，怎么扩展的三大问题，甚至不需要训练额外的奖励模型就可以开始强化学习，易用性和可扩展性极佳。

具体的PRIME算法流程如下图所示，它是一种在线强化学习算法，能够将每个token的过程奖励无缝应用于强化学习流程中。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpmw6fLXUOcVcpiaNKial41iaYIDkc0vyzuzH02xaNjDiaAsxDD5ECrRn0Aw/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1)

## 实验结果

研究人员详细比较了PRIME算法和基线方法。

相比于仅用结果监督，PRIME有着**2.5倍**的采样效率提升，在下游任务上也有着显著提升。

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpibnaxOT2IzZwn7dfHSWDQqjutddAjicI6HZT5BBj5gwqSpWpnibVArz0A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpTACgyNEy8K3rgcfaOricvv58OdrBFS6OdUgWVKBcXeL9BlaK4g56Zyw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

研究人员还验证了PRM在线更新的重要性，可以看到，在线的PRM更新要显著优于固定不更新的PRM，这也证明了PRIME算法设计和合理性。

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpfLficPaSKCiacZ6RicQBcLHrYw3GUzmX9fJJ0bB2adcRjDz9fNrXIhuUQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

此外，研究人员还额外收集数据，基于Qwen2.5-Math-Instruct训练了SOTA水平的EurusPRM，能够在Best-of-N采样中达到开源领先水平。

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpNfM7aVL5RjXzt2Z25GkVQ8GJGGtJTquk205SH4AofMUaE7lz5yTibvQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## Showcase演示

**Question （AIME 2024试题，Claude-3.5-Sonnet做错）**

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylp1YjAdc7WzJryQQcR3BsuSEjEgbjCWNAgfbql77UpkZwmyVu0APeZMQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**Answer**

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpO7qGplWT59PsYg2pPweGRxkt4tuj35oxgXAribktIyajZXKZG2zc8UA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**Question**

Which number is larger? 9.11 or 9.9?

**Answer**

![图片](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBFUd9q3MmibDnsj5KIaxylpRicKuJsdWxsqRx7WLfibWvqMLSJ2gl4PttEPxyPVmoUkJcVTMticDJfAg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

强化学习是连接已有智能体（大模型）和现实世界（世界模型，具身智能）的桥梁，以及将世界反馈内化为模型智能的路径，将在下一代人工智能的发展中起到重要作用。PRIME 算法创新性地将隐式过程奖励与强化学习结合，解决了大模型强化学习的奖励稀疏问题，有望推动大模型复杂推理能力的进一步提升。

blog链接：https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f
GitHub链接：https://github.com/PRIME-RL/PRIME