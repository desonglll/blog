---
title: How to build an effective agent
date: 2025-01-19 22:12:45
---

# 如何构建有效的LLM智能体

## 引言

随着大型语言模型（LLM）的发展，基于LLM的智能体在各领域的应用日益广泛。例如，清华大学与面壁团队提出了**智能体互联网**（Internet of Agents，IoA）的概念，旨在连接全球的智能体，实现协同工作。 此外，基于LLM的教学智能体在教育领域的应用也受到关注，研究者探索了其在个性化教学和人机协同学习中的潜力。 

## 什么是智能体

智能体的定义有很多种方式，一种定义是基于完全自主系统视角，在这一视角下，智能体被视为能够长时间独立运行的系统，具备感知环境（通过传感器）、决策以及对环境采取行动（通过执行器）的能力。其核心特征是自主性和适应性，旨在通过感知-决策-行动的循环来优化特定的性能指标。另外一种定义基于工作流视角，智能体被定义为遵循预定义工作流程的规范性实现。在这种定义下，智能体的行为被设计为按照特定的流程或规则执行，以完成特定任务。

Yutao Yue（2022）在论文《A World-Self Model Towards Understanding Intelligence》中提出了**世界-自我模型**（World-Self Model，WSM），强调智能体应具备对外部世界和自身的建模能力，以实现感知与认知的有效连接。 这一模型为理解智能体的本质提供了新的视角。Clément Moulin-Frier等人（2017）在论文《Embodied Artificial Intelligence through Distributed Adaptive Control: An Integrated Framework》中提出了**分布式自适应控制**（Distributed Adaptive Control， DAC）框架，强调智能体的**具身性**（embodiment）和**自适应性**（adaptivity），主张通过整合多种人工智能方法，实现智能体在动态环境中的自主学习和适应能力。Yi Zeng等人（2024）在论文《Brain-inspired and Self-based Artificial Intelligence》中提出了**基于自我的人工智能**（Self-based AI）概念，强调智能体应具备**自我模型**（self-model），以实现对自身状态和行为的认知与控制。 这一观点强调了自我意识在智能体中的重要性。Warisa Sritriratanarak和Paulo Garcia（2023）在论文《On a Functional Definition of Intelligence》中提出了智能体的**功能性定义**，主张从外部观察智能体的行为，以评估其智能水平。 这一方法为智能体智能程度的量化测量提供了新的思路。

## 什么情况下需要使用智能体（或者不使用智能体）

## 何时以及如何使用提前设置好的框架

## 构建的顺序