# Learning When to Answer: Behavior-Oriented Reinforcement Learning for Hallucination Mitigation

## Authors
- [Max](https://github.com/milk333445)
- [Eddie](https://github.com/a227799770055)
- [CHI_YA](https://github.com/CHIYA-1225)

> **Note:** This README provides a brief overview. For the complete paper with full technical details, methodology, and experimental results, please refer to `paper.pdf`.

## 1. Abstract
Large language models (LLMs) often hallucinate under uncertainty, producing fluent yet unsupported responses.
We argue that hallucination is fundamentally a decision-making problem, involving when to answer, when to abstain, and how confidently to respond, rather than purely a generation error.

We propose a behavior-oriented reinforcement learning framework that explicitly models these decisions.
Our method integrates behavior alignment, entropy-based uncertainty modeling, response quality shaping, and length regularization into a unified reward function, and is optimized via a two-stage training process combining preference learning (DPO) and reinforcement learning (GRPO).

Experimental results show that our approach reduces hallucination by over 65%, consistently outperforming strong baselines such as GPT-4o.
The learned behavior generalizes effectively to unseen domains, improves response quality across eight evaluation dimensions, and preserves general knowledge capability without catastrophic forgetting.

These findings demonstrate that hallucination mitigation can be effectively achieved by learning behavior under uncertainty, enabling LLMs to produce responses that are both reliable and useful.

## Acknowledgments
Special thanks to [Eddie](https://github.com/a227799770055) and [CHI_YA](https://github.com/CHIYA-1225) for their valuable contributions to this project.
