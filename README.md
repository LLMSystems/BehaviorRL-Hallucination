# Learning When to Answer: Behavior-Oriented Reinforcement Learning for Hallucination Mitigation

## 1. Abstract
Large language models (LLMs) often hallucinate under uncertainty, producing fluent yet unsupported responses.
We argue that hallucination is fundamentally a decision-making problem, involving when to answer, when to abstain, and how confidently to respond, rather than purely a generation error.

We propose a behavior-oriented reinforcement learning framework that explicitly models these decisions.
Our method integrates behavior alignment, entropy-based uncertainty modeling, response quality shaping, and length regularization into a unified reward function, and is optimized via a two-stage training process combining preference learning (DPO) and reinforcement learning (GRPO).

Experimental results show that our approach reduces hallucination by over 65%, consistently outperforming strong baselines such as GPT-4o.
The learned behavior generalizes effectively to unseen domains, improves response quality across eight evaluation dimensions, and preserves general knowledge capability without catastrophic forgetting.

These findings demonstrate that hallucination mitigation can be effectively achieved by learning behavior under uncertainty, enabling LLMs to produce responses that are both reliable and useful.

## 2. Introduction
Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks.
However, they are prone to hallucination, generating fluent but unsupported or incorrect responses, particularly under conditions of uncertainty or incomplete information.
Such behavior poses significant challenges for real-world applications where reliability is critical.

Existing approaches to mitigate hallucination primarily focus on improving knowledge access or post-hoc verification.
For example, retrieval-augmented generation (RAG) enhances factual grounding by incorporating external documents, while various detection methods attempt to identify hallucinated outputs after generation.
Despite their effectiveness, these approaches do not explicitly address a fundamental issue:
LLMs often fail to make appropriate decisions under uncertainty, such as when to answer, when to abstain, and how confidently to respond.

In this work, we argue that hallucination is not merely a generation error, but a consequence of suboptimal decision-making under uncertainty.
From this perspective, mitigating hallucination requires learning a policy that balances correctness, confidence, and usefulness, rather than solely improving token-level prediction.

To this end, we propose a behavior-oriented reinforcement learning framework that explicitly models hallucination as a decision problem.
Our approach optimizes model behavior along three dimensions:

1. **Whether to answer or abstain**, based on the availability of reliable information
2. **How to construct responses**, ensuring grounded and informative outputs
3. **How confident to be**, avoiding overconfident generation under uncertainty

We design a unified reward function that integrates:

- **behavior alignment**, encouraging correct decision boundaries between answering and abstaining
- **uncertainty modeling**, via entropy-based overconfidence penalties
- **response quality shaping**, promoting completeness, actionability, and evidence use
- **length regularization**, preventing degenerate short-answer or overly verbose behaviors

To optimize this objective, we adopt a two-stage training framework consisting of preference-based initialization (DPO) followed by reinforcement learning (GRPO), enabling stable and fine-grained policy optimization.

We conduct extensive experiments to evaluate the proposed method.
The results show that our approach:

* reduces hallucination by over **65%**, outperforming strong baselines such as GPT-4o
* generalizes effectively to unseen domains
* improves response quality across eight evaluation dimensions
* preserves general knowledge capability without catastrophic forgetting

These findings demonstrate that hallucination mitigation can be effectively achieved by modeling it as a behavior learning problem under uncertainty, rather than solely improving knowledge or generation.

## 3. Method

### 3.1 Problem Formulation
We consider the problem of hallucination in large language models as a decision-making task under uncertainty, rather than purely a generation problem.

Given an input query $x$ and a set of contextual information $C = {c_1, c_2, \dots, c_k}$, a language model generates a response $y$.
The goal is to produce responses that are both grounded and useful, while avoiding unsupported or misleading outputs.

#### 3.1.1 Hallucination as Decision-Making
We define hallucination as a failure to make appropriate decisions under uncertainty.
Specifically, the model must jointly determine:

1. Whether to answer or abstain
2. How to construct the response
3. How confident the response should be

This leads to a unified formulation where hallucination arises when the model produces a response that is:

- unsupported by available information, or
- expressed with excessive confidence under uncertainty.

#### 3.1.2 Controlled Contextual Scenarios

To systematically study model behavior, we consider two controlled scenarios:

* No reliable information
  The provided context does not contain sufficient evidence to answer the query.
  The desired behavior is to abstain.

* Partially correct information
  The context contains a mixture of relevant and irrelevant information, including at least one correct supporting evidence.
  The desired behavior is to generate a selective and grounded response.

These scenarios allow us to explicitly model the decision boundary between answering and abstaining.

#### 3.1.3 Groundedness and Uncertainty

We introduce a groundedness score $s \in [0,1]$, which measures the degree to which a response is supported by the available context.

In addition, we quantify model uncertainty using the average token-level entropy:

$$
H_{\text{avg}} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}} p_t(v)\log p_t(v)
$$

where $T$ is the response length and $p_t(v)$ is the predicted probability of token $v$ at step $t$.

We interpret:

* low entropy as high confidence
* high entropy as high uncertainty

Hallucination often occurs when the model exhibits low entropy (high confidence) despite low groundedness.

#### 3.1.4 Objective

Our objective is to learn a policy that:

* abstains when reliable information is unavailable
* produces grounded responses when sufficient evidence exists
* avoids overconfident behavior under uncertainty
* maintains response quality and usefulness

This formulation naturally leads to a reinforcement learning framework, where the model is trained to optimize behavior under uncertainty rather than purely maximizing likelihood.

### 3.2 Reward Function
We design a behavior-oriented reward function to explicitly model hallucination as a decision-making problem under uncertainty.
The reward integrates four complementary components: behavior alignment, confidence control, response quality, and length regularization.

Formally, for a generated response $y$, the overall reward is defined as:

$$
R=R_{behavior}​+R_{confidence}​+R_{quality}​+R_{length}​
$$

Each component is described below.

#### 3.2.1 Behavior Alignment
We first enforce correct high-level behavior by distinguishing between two controlled scenarios:
- No reliable information → the model should abstain
- Partially correct information → the model should answer

Let $a\in{0,1}$ denote the model’s action, where $a=1$ indicates abstention and $a=0$ indicates answering.
The behavior reward is defined as:

$$
R_{\text{behavior}}=
\begin{cases}
+R_{\text{abstain}},& \text{if abstention is correct} \\
-\alpha, & \text{if abstention is incorrect} \\
0, & \text{otherwise}
\end{cases}
$$

This component ensures that the model learns the correct decision boundary between answering and abstaining.

#### 3.2.2 Confidence Control via Entropy
To regulate hallucination under uncertainty, we introduce an overconfidence penalty based on the entropy of the model’s output distribution.

Given a generated sequence of length (T), the average token-level entropy is:

$$
H_{\text{avg}} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}} p_t(v)\log p_t(v)
$$

We define overconfidence as:

$$
\text{overconf} = \max(0, U_{\min} - H_{\text{avg}})
$$

where $U_{\min}$ is a threshold representing minimum acceptable uncertainty.

The confidence penalty is then applied as:

$$
R_{\text{confidence}} = -\gamma (1 - s)\cdot \text{overconf}
$$

where $s \in [0,1]$ denotes a groundedness score measuring how well the response is supported by available information.

This formulation ensures that:
- Low-quality responses $low (s)$ are discouraged from being overly confident
- High-quality responses are not penalized for confident generation

#### 3.2.3 Response Quality Shaping
While the groundedness score $s$ captures factual support, it does not fully reflect response utility.
To address this, we introduce an additional quality signal $q \in [0,1]$, computed from multiple dimensions:
- Completeness
- Actionability
- Evidence use
- Conciseness

The quality reward is defined as:

$$
R_{\text{quality}} = \beta \cdot q
$$

This component encourages responses that are not only correct, but also informative, actionable, and well-structured.
Importantly, this signal complements the groundedness score rather than duplicating it, enabling the model to maintain response usefulness while reducing hallucination.

#### 3.2.4 Length Regularization
During reinforcement learning, we observe a degenerate short-answer behavior, where the model produces overly brief responses to minimize hallucination risk.

To mitigate this issue, we introduce a length band constraint:

$$
L \in [L_{\text{low}}, \infty]
$$

where (L) is the number of generated tokens.
The length penalty is defined as:

$$
R_{\text{length}} =
-\eta \left(
\max\left(0, \frac{L_{\text{low}} - L}{L_{\text{low}}}\right)
\right)
$$

This formulation:
- penalizes under-informative responses (too short)

Empirically, we set $L_{\text{low}} = 50$, based on observed trade-offs between response completeness and conciseness.

#### 3.2.5 Summary
The proposed reward function jointly optimizes:

- Correct decision-making (when to answer vs. abstain)
- Uncertainty-aware behavior (avoid overconfident hallucination)
- Response utility (quality and usefulness)
- Stability of generation (prevent degenerate short)

This unified formulation enables the model to learn balanced behavior under uncertainty, rather than optimizing solely for correctness or fluency.

### 3.3 Training Framework
To optimize the proposed behavior-oriented reward function, we adopt a two-stage LORA training framework consisting of preference-based initialization followed by reinforcement learning refinement.

#### 3.3.1 Stage 1: Preference Optimization (DPO Initialization)

Directly applying reinforcement learning to optimize complex behavior objectives can lead to unstable training dynamics and degenerate policies.
To mitigate this, we first perform a preference-based initialization using Direct Preference Optimization (DPO).

Given an input $x$ and contextual information $C$, we construct preference pairs $(y^{+}, y^{-})$, where:

* $y^{+}$: preferred response (e.g., grounded or correct behavior)
* $y^{-}$: less preferred response (e.g., hallucinated or incorrect behavior)

The DPO objective encourages the model to assign higher likelihood to preferred responses:

$$
\mathcal{L}*{\text{DPO}} = - \log \sigma \left( \beta*{\text{dpo}} \left( \log \pi_{\theta}(y^{+}|x,C) - \log \pi_{\theta}(y^{-}|x,C) \right) \right)
$$

This stage provides a stable initialization, aligning the model with desirable high-level behaviors such as answering correctly or abstaining when appropriate.

#### 3.3.2 Stage 2: Reinforcement Learning (GRPO)
After initialization, we further refine the model using reinforcement learning to optimize fine-grained behavioral signals.

For each input $(x, C)$, we sample a set of responses:

$$
{y_1, y_2, \dots, y_N} \sim \pi_{\theta}(\cdot | x, C)
$$

Each response is evaluated using the proposed reward function $R(y)$, which incorporates behavior alignment, confidence control, response quality, and length regularization.

We adopt Group Relative Policy Optimization (GRPO), which optimizes relative preferences within a group of sampled responses rather than relying on absolute reward values.

Specifically, the advantage of each response is computed relative to the group:

$$
A_i = R(y_i) - \frac{1}{N} \sum_{j=1}^{N} R(y_j)
$$

The policy is then updated to increase the likelihood of higher-reward responses while decreasing that of lower-reward ones.

#### 3.3.3 Training Stability Considerations
During reinforcement learning, we observe a tendency for the model to adopt degenerate strategies, such as producing overly short responses to minimize hallucination risk.

To address this issue, we incorporate:

* quality-based reward shaping, ensuring that responses remain informative and actionable
* length band regularization, preventing both under-informative

These mechanisms are critical for maintaining a balance between hallucination mitigation and response usefulness.


## 4. Experiments
### 4.1 Experimental Setup
#### 4.1.1 Models
We evaluate our method on a 12B-parameter open-weight language model (Gemma3-12B-IT).
We compare against:

* the base model (without fine-tuning)
* GPT-4o as a strong closed-source baseline

#### 4.1.2 Evaluation Settings

We conduct experiments under controlled contextual conditions with varying information quality, including:

- scenarios with no reliable information
- scenarios with partially correct information

This setup allows us to evaluate both hallucination suppression and response quality under uncertainty.

#### 4.1.3 Metrics
##### (1) Hallucination Metric
We measure hallucination using a binary judgment:

* whether the response is supported by available information
* or contains unsupported or incorrect content

We report:

* Hallucination Rate (%)
* analyzed across different difficulty levels (e.g., varying context size)

##### (2) Response Quality Metrics (LLM-as-a-Judge)
To evaluate response utility, we adopt an LLM-as-a-judge framework with eight evaluation dimensions.

Each response is scored independently on a 1–5 scale (except fluency, which uses 1–3), and normalized to $[0,1]$ for aggregation.

1. Relevance (1–5)
Measures whether the response focuses on essential information without introducing irrelevant or redundant content.

2. Coherence (1–5)
Evaluates the logical organization and clarity of the response, including sentence flow and structural consistency.

3. Consistency (1–5)
Assesses whether all statements in the response are fully supported by the provided information, without hallucinated content.

4. Fluency (1–3)
Measures grammatical correctness, readability, and natural language usage.

5. Completeness (1–5)
Evaluates whether the response fully addresses all key aspects of the user query.

6. Actionability (1–5)
Measures whether the response provides clear and actionable next steps, conditions, or recommendations.

7. Evidence Use (1–5)
Assesses whether the response explicitly explains the reasoning or supporting evidence behind its conclusions.

8. Conciseness (1–5)
Evaluates whether the response is succinct while preserving necessary information, without redundancy or verbosity.

This metric reflects the overall usefulness and quality of the response, complementing hallucination evaluation.

#### 4.1.4 Dataset
We construct a controlled dataset to evaluate hallucination behavior under different information conditions.

The dataset consists of two types of scenarios:

- No reliable information (all_wrong)
All provided context is irrelevant or incorrect. The model is expected to abstain.
- Partially correct information (one_correct)
The context contains a mixture of relevant and irrelevant information, including at least one correct supporting evidence.
##### Dataset Size
The dataset contains a total of X samples, distributed as follows:
all_wrong
- $K=1$: 300
- $K=3$: 300
- $K=5$: 300
- $K=7$: 150
- $K=9$: 150

one_correct
- $K=1$: 150
- $K=3$: 150
- $K=5$: 150
- $K=7$: 150
- $K=9$: 150

where $K$ denotes the number of contextual documents provided.

The dataset used in this work is constructed from internal sources and is not publicly available due to data privacy and usage constraints. 
However, the dataset is designed with controlled scenarios to enable systematic evaluation of hallucination behavior under different information conditions.

### 4.2 Main Results
We evaluate the proposed method in terms of hallucination mitigation, generalization ability, and response quality.

#### 4.2.1 Hallucination Reduction
Table 1: Hallucination Rate (%) across Different Context Sizes (Lower is Better)

| K       | GPT-4o | Gemma3-12B (Base) | Gemma3-12B (Ours) |
| ------- | ------ | ----------------- | ----------------- |
| 1       | 14.3%  | 24.3%             | **6.0%**          |
| 3       | 16.3%  | 22.0%             | **8.3%**          |
| 5       | 17.0%  | 23.3%             | **8.6%**          |
| 7       | 23.0%  | 25.6%             | **8.6%**          |
| 9       | 19.6%  | 26.0%             | **10.3%**         |
| **Avg** | 18.04% | 24.24%            | **8.36%**         |

Our method significantly reduces hallucination compared to both the base model and GPT-4o.

- The fine-tuned model achieves an average hallucination rate of **8.36%**,
  compared to **24.24% for the base model** and **18.04% for GPT-4o**.

- This corresponds to a **65.5% relative reduction** compared to the base model,
  and a **53.6% reduction compared to GPT-4o**.

- The improvement is consistent across all context sizes $(K = 1 \sim 9)$,
  indicating robustness under increasing contextual complexity.

Notably, the performance gap becomes more pronounced at larger $K$,
suggesting that the proposed method is particularly effective under noisy or complex contexts.

#### 4.2.2 Generalization Performance
To evaluate generalization, we test on an unseen domain dataset.

**Table 2: Hallucination Rate (%) on Unseen Domain Dataset (Lower is Better)**

| K       | GPT-4o | Gemma3-12B (Base) | Gemma3-12B (Ours) |
| ------- | ------ | ----------------- | ----------------- |
| 1       | 2.3%   | 12.7%             | **2.3%**          |
| 3       | 5.3%   | 12.3%             | **5.0%**          |
| 5       | 6.7%   | 14.7%             | **5.6%**          |
| 7       | 9.3%   | 18.3%             | **7.6%**          |
| 9       | 8.0%   | 17.3%             | 8.6%         |
| **Avg** | 6.32%  | 15.06%            | **5.82%**         |

The fine-tuned model maintains strong hallucination suppression under domain shift.

- It achieves an average hallucination rate of **5.82%**, outperforming both the base model (**15.06%**) and GPT-4o (**6.32%**).

- The model consistently matches or surpasses GPT-4o across most settings.

These results indicate that the learned behavior is **not dataset-specific**,
but generalizes effectively to unseen domains.

#### 4.2.3 Response Quality
We further evaluate response quality using eight LLM-as-a-judge metrics.

**Table 3: Response Quality Evaluation (Higher is Better)**

| Metric        | Range | Gemma3-12B (Base) | Gemma3-12B (Ours) |
| ------------- | ----- | ----------------- | ----------------- |
| Relevance     | 1–5   | 4.10              | **4.62**          |
| Coherence     | 1–5   | 3.62              | **4.42**          |
| Consistency   | 1–5   | 4.69              | **4.83**          |
| Fluency       | 1–3   | 2.86              | **2.87**          |
| Completeness  | 1–5   | 3.29              | **3.95**          |
| Actionability | 1–5   | 2.70              | **3.32**          |
| Evidence Use  | 1–5   | 2.09              | **4.79**          |
| Conciseness   | 1–5   | 3.97              | **4.37**          |
| **Average**   |       | 3.42              | **4.15**          |

The proposed method improves response quality across all evaluated dimensions.

- The overall score increases from **3.42 to 4.15**,
  corresponding to a **+21.4% relative improvement**.

- The most significant improvement is observed in **Evidence Use**
  (2.09 → 4.79, +129%), indicating substantially better grounding and reasoning transparency.

- Improvements are also observed in:

  - **Completeness** (3.29 → 3.95)
  - **Actionability** (2.70 → 3.32)

- Fluency remains stable (2.86 → 2.87),
  suggesting that gains are achieved without sacrificing language quality.

#### 4.2.4 General Knowledge Preservation
To ensure that hallucination reduction is not achieved at the expense of general capabilities,
we evaluate the model on a diverse set of benchmark datasets covering:

- English knowledge (MMLU, MMLU-Pro)
- Chinese knowledge (CMMLU, CEVAL)
- Instruction following (IFEval)
- Mathematical reasoning (GSM8K, Math-500)

**Table 4: General Knowledge Evaluation (Accuracy, Higher is Better)**

| Domain                 | Dataset  | Base   | Ours       |
| ---------------------- | -------- | ------ | ---------- |
| English                | MMLU     | 0.7714 | 0.7686     |
| English                | MMLU-Pro | 0.6034 | 0.6000     |
| Instruction            | IFEval   | 0.8529 | 0.8571 |
| Chinese                | CMMLU    | 0.6344 | 0.6200     |
| Chinese                | CEVAL    | 0.6636 | 0.6759 |
| Math                   | GSM8K    | 0.9574 | 0.8842     |
| Math                   | Math-500 | 0.9000 | 0.8000     |
| **Average**            |          | 0.7690 | 0.7436     |
| **Average (w/o Math)** |          | 0.7051 | 0.7043 |

The results show that the proposed method maintains general knowledge capability.

- When excluding mathematical reasoning tasks,
  the average accuracy remains nearly unchanged (**0.7051 → 0.7043**),
  indicating **no significant degradation in general knowledge performance**.

- Instruction-following ability slightly improves (IFEval: 0.8529 → 0.8571),
  suggesting that behavior-oriented training enhances controllability.

- While performance decreases on mathematical reasoning tasks,
  this is expected since the proposed reward function does not explicitly optimize for multi-step reasoning.

Overall, these results demonstrate that hallucination mitigation is achieved **without catastrophic forgetting**,
and the model retains its general-purpose capabilities.

#### 4.2.5 Summary
Overall, the proposed method achieves a strong balance between hallucination mitigation, response quality, and general capability preservation.

- It reduces hallucination by over 65% compared to the base model, and consistently outperforms GPT-4o across different context settings.
- The learned behavior generalizes effectively to unseen domains, maintaining superior performance under distribution shift.
- At the same time, response quality improves significantly across all eight evaluation dimensions, particularly in completeness, actionability, and evidence use.
- Importantly, the model preserves its general knowledge capability, with no significant degradation observed outside of mathematical reasoning tasks.

These results demonstrate that hallucination can be effectively mitigated by treating it as a decision-making problem under uncertainty, rather than purely a generation problem.
By jointly optimizing behavior, confidence, and response quality, the model learns to produce responses that are both reliable and useful, without sacrificing overall capability.


### 4.3 Analysis
#### 4.3.1 Entropy and Overconfidence
To understand the role of uncertainty in hallucination, we analyze the distribution of token-level entropy across different scenarios.

We compute the average token entropy for each response and apply a logarithmic transformation for stability:

$$
\tilde{H} = \log_{10}(H_{\text{avg}} + \epsilon)
$$
##### Entropy Across Information Conditions
Table 4 shows the median entropy under two controlled scenarios.

**Table 4: Median Entropy under Different Information Conditions**

| K | all_wrong | one_correct |
| - | --------- | ----------- |
| 1 | -0.6790   | -1.5007     |
| 3 | -0.8695   | -1.7862     |
| 5 | -0.8462   | -1.8510     |
| 7 | -0.8746   | -1.7624     |
| 9 | -0.8994   | -1.7480     |

We observe a clear separation between the two conditions:

- all_wrong responses exhibit significantly higher entropy (less confident)
- one_correct responses exhibit lower entropy (more confident)

This indicates that entropy serves as an effective proxy for distinguishing between **answerable and unanswerable scenarios**.

##### Entropy and Abstention Behavior
We further analyze entropy conditioned on model behavior in the no-information scenario.

**Table 5: Median Entropy for Abstained vs. Non-Abstained Responses**

| K | Abstained | Not Abstained |
| - | --------- | ------------- |
| 1 | -0.6444   | -1.1390       |
| 3 | -0.7498   | -1.3853       |
| 5 | -0.7753   | -1.3675       |
| 7 | -0.7649   | -1.3284       |
| 9 | -0.7664   | -1.5338       |

We find that:

* Responses that correctly abstain have higher entropy
* Responses that incorrectly answer exhibit lower entropy (higher confidence)

This suggests that hallucination is strongly associated with overconfident behavior under insufficient information.

These observations support our reward design:

* entropy captures uncertainty in generation
* hallucination arises when the model is overconfident despite low groundedness

This motivates the use of an overconfidence penalty, which discourages low-entropy responses when groundedness is low.

## 5. Conclusion
In this work, we revisit hallucination in large language models from a behavioral perspective.
Rather than treating hallucination as a purely generative error, we formulate it as a **decision-making problem under uncertainty**, involving when to answer, how to answer, and how confidently to respond.

We propose a behavior-oriented reinforcement learning framework that integrates:

- **behavior alignment**, to distinguish between answering and abstaining
- **uncertainty modeling**, via entropy-based overconfidence control
- **response quality shaping**, to ensure usefulness and completeness
- **length regularization**, to prevent degenerate generation strategies

Through extensive experiments, we demonstrate that the proposed approach:

- reduces hallucination by over **65%**, consistently outperforming strong baselines such as GPT-4o
- generalizes effectively to unseen domains
- significantly improves response quality across multiple dimensions
- preserves general knowledge capability without catastrophic forgetting

These results highlight that hallucination is not merely a limitation of knowledge, but a consequence of **suboptimal decision-making under uncertainty**.
By explicitly modeling behavior, confidence, and response quality, our framework enables language models to produce outputs that are both **reliable and useful**.


