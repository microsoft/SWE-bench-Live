# SWE-bench-Live-Verified: Automatic Filtering of SWE-bench-Live-Full

The process of automatically filtering the SWE-bench-Live-Full dataset to produce **SWE-bench-Live-Verified** is now live.

---

## Methodology

We input the issue description, gold patch, and Fail2Pass test cases of each instance into a large language model (LLM) to assess whether the instance is suitable ("good") for evaluating agents' debugging capabilities.

Based on insights from several previous analyses:

- **[SWE-bench-Verified](https://openai.com/index/introducing-swe-bench-verified/)**
- **[Agentless](https://arxiv.org/abs/2407.01489)**
- **[Meng et al.](https://arxiv.org/abs/2411.10213)**
- **[SWE-Bench+](https://arxiv.org/abs/2410.06992)**

We prompt the LLM to classify instances into the following categories:

| Category | Description |
|---|---|
| **1** | Issue description has minor vagueness or missing details, making it difficult (but possible) to understand, reproduce, and solve the bug. |
| **2** | Issue description is highly vague, unclear, or incomplete, making reproduction and resolution impossible. |
| **3** | Issue description includes proposed solutions, but at least one solution is misleading or incorrect relative to the gold patch. |
| **4** | Issue description is sufficient, but the provided test cases are inadequate, overly broad, missing, or underspecified, such that passing tests does not reliably indicate issue resolution. |
| **5** | Gold patch or test cases require specific outputs, errors, or formats not described in the issue description, or impose unnecessarily narrow or incorrect constraints, causing valid solutions to fail. |
| **6** | Ground truth fix (patch or natural-language description) is explicitly provided in the issue, making the fix trivial. |
| **7** | Issue description is complete, precise, and suitable for evaluation—neither overly simple nor overly complex (**Good instance**). |
| **8** | Other issues (e.g., environmental constraints, flaky tests, licensing) render the instance unsolvable or unreliable for evaluation. |

---

## Results

We used **GPT-o3** to categorize 1699 SWE-bench-full instances previously sampled by [SWE-bench-Verified](https://openai.com/index/introducing-swe-bench-verified/):

### Negative Instance Filtering (Including Category 6):

- SWE-bench-Verified identified **1160 negative instances**.
- GPT-o3 identified **652 negative instances**, with a filter ratio of **38.4%**.
- Overlap between methods: **469 instances**.

| Metric | Value |
|--------|-------|
| **Precision** | 469 / 652 ≈ **72%** |
| **Recall** | 469 / 1160 ≈ **40%** |

### Negative Instance Filtering (Excluding Category 6):

Since our prompt explicitly excludes too easy cases (Category 6), but SWE-bench-Verified did not, we also analyzed the filtered instances exluding Category 6:

- GPT-o3 identified **433 negative instances** (excluding Category 6).
- Overlap with SWE-bench-Verified: **400 instances**.

| Metric | Value |
|--------|-------|
| **Precision** | 400 / 433 ≈ **92%** |
| **Recall** | 400 / 1160 ≈ **35%** |

---

## Model Comparison (GPT-4.1 vs. GPT-o3)

We additionally evaluated **GPT-4.1**:

### Negative Instances with Category 6:

- GPT-4.1 Precision: **122 / 161 ≈ 76%**
- GPT-4.1 Recall: **122 / 1160 ≈ 10.5%**

### Negative Instances without Category 6:

- GPT-4.1 Precision: **115 / 133 ≈ 86.5%**
- GPT-4.1 Recall: **115 / 1160 ≈ 10%**

### Indications

- Reasoning models (**GPT-o3**) demonstrate superior precision and recall compared to general-purpose models (**GPT-4.1**).
- LLMs generally show **very high precision** identifying problematic instances but adopt a **conservative, cautious** approach, resulting in lower recall (fewer filtered instances).

---

## SWE-bench-Live-Verified and Agents' Performance on it

The initial **Verified** subset includes **500 instances** filtered from the SWE-bench-Live-full set from **July 2024** to **April 2025**. The filter ratio is around 38%.

Due to time constraints and computational limitations, we simply retain instances from the Live-Lite subset (version 2025-04-20) that overlap with the Live-Verified subset to recalculate the success rates of various agents based on these remaining instances. After filtering, **174 instances** remain.

### Success Rates on Filtered Live-Verified (174 instances)

| Agent Type           | GPT-4o | GPT-4.1 | Claude-3.7-Sonnet | DeepSeek-V3 |
|----------------------|--------|---------|-------------------|--------------|
| **SWE-agent**        | 14.94% | 16.09%  | **19.54%**        | 13.22%       |
| **Agentless**        | 13.22% | 11.49%  | 12.07%            | 13.79%       |
| **Openhands+CodeAct**| 6.32%  | 12.07%  | **20.69%**        | 13.22%       |

### Observations

- The results do not demonstrate a significant increase of resolved rate compared to the previously evaluated SWE-bench-Verified dataset, suggesting **current LLMs and debugging agents are likely overfitting** to the existing SWE-bench-Verified benchmark.
- Additionally, because trivial instances have been explicitly excluded from our new Verified subset, which SWE-bench-Verified did not, it is reasonable that overall success rate increase remains modest.


## SWE-bench-Live-Verified: Current Limitations and Future Directions

Currently, retrieving Fail2Pass instance content relies on hardcoded parsers, missing some test cases. 

To address this, future developments will include an LLM-based agent capable of automatically extracting Fail2Pass test case contents, enabling seamless scalability across programming languages.
