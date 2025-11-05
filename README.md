
---

# SWE-Zero Pipeline

End-to-end workflow for solving [SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live) tasks using [mini-swe-agent](https://github.com/andysternberg/mini-swe-agent) with self-hosted Qwen model on [Modal](https://modal.com/).

---

## Prerequisites

```bash
# Install dependencies
pip install modal mini-swe-agent swebench

# Setup Modal (get token from https://modal.com/settings)
modal setup
modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>
```

---

## Quick Start

```bash
# Full Modal (GPU + Docker on Modal, maximum parallelism)
modal run run_swebench_modal_selfhosted.py::main --slice 0:10
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench-Live/SWE-bench-Live \
  --split lite \
  --namespace starryzhang \
  --predictions_path predictions_selfhosted_lite_0-10.json \
  --run_id my_eval \
  --modal true

# Hybrid Mode (GPU on Modal, Docker local, cost-optimized, requires local Docker)
modal run run_swebench_modal_selfhosted.py::main --docker-mode local --slice 0:10
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench-Live/SWE-bench-Live \
  --split lite \
  --namespace starryzhang \
  --predictions_path predictions_selfhosted_lite_0-10.json \
  --run_id my_eval \
  --modal false

# Options: --docker-mode [modal|local], --gpu [modal|local], --extract [post-agent|original], --modal [true|false]
```

---

## Pipeline Overview

### Part 1: Patch Generation
**Script**: `run_swebench_modal_selfhosted.py`

Loads Qwen3-Coder-30B-A3B on Modal GPUs → Pulls pre-built Docker images → Runs mini-swe-agent to generate patches → Saves to JSON

**Output**: `predictions_selfhosted_<split>_<slice>.json`

### Part 2: Patch Evaluation
**Command**: `python -m swebench.harness.run_evaluation`

Pulls same Docker images → Applies patches → Runs test suites → Reports results

**Output**: `logs/run_evaluation/<run_id>/`


## Parameters

### Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `SWE-bench-Live/SWE-bench-Live` | HuggingFace dataset |
| `--split` | `lite` | Split: `lite`, `verified`, `test`, `full` |
| `--slice` | `0:5` | Instance range (e.g., `0:10`, `5:15`) |
| `--docker-mode` | `modal` | Docker execution: `modal` or `local` |
| `--gpu` | `modal` | GPU execution: `modal` or `local` |
| `--extract` | `post-agent` | Patch extraction: `post-agent` (direct git diff) or `original` (stream capture) |

### Evaluation
| Parameter | Description |
|-----------|-------------|
| `--dataset_name` | HuggingFace dataset name |
| `--split` | Dataset split to evaluate |
| `--namespace` | DockerHub namespace (use `starryzhang`) |
| `--predictions_path` | Path to predictions JSON |
| `--run_id` | Unique run identifier |
| `--modal` | `true` for Modal sandboxes, `false` for local Docker |
| `--max_workers` | Parallel workers (default: 4) |
| `--timeout` | Per-instance timeout in seconds (default: 1800) |

---

## Project Structure

```
.
├── run_swebench_modal_selfhosted.py  # Patch generation (Modal)
├── run_swebench_local.py             # Local Docker helper
├── swebench/harness/                 # Evaluation harness
│   ├── run_evaluation.py             # Evaluation entry point
│   └── modal_eval/                   # Modal sandbox support
└── mini-swe-agent/                    # Agent framework (submodule)
```

---

## Docker Images

Pre-built images on DockerHub: `starryzhang/sweb.eval.x86_64.<instance_id>:latest`

**Example**: `starryzhang/sweb.eval.x86_64.aws-cloudformation_1776_cfn-lint-3798:latest`

---

## Patch Extraction

Two extraction modes available via `--extract` flag:

### Post-Agent Mode (Default, Recommended)
Extracts patches by executing `git diff` directly after agent completes.

**How it works**:
1. Agent completes task and submits with: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
2. System executes: `git add -A && git diff --cached`
3. Patch captured directly from git command output

**Advantages**:
- ✅ Complete patches (no truncation)
- ✅ Reliable and atomic
- ✅ No stream timing issues
- ✅ Tested with 10KB+ patches

**Usage**:
```bash
# Default (no flag needed)
modal run run_swebench_modal_selfhosted.py::main --slice 0:10

# Explicit
modal run run_swebench_modal_selfhosted.py::main --slice 0:10 --extract post-agent
```

### Original Mode (Legacy)
Captures patches from agent's command output stream (mini-swe-agent default behavior).

**How it works**:
1. Agent submits with: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`
2. Patch captured from command output stream (limited to 5000 chars by mini-swe-agent)

**Limitations**:
- ⚠️ Stream buffer limits (5000 chars)
- ⚠️ May truncate large patches
- ℹ️ Useful for debugging or compatibility

**Usage**:
```bash
modal run run_swebench_modal_selfhosted.py::main --slice 0:10 --extract original
```

---

## Modal Setup

1. **Get Token**: Visit https://modal.com/settings
2. **Configure**:
   ```bash
   modal setup
   modal token set --token-id <id> --token-secret <secret>
   ```
3. **Verify**: `modal token verify`

**Model Configuration**:
- Model: Qwen/Qwen3-Coder-30B-A3B-Instruct
- GPUs: 4x A100-80GB (tensor parallelism)
- Context: 16K tokens, generation: 4K max
- Framework: vLLM 0.10.0

---

## Citation for the starter repos

```bibtex
@article{zhang2025swebenchgoeslive,
  title={SWE-bench Goes Live!},
  author={Linghao Zhang and Shilin He and Chaoyun Zhang and Yu Kang and Bowen Li and Chengxing Xie and Junhao Wang and Maoquan Wang and Yufan Huang and Shengyu Fu and Elsie Nallipogu and Qingwei Lin and Yingnong Dang and Saravan Rajmohan and Dongmei Zhang},
  journal={arXiv preprint arXiv:2505.23419},
  year={2025}
}

@inproceedings{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-world Github Issues?},
  author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=VTF8yNQM66}
}

@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)
