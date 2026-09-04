# 🚥 Evaluation

Evaluate your model-agent combination on SWE-bench-Live.

## Agent Rollout on SWE-bench-Live

### The forked source codes of the agents we used to run SWE-bench-Live:

[SWE-agent](https://github.com/njukenanli/SWE-agent-for-eval)

[OpenHands](https://github.com/njukenanli/OpenHands-for-eval)

[ClaudeCode](https://github.com/njukenanli/ClaudeCode-for-eval)

[Win-Agent](https://github.com/njukenanli/Win-Agent) (for windows tasks)

### Example trajectories of the above agents on SWE-bench-Live:

[SWE-Live-Trajectory-Archive](https://github.com/SWE-bench-Live/submission)

### Run your model-agent combination on SWE-bench-Live

SWE-bench-Live evaluation strictly follows the original [`SWE-bench`](https://github.com/swe-bench/SWE-bench) protocol:

1. During a rollout, the agent may access only the `problem_statement` field of the Hugging Face dataset and the docker image of the task instance. It must not access any other fields, such as `hint`, `FAIL_TO_PASS`, or `test_patch`. The `test_patch` must not be applied to the repository before or during the rollout. The agent must perform a single rollout based solely on the `problem_statement` on the docker container started from the image of the task instance.
2. Prompts, skills, and workflow instructions provided to the agent must not contain solutions specific to any task instance. They may contain only general instructions for the entire benchmark or, at most, for a specific repository. The agent must not use results from the ground-truth evaluation script to refine its solution.

Compliant prompts should follow the [SWE-agent prompt](https://github.com/SWE-agent/SWE-agent/blob/a1193dd8fd84eb3e2cd6b0ecbd0bed1cdbb84993/config/default.yaml) and the [OpenHands prompt](https://github.com/OpenHands/benchmarks/blob/701700e6cad1f6309f456213a974544644bda0f4/benchmarks/swtbench/prompts/default.j2), which contain only the problem statement and general workflow instructions.

When submitting results to our [submissions repository](https://github.com/SWE-bench-Live/submission), you must include your agent's raw rollout trajectories so that the maintainers can verify compliance with the SWE-bench protocol. A trajectory consists of the complete sequence of inputs to and outputs from your agent across all rollout rounds for a given task instance, including the initial prompt provided to the agent. Please follow this [SWE-agent compliant trajectory example](https://github.com/SWE-bench-Live/submission/blob/main/submissions/multilang/all_languages/sweagent/gpt-5.5-medium/oxc-project__oxc-21092/trajectory.txt) when submitting your result. If your organization's policies prohibit sharing the complete set of trajectories, you must provide at least some representative samples for verification. There is a checklist when submitting a PR to help you check whether you meet the protocol requirements again.

### Collect patch diff of your agent

```bash
# unix
cd /testbed;
[ -d .git ] || { g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); [ -n "$g" ] && cd "${g%/.git}"; } ;
git --no-pager diff HEAD  --text;
```

```powershell
# win
cd C:\testbed;
if (-not (Test-Path .git)) { $g = Get-ChildItem -Directory -Recurse -Depth 2 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '.git' } | Select-Object -First 1; if ($g) { Set-Location $g.Parent.FullName } };
git --no-pager diff HEAD  --text;

```

Prediction patch file format:

```json
{
    "instance_id1": {
        "model_patch": "git diff", 
        ...
    },
    "instance_id2": {
        "model_patch": "git diff", 
        ...
    },
    ...
}
```

## Run gold patch

> [!NOTE]
> 
> Estimated resource for one instance: 4 CPUs with 16 GB RAM. For some large repos like C++ repos even 50GB RAM is required. Otherwise these large repos would go OOM and fail...
> 
> Though we have run tests 3 times during task creation to filter out unstable instances, tests may become invalid overtime, and users have reported that different tests may fail on different machines -- docker does not guarantee full isolation. For benchmarking and training we suggest running evaluation with gold patch three times to filter invalid instances. We allow success rate report with the dorminator the actual number of instances passed with gold patch on your machine at your experiment time.

```bash

# For windows if there are decoding issues: $env:PYTHONUTF8="1" ; $env:PYTHONIOENCODING="utf-8"

python -m evaluation.evaluation \
    --dataset SWE-bench-Live/SWE-bench-Live \
    # or SWE-bench-Live/MultiLang, SWE-bench-Live/Windows
    # or path to local dataset file like jsonl
    --split < refer to Huggingface SWE-bench-Live > \
    # if local jsonl file then ignore this field
    --platform linux \
    # or windows 
    --patch_dir gold \
    --output_dir logs/gold \
    --workers 10 \
    --overwrite 0 \
    # 0 for no and 1 for yes
    --start-month 2025-06 \
    --end-month 2025-07
    # default to oldest and newest if not specified
```

The still valid instances on your machine will be saved to `logs/gold/gold_patch_evaluated_instances.jsonl`, which is the actual subset you can use for benchmarking&training.

## Evaluation of agent-predicted patches

```bash
# For windows if there are decoding issues: $env:PYTHONUTF8="1" ; $env:PYTHONIOENCODING="utf-8"

python -m evaluation.evaluation \
    --dataset SWE-bench-Live/SWE-bench-Live \
    # or SWE-bench-Live/MultiLang, SWE-bench-Live/Windows
    # or path to local dataset file like jsonl
    --split < refer to Huggingface SWE-bench-Live > \
    # if local jsonl file then ignore this field
    --platform linux \
    # or windows 
    --patch_dir <prediction-patch-file-path> \
    --output_dir logs/test \
    --workers 10 \
    --overwrite 0
    # 0 for no and 1 for yes
```

## Docker images

Instance-level Docker images are hosted on DockerHub with name:

```python
def get_default_image_name(instance_id: str, platform: Literal["windows", "linux"]) -> str:
    if platform == "linux":
        med = "x86_64"
    else:
        med = "win"
    name = instance_id.replace("__", "_1776_").lower()
    image = f"starryzhang/sweb.eval.{med}.{name}"
    return image
```
