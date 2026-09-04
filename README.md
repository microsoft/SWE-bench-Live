<p align="center">
  <a href="http://swe-bench-live.github.io">
    <img src="assets/banner.png" style="height: 10em" alt="swe-bench-live" />
  </a>
</p>

<p align="center">
  <em>A brand-new, continuously updated SWE-bench-like dataset powered by an automated curation pipeline.</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.23419">
        <img alt="paper" src="https://img.shields.io/badge/ArXiv-%23B31B1B?style=for-the-badge&logo=arXiv">
  </a>
  <a href="./LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/SWE-bench/SWE-bench?style=for-the-badge">
  </a>
  <a href="https://swe-bench-live.github.io">
        <img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-%F0%9F%8F%86-1?style=for-the-badge">
  </a>
  <a href="https://huggingface.co/collections/SWE-bench-Live/swe-bench-live">
        <img alt="dataset" src="https://img.shields.io/badge/Dataset-HF-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=FFD21E">
  </a>
</p>

---

SWE-bench-Live is the **first automatically-updating, multi-language and multi-os** SWE task set designed for agentic benchmarking and training. This repository provides:
1. The **evaluation script** to evaluate the prediction patches of your agent on our public huggingface datasets: _SWE-bench-Live/SWE-bench-Live (Python)_, _SWE-bench-Live/MultiLang_ and _SWE-bench-Live/Windows_. 
2. The **task-creation source code** for you to create your customized SWE tasks for large-scale agentic RFT/RL, each paired with an executable docker sandbox.

## News

- **21/08/2026**: Now the size of SWE-bench-Live/MultiLang reaches 1,077! Each language split has more than 100 task instances! For this update we take an improved strategy suggested by RepoLaunch's users to build&test the repos: only one commit for each repo is selected for RepoLaunch to build&test; after RepoLaunch completes that commit, other commits of the same repo in the dataset is git-checked out directly from the built image, and the RepoLaunch extracted commands and parsers are re-used. This method achieves >=98% success with 82% savings on LM API cost and 78% savings on Docker image storage space when creating execution environments for 856 GitHub issues from 93 repos. See [Development.md](./Development.md#execution-environment-setup-with-repolaunch).
- **08/03/2026**: SWE-bench-Live/Windows has been released along with the leaderboard, evaluating LLM's ability to resolve Windows-specific implementation and take actions in powershell. Newest paper on the multi-language and multi-os SWE task sets is available at [RepoLaunch: Automating Build and Management of Code Repositories across Languages and Platforms](https://arxiv.org/abs/2603.05026).
- **10/01/2026**: SWE-bench-Live/Multi-Language with the leaderboard has been released. Merged into main. Supported languages: C/C++, C#, Java, TS/JS, Go, Rust. For old source code SWE-bench-Live/SWE-bench-Live (Python-only, the NIPS paper version), refer to [python-only branch](https://github.com/microsoft/SWE-bench-Live/tree/python-only).
- **09/17/2025**: Dataset updated (through 08/2025)! We’ve finalized the update process for huggingface dataset SWE-bench-Live/SWE-bench-Live (Python tasks): **Each month, we will add 50 newly verified, high-quality issues to the dataset test split**. The `lite` and `verified` splits will remain frozen, ensuring fair leaderboard comparisons and keeping evaluation costs manageable. To access all the latest issues, please refer to the `full` split!


## 🚀 Set Up

```bash
# Python >= 3.10
pip install -e .
```

> [!NOTE]
> Though this eval script has ensured backward compatibility with SWE-bench-Live/SWE-bench-Live (Python-only, the NIPS paper version), which uses swebench library for evaluation, if you want to evaluate on SWE-bench-Live/SWE-bench-Live (Python), for fair comparison we still recommend you to go to our old [Python-only branch](https://github.com/microsoft/SWE-bench-Live/blob/python-only/README.md) and follow the old evaluation method. The below eval script is more suitable for our new datasets SWE-bench-Live/MultiLang and SWE-bench-Live/Windows.

Test your installation by running:
```bash
python -m evaluation.evaluation \
    --dataset SWE-bench-Live/MultiLang \
    --instance_ids rsyslog__rsyslog-6047 \
    --platform linux \
    --patch_dir gold \
    --output_dir logs/test \
    --workers 1 \
    --overwrite 1
```

## 🚥 Evaluation

> [!NOTE]
> Several users have raised questions about the evaluation protocol, so we would like to clarify that SWE-bench-Live evaluation strictly follows the original [`SWE-bench`](https://github.com/swe-bench/SWE-bench) protocol:
>
> 1. During a rollout, the agent may access only the `problem_statement` field of the Hugging Face dataset and the docker image of the task instance. It must not access any other fields, such as `hint`, `FAIL_TO_PASS`, or `test_patch`. The `test_patch` must not be applied to the repository before or during the rollout. The agent must perform a single rollout based solely on the `problem_statement` on the docker container started from the image of the task instance.
> 2. Prompts, skills, and workflow instructions provided to the agent must not contain solutions specific to any task instance. They may contain only general instructions for the entire benchmark or, at most, for a specific repository. The agent must not use results from the ground-truth evaluation script to refine its solution.
>
> Compliant prompts should follow the [SWE-agent prompt](https://github.com/SWE-agent/SWE-agent/blob/a1193dd8fd84eb3e2cd6b0ecbd0bed1cdbb84993/config/default.yaml) and the [OpenHands prompt](https://github.com/OpenHands/benchmarks/blob/701700e6cad1f6309f456213a974544644bda0f4/benchmarks/swtbench/prompts/default.j2), which contain only the problem statement and general workflow instructions.
>
> When submitting results to our [submissions repository](https://github.com/SWE-bench-Live/submission), you must include your agent's raw rollout trajectories so that the maintainers can verify compliance with the SWE-bench protocol. A trajectory consists of the complete sequence of inputs to and outputs from your agent across all rollout rounds for a given task instance, including the initial prompt provided to the agent. Please follow this [SWE-agent compliant trajectory example](https://github.com/SWE-bench-Live/submission/blob/main/submissions/multilang/all_languages/sweagent/gpt-5.5-medium/oxc-project__oxc-21092/trajectory.txt) when submitting your result. If your organization's policies prohibit sharing the complete set of trajectories, you must provide at least some representative samples for verification. There is a checklist when submitting a PR to help you check whether you meet the protocol requirements again.

Guide on running your model/agent on SWE-bench-Live: GO TO [evaluation/README.md](./evaluation/README.md)

## ⬆️ Submit your results

Thank you for your interest in submitting the success rate of your agent/model to SWE-bench-Live! We coordinate results submission via Pull Requests, see [SWE-bench-Live/submissions](https://github.com/swe-bench-live/submission) for instructions.

## 🐳 Development

If you would like to create your own SWE task instances with executable sandboxes, please follow [Development.md](./Development.md).

### Dataset Curation

In SWE-bench-Live, we propose an automated pipeline for curating SWE-bench-like dataset.

<p align="center">
  <img src="assets/overview.png" alt="SWE-bench-Live Curation Pipeline" style="width: 100%; max-width: 800px;" />
  <br>
  <em>SWE-bench-Live Curation Pipeline</em>
</p>

### RepoLaunch

We addresses the bottleneck of setting up execution environments by automating the process through an LLM-based agentic tool – [RepoLaunch](https://github.com/microsoft/RepoLaunch). It can deliver a testable containerized environment for any given GitHub repository, thereby enabling test-based evaluation in SWE-bench-Live. 

### Collaboration

We welcome external collaborators to help us create more SWE tasks each month, and improve the curation and RepoLaunch source code. Please feel free to raise issues, open discussions and contribute pull requests to this repository and to the [RepoLaunch](https://github.com/microsoft/RepoLaunch) repository to help us improve.


## 📚 Citation

If you refer to the SWE task creation pipeline of SWE-bench-Live, or SWE-bench-Live/SWE-bench-Live (Python only tasks), please cite

```bibtex

@article{zhang2025swebenchgoeslive,
  title={SWE-bench Goes Live!},
  author={Linghao Zhang and Shilin He and Chaoyun Zhang and Yu Kang and Bowen Li and Chengxing Xie and Junhao Wang and Maoquan Wang and Yufan Huang and Shengyu Fu and Elsie Nallipogu and Qingwei Lin and Yingnong Dang and Saravan Rajmohan and Dongmei Zhang},
  journal={arXiv preprint arXiv:2505.23419},
  year={2025}
}

```

If you refer to the automated build and test tool _RepoLaunch_, SWE benchmarking/training/RFT/RL environment build, SWE-bench-Live/Multi-Language or SWE-bench-Live/Windows, please cite

```bibtex
@article{li2026repolaunch,
  title={RepoLaunch: Automating Build and Management of Code Repositories across Languages and Platforms},
  author={Li, Kenan and Li, Rongzhi and Zhang, Linghao and Jin, Qirui and Zhu, Liao and Huang, Xiaosong and Zhang, Geng and Zhang, Yikai and He, Shilin and Xie, Chengxing and others},
  journal={arXiv preprint arXiv:2603.05026},
  year={2026}
}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
