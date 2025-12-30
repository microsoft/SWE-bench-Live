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
  <a href="./assets/dataset.md">
        <img alt="dataset" src="https://img.shields.io/badge/Dataset-HF-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=FFD21E">
  </a>
</p>

---

SWE-bench-Live is a live benchmark for issue resolving, designed to evaluate an AI system's ability to complete real-world software engineering tasks. Thanks to our automated dataset curation pipeline, we plan to update SWE-bench-Live on a monthly basis to provide the community with up-to-date task instances and support rigorous and contamination-free evaluation.

## News
- **25/12/2025**: [SWE-bench-Live Multi-Language Linux-Dist](https://huggingface.co/datasets/SWE-bench-Live/MultiLang) Dataset has been released, together with RepoLaunch that supports (C C++ C# Python Java Go JS/TS Rust) languages & (Windows, Linux) images.
- **04/12/2025**: We have updated eval result of GPT-5 and Claude-4.5 on our website. Though Claude might have seen the ground truth because its knowledge cutoff month is July 2025. We have also separated the RepoLaunch project to [RepoLaunch](https://github.com/microsoft/RepoLaunch/). Please contribute repolaunch agent relevant codes to this new repository. For more info please refer to [PR#35](https://github.com/microsoft/SWE-bench-Live/pull/35).
- **09/17/2025**: Dataset updated (through 08/2025)! We’ve finalized the update process for huggingface dataset SWE-bench-Live/SWE-bench-Live (Python tasks): **Each month, we will add 50 newly verified, high-quality issues to the dataset test split**. The `lite` and `verified` splits will remain frozen, ensuring fair leaderboard comparisons and keeping evaluation costs manageable. To access all the latest issues, please refer to the `full` split!


## 🚀 Set Up

```bash
# Python >= 3.10
pip install -e .
```

> [!NOTE]
> Though this eval script has ensured backward compatibility with SWE-bench-Live/SWE-bench-Live (Python-only, the NIPS paper version), which uses swebench library for evaluation, if you want to evaluate on SWE-bench-Live/SWE-bench-Live (Python), for fair comparison we still recommend you to go to our old [Python-only branch](https://github.com/microsoft/SWE-bench-Live/blob/python-only/README.md) and follow the old evaluation method.

The below eval script is more suitable for our new datasets SWE-bench-Live/MultiLang and SWE-bench-Live/Windows.


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

Evaluate your model on SWE-bench-Live.

Collect patch diff of your agent:

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

Evaluation command:


## ⬆️ Submit your results

Thank you for your interest in submitting results to SWE-bench-Live! We coordinate results submission via Pull Requests, see [SWE-bench-Live/submissions](https://github.com/swe-bench-live/submission) for instructions.

## 🐳 Development

If you would like to run our source code, please refer to [Development.md](./Development.md).

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

We welcome external collaborators to help us create more SWE tasks each month, and improve curation and launch source code. Please contact SWE-bench-Live@microsoft.com

Please feel free to raise issues and contribute pull requests to help us improve.


## 📚 Citation

SWE-bench-Live is built upon the foundation of [SWE-bench](https://swebench.com). We extend our gratitude to the original SWE-bench team for their pioneering work in software engineering evaluation benchmarks.

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
