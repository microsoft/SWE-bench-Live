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
  <a href="https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live">
        <img alt="dataset" src="https://img.shields.io/badge/Dataset-HF-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=FFD21E">
  </a>
</p>

---

> [!NOTE]
> The evaluation code in this repo is forked from [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench), with only minimal modifications to support evaluation on the SWE-bench-Live dataset. All other settings remain consistent with SWE-bench to reduce the migration effort. For code part, please respect the original [license](https://github.com/SWE-bench/SWE-bench/blob/main/LICENSE) from the SWE-bench repository.

SWE-bench-Live is a live benchmark for issue resolving, designed to evaluate an AI system's ability to complete real-world software engineering tasks. Thanks to our automated dataset curation pipeline, we plan to update SWE-bench-Live on a monthly basis to provide the community with up-to-date task instances and support rigorous and contamination-free evaluation.

## News
- **04/12/2025**: We have updated eval result of GPT-5 and Claude-4.5 on our website. Though Claude might have seen the ground truth because its knowledge cutoff month is July 2025. We have also separated the RepoLaunch project to [RepoLaunch](https://github.com/microsoft/RepoLaunch/). Please contribute repolaunch agent relevant codes to this new repository. 
- **09/23/2025**: We upgraded RepoLaunch Agent to support building repos on all mainstram languages (C C++ C# Python Java Go JS/TS Rust) and on both Linux&Windows platforms. We added test log parsing functionalities so test log parsing does not depend on pytest any more! We also added minimal rebuild command generation for languages that require resolving dependencies and compiling again after code-fix for automated test. Swebench-Live-MultiLang will be released soon following this major advancement! For RepoLaunch preview, please refer to [RepoLaunch_Preview](https://github.com/microsoft/SWE-bench-Live/tree/repolaunch_preview/launch).
- **09/17/2025**: Dataset updated (through 08/2025)! We’ve finalized the update process for SWE-bench-Live: **Each month, we will add 50 newly verified, high-quality issues to the dataset test split**. The `lite` and `verified` splits will remain frozen, ensuring fair leaderboard comparisons and keeping evaluation costs manageable. To access all the latest issues, please refer to the `full` split!
- **07/19/2025**: We've employed a LLM filter to automatically filter full dataset to create [SWE-bench-Live-Verified](./swebench/collect/produce/README.md). The initial Verified subset contains 500 instances from 2024-07 to 2025-04.
- **06/30/2025**: We’ve updated the dataset — it now includes a total of **1,565** task instances across **164** repositories!

## 📁 Repository Structure

```
├── swebench/             # Core evaluation code (a fork of SWE-bench)
├── launch/               # RepoLaunch tool for environment setup
├── curation/             # Curation pipeline (scripts)
├── assets/               # Repo assets
├── ...
└── README.md             # This file
```

## 🚀 Set Up

```bash
# Python >= 3.10
pip install -e .
```

Test your installation by running:
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench-Live/SWE-bench-Live \
    --split lite \
    --instance_ids amoffat__sh-744 \
    --namespace starryzhang \
    --predictions_path gold \
    --max_workers 1 \
    --run_id validate-gold
```

## 🚥 Evaluation

Evaluate your model on SWE-bench-Live.

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench-Live/SWE-bench-Live \
    --split <lite/full> \
    --namespace starryzhang \
    --predictions_path <path_to_your_preds or gold> \
    --max_workers <num_workers> \
    --run_id <run_id>
```

Instance-level Docker images are hosted on DockerHub.


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

We addresses the bottleneck of setting up execution environments by automating the process through an LLM-based agentic tool – [RepoLaunch](./launch/README.md). It can deliver a testable containerized environment for any given GitHub repository, thereby enabling test-based evaluation in SWE-bench-Live. 

See [./launch](./launch/) folder for RepoLaunch code.

### Collaboration

We welcome external collaborators to help us create more SWE tasks each month. Please contact SWE-bench-Live@microsoft.com

Please feel free to raise issues and contribute pull requests to help us improve.

## 🙏 Acknowledgements

SWE-bench-Live is built upon the foundation of [SWE-bench](https://swebench.com). We extend our gratitude to the original SWE-bench team for their pioneering work in software engineering evaluation benchmarks.

## 📚 Citation

If you found the [SWE-bench-Live](https://swe-bench-live.github.io/) and [SWE-bench](https://swebench.com/) helpful for your research, please cite as follows

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
