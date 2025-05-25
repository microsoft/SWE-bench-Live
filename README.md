<p align="center">
  <a href="http://swe-bench.github.io">
    <img src="assets/banner.png" style="height: 10em" alt="swe-bench-live" />
  </a>
</p>

<p align="center">
  <em>A brand-new, continuously updated SWE-bench-like dataset powered by an automated curation pipeline.</em>
</p>

<p align="center">
  <a href="./LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/SWE-bench/SWE-bench?style=for-the-badge">
  </a>
  <a href="https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live">
        <img alt="dataset" src="https://img.shields.io/badge/Dataset-HF-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=FFD21E">
  </a>
</p>

---

> [!NOTE]
> The evaluation code in this repo is forked from [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench), with only minimal modifications to support evaluation on the SWE-bench-Live dataset. All other settings remain consistent with SWE-bench to reduce the migration effort. For code part, please respect the original [license](https://github.com/SWE-bench/SWE-bench/blob/main/LICENSE) from the SWE-bench repository.

SWE-bench-Live is a live benchmark for issue resolving, designed to evaluate an AI system's ability to complete real-world software engineering tasks. Thanks to our automated dataset curation pipeline, we plan to update SWE-bench-Live on a monthly basis to provide the community with up-to-date task instances and support rigorous and contamination-free evaluation.

The initial release of SWE-bench-Live includes **1,319** latest (created after 2024) task instances, each paired with an instance-level Docker image for test execution, covering **93** repositories.

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

## Evaluation

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


## Acknowledgements

We greatly thank the SWE-bench team for their efforts in building [SWE-bench](https://swebench.com).


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
