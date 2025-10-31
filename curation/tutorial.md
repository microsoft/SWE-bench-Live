# Data Curation of SWE-bench-Live

This tutorial walks you through how to automatically curate new issue-resolving tasks from real GitHub issues.

## Setup

Dependencies: python, git, docker

```shell
pip install -e .
pip install -e launch/.
```

## Repositories Crawling

This step crawls the initial source repo list, from which we find issues. You should prepare GitHub tokens in advance to unlock the API rate limit.

1. Crawl raw repositories within a given star range, supporting multiple tokens for higher rate limits

```shell
cd curation
mkdir -p output

# max_stars is optional
python crawl_repo.py \
    --language Python \
    --min_stars 10000 \
    --max_stars 100000 \
    --tokens_file tokens.txt \
    --output_file output/raw_repos.jsonl
```

2. Filter the crawled raw repositories based on some predefined quality control-related criteria.

```shell
# More than 200 pulls and issues
# More than 200 forks
# The percentage of main language code should be more than 60%
python filter_repo.py \
    --input_file output/raw_repos.jsonl \
    --output_file output/filtered_repos.jsonl \
    --tokens_file tokens.txt \
    --language Python \
     --max_workers 20
```

## Issue-PR Pairs Crawling

This step crawls Issue-PR pairs created after the cut-off date from the given repositories, and converts them into SWE-bench-like task instances.

```shell
mkdir -p job_status

./swe_task_crawling/run_get_tasks_pipeline.sh \
    --repos-jsonl output/filtered_repos.jsonl \
    --token-file tokens.txt \
    --cutoff-date 20250501 \
    --path-prs output/prs \
    --path-tasks output/tasks \
    --output-dir output/split_jobs

python swe_task_crawling/merge_tasks.py \
    --input_folder output/tasks \
    --input_repos output/filtered_repos.jsonl \
    --output output/raw_tasks.jsonl

```

## Execution Env Setup with `RepoLaunch`

Next, we will use `RepoLaunch` to attempt to create an execution environment for each task instance to support test execution.

```shell
cd ../launch
```

Create a run config for RepoLaunch and save it in `config.json`:
```json
{
    "llm_provider_name": "OpenAI",
    "model_config": {        
        "model_name": "gpt-4.1",
        "temperature": 0.0
    },
    "workspace_root": "playground/tutorial-run/",
    "dataset": "../curation/output/raw_tasks.jsonl",
    "print_to_console": false,
    "max_workers": 5,
    "overwrite": false
}
```

Prepare your llm API Key.

```shell
export OPENAI_API_KEY=...

export TAVILY_API_KEY=...
```

Fire your RepoLaunch run!
```shell
# recommended with tmux/nohup, it takes long time
python -m launch.run --config-path config.json
```
In RepoLaunch step, each instance that is successfully set up will be committed to a Docker image, with `starryzhang` as the default namespace. An example image key: `starryzhang/sweb.eval.x86_64.streamlink_1776_streamlink-6535`. The image name part (`sweb.eval.*`) follows the same naming convention as SWE-bench.

<blockquote style="border-left: 4px solid #3498db; background: #f4faff; padding: 0.75em;">

Note: Some instances would require many file descriptors. If you see "too many files open error", try
```shell
ulimit -a
ulimit -n 32768
```

</blockquote>

Export successfully set up instances to pre-validated SWE-bench-Live instances file:
```shell
python to_swebench.py \
    --playground playground/tutorial-run \
    --output_jsonl ../curation/output/pre-validated-instances.jsonl
```

## Validation

In this step we apply gold patches to instances, run test cases, and get `FAIL_TO_PASS` and `PASS_TO_PASS` test cases for each instance.

```shell
# cd in repo root
cd ..

python -m swebench.harness.run_validation \
    --dataset_name curation/output/pre-validated-instances.jsonl \
    --predictions_path gold \
    --max_workers 10 \
    --run_id tutorial-validation \
    --namespace starryzhang
```

## Production

This step writes valid instances with both `FAIL_TO_PASS` and `PASS_TO_PASS` test cases to final dataset.

```shell
python swebench/collect/produce/make_full.py \
    --input-dir logs/run_evaluation/tutorial-validation/gold \
    --output-dir datasets

# If you want to merge with old data we published
python swebench/collect/produce/merge_with_old.py \
    --input-dir "datasets/full-{today}.jsonl"

python swebench/collect/produce/make_lite.py \
    --start-month 2024-12 --end-month 2025-05 
    # If you want to control month range to sample from

python -m swebench.collect.produce.make_verified \
    --start-month 2024-06 --end-month 2025-05 \
    --provider OpenAI --model o3-20250416 
    # Optional --input-file datasets/full-{date}.jsonl
```

The default output files are: 

- `datasets/full-{today}.jsonl`
- `datasets/lite-{today}.jsonl`
- `datasets/verified-{today}.jsonl` & `datasets/verified-log-{today}.jsonl` (LLM's reasons to filter)

where `{today}` is the current date in ISO format (e.g., `2025-01-15`).

To quickly check whether all instances can be solved by the gold patches (usually they do), run

```shell
python -m swebench.harness.run_evaluation \
  --dataset_name datasets/full-{today}.jsonl \
  --split full \
  --predictions_path gold \
  --run_id tutorial-validation \
  --rewrite_reports true
```
