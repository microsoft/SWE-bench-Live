# Data Curation of SWE-bench-Live

This tutorial walks you through how to automatically curate new issue-resolving tasks from real GitHub issues.

## Setup

Dependencies: python, git, docker

```shell
docker ps # make sure docker is running and you have privilage
git clone --recursive https://github.com/microsoft/SWE-bench-Live
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

## Simple Filtering based on LLM Judge


### Verify the quality of instances

This step basically follows the idea of SWE-bench-Verified to filter instance with

1. Vague problem statements;
2. Test patches with requirements not required in problem statements;
3. Answer in problem statement.

Prepare your llm API Key.

```shell
export OPENAI_API_KEY=...

python -m llm_filter.verify \
    --input_dir output/raw_tasks.jsonl \
    --output_dir  output/verified_tasks.jsonl \
    --llm_provider  AOAI \
    --model_name   gpt-5-20250807
```

### Split windows-specific tasks VS general tasks

```shell
python -m llm_filter.split_os \
    --input_file output/raw_tasks.jsonl  \
    --windows_file  output/windows_tasks.jsonl \
    --general_file  output/general_tasks.jsonl \
    --llm_provider  AOAI \
    --model_name   gpt-5-20250807
```

## Execution Env Setup with `RepoLaunch`

Next, we will use `RepoLaunch` to attempt to create an execution environment for each task instance to support test execution.

Create a run config for RepoLaunch and save it in `launch/data/your_experiment/config.json`. The example config.json in `launch/data/examples` is:
```json
{
    "mode": {
        "setup": true,
        "organize": true
    },
    "llm_provider_name": "OpenAI",
    "model_config": {        
        "model_name": "gpt-4.1-20250414"
    },
    "workspace_root": "data/examples/",
    "dataset": "data/examples/dataset.jsonl",
    "print_to_console": false,
    "first_N_repos": -1,
    "overwrite": false,
    "max_workers": 5,
    "os": "linux",
    "max_trials": 2,
    "max_steps_setup": 60,
    "max_steps_verify": 20,
    "max_steps_organize": 40,
    "cmd_timeout": 60,
    "image_prefix": "repolaunch/dev"
}
```

Prepare your llm API Key.

```shell
export OPENAI_API_KEY=...

export TAVILY_API_KEY=...
```

Fire your RepoLaunch run!
```shell
cd ../launch

# recommended in a tmux session, it takes long time
python -m launch.run --config-path data/your_experiment/config.json
```

<blockquote style="border-left: 4px solid #3498db; background: #f4faff; padding: 0.75em;">

Note: Some instances would require many file descriptors. If you see "too many files open error", try
```shell
ulimit -a
ulimit -n 32768
```
</blockquote>

<blockquote style="border-left: 4px solid #3498db; background: #f4faff; padding: 0.75em;">
Note: We observe that as the execution becomes very long, the docker response (docker run container; docker commit and docker remove container) would become lower and lower and even return None. 
In this case:

```shell
stop running launch
restart docker
docker container prune
start running launch again
```

</blockquote>


## Validation

In this step we apply gold patches to instances, run test cases, and get `FAIL_TO_PASS` and `PASS_TO_PASS` test cases for each instance.

```shell
# cd in repo root
cd ../

# For Windows system if there are decoding issues: $env:PYTHONUTF8="1" ; $env:PYTHONIOENCODING="utf-8"

# Get Fail to Pass
# apply test_patch -> build -> apply gold_patch -> build
# test is run 3 times automatically in validation.py to filter flaky instances
python -m  evaluation.validation \
    --input_dir launch/data/examples/organize.jsonl \
    --output_dir logs/val \
    --platform  linux \#or windows
    --workers  4 \
    --overwrite  0 # or 1 for yes

# filter instances that fail when only apply test_patch -> apply gold_patch -> build
python -m  evaluation.evaluation \
    --dataset logs/val/validated_instances.jsonl \
    --output_dir logs/eval \
    --patch_dir gold \
    --platform  linux \#or windows
    --workers  4 \
    --overwrite  0 # or 1 for yes
```

Result is saved to `logs/eval/gold_patch_evaluated_instances.jsonl`.


## For New Task Instance Contributors

The demo to upload dataset to huggingface is 

```bash
cd curation
hf auth login
python push_dataset/push_multilang.py
```

The demo to upload docker image to dockerhub is 

```bash
cd launch
docker login
python -m launch.scripts.upload_docker --dataset ... --clear_after_push 0
```
