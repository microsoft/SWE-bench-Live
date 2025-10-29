# RepoLaunch Agent Tutorial

## Run RepoLaunch

Before getting started, please set your `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variable. We use [tavily](https://www.tavily.com/) for LLM search engine support.

We provide an example input file `test-dataset.jsonl` and a run config `test-config.json` in [examples](./examples) to help you quickly go through the launch process.

```shell
pip install -e .

export TAVILY_API_KEY=...

python -m launch.run --config-path test-config.json
```

For helpers to run RepoLaunch on Windows, see [Development-Windows.md](./Development-Windows.md)


## Input

For the input data used to set up the environment, we require the following fields:

| Field        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `repo`       | Full name of the repository like {user_name}/{project_name}                                                |
| `base_commit`| Commit to check out                                                         |
| `instance_id`| Unique identifier of the instance                                           |
| `language`   | Main language of the repo |
| `created_at` | (Optional) Creation time of the instance, used to support time-aware environment setup, useful in Python |
| `hints`      | (Optional)  Any hints for setting up the repo you want to give the agent, such as GitHub run checks info |


## Run Config

### Step 1 Setup
RepoLaunch is a two step process, the first step is to setup the repo, installing dependencies, build the repo and find test cases to test the build of the repo. The following configs are required.

| Field              | Type    |  Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `llm_provider_name`| string  |  Name of the LLM provider (e.g., "OpenAI", "AOAI")                          |
| `print_to_console` | boolean |  Whether to print logs to console                                           |
| `model_config`     | dict    |  Configuration for the LLM model (contains `model_name` and `temperature`)  |
| `workspace_root`   | string  |  Workspace folder for one run                                      |
| `dataset`          | string  |  Path to the dataset file                                                    |
| `instance_id`      | string  |  Specific instance ID to run, null to run all instances in the dataset      |
| `first_N_repos`    | integer |  Limit processing to first N repos (-1 for all repos)                       |
| `max_workers`      | integer |  Number of parallel workers for processing                                   |
| `overwrite`        | boolean |  Whether to overwrite existing results (false will skip existing repos)     |
| `os`               | str     |  which docker image os architecture to build on (default linux, can choose windows)   |
| `max_trials`       | integer |   how many rounds of setup-verify loop agent can attempt, default 1   |
| `max_steps_setup`  | integer |   how many steps agent can attemp to setup the environment, default 20   |
| `max_steps_verify` | integer |   how many steps agent can attemp to verify the setup, default 20   |
| `timeout`          | integer |   time limit of each round of setup, default 30 min   |
| `image_prefix`     | string  | prefix of the output_image in the format {namespace}/{dockerhub_repo}, defaults to repolaunch/dev |


### Step 2 Organize
RepoLaunch also provides a second optional step to 

1) Organize the commands to rebuild to repo after edits of the source code;
2) Organize the commands to test the repo with verbose testcase-status output, write a python script to parse the output into clean testcase-status mapping in JSON format:
    {
        "testcase1": "pass",
        "testcase2": "fail",
        "testcase3": "skip",
    };
3) Make best effort to find the command to run a single testcase separately.

The configs required for this step:


| Field              | Type    |  Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `mode`             | dict     |   default to {"setup": true, "organize": false}, set to {"setup": true, "organize": true} to do the two steps together, or set to {"setup": false, "organize": true} to do the second step separately AFTER the first step is DONE    |
| `max_steps_organize` | integer |   how many steps agent can attemp to organize the commands, default 20   |


## Output

The per-instance output will be saved in `{workspace_root}/playground/{instance_id}/result.json`.

### Step 1 Setup

| Field            | Description                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------|
| `instance_id`    | Unique identifier of the instance                                                                |
| `base_image`     | Docker base image                            |
| `docker_image`   | Commited Image                               |
| `setup_commands` | Records of shell commands used to set up the environment                                            |
| `test_commands`  | Records of shell commands used to run the tests with verbose output                                                 |
| `duration`       | Time taken to run the process (in minutes)         |
| `completed`      | Boolean indicating whether the execution completed successfully                                  |
| `exception`      | Error message or `null` if no exception occurred                                                 |

Summary would be saved to `{workspace_root}/setup.jsonl`

### Step 2 Organize

The `setup_commands` and `test_commands` of the first step would be noisy, with useless error commands and exploration commands. This is why we design the second step. The second step output would add these fields:

| Field            | Description                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------|
| `organize_duration`       | Time taken to run the process (in minutes)         |
| `organize_completed`      | Boolean indicating whether the organization attempt completed successfully                                  |
| `rebuild_commands`    | Minimal commands to rebuild the repo instance                                                                |
| `test_commands`     | Clean test commands                            |
| `parse`   | python script to parse the test output intp testcase-status mapping                               |
| `test_status` | Parsed testcase-status mapping in JSON                                         |
| `pertest_command` | Command to specify a testcase to run, might do not exists                                         |


Summary would be saved to `{workspace_root}/organize.jsonl`

## Helper scripts

### To use the output parser to parse test output:

```python
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser

# load an instance from organize.jsonl

container = SetupRuntime.from_launch_image(instance["docker_image"], instance["instance_id"])
testlog = container.send_command(";".join(instance["test_commands"])).output
status = run_parser(instance["parser"], testlog)

print(status)
# {"testcase1": "pass", "testcase2": "fail", "testcase3": "skip"}

container.clean_up()
```

### To evaluate the effect of a new diff patch:

```python
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser

# load an instance from organize.jsonl
# load your diff_patch

container = SetupRuntime.from_launch_image(instance["docker_image"], instance["instance_id"])
container.send_command(f"""git apply - <<'NEW_PATCH'\n{diff_patch}\nNEW_PATCH""")
# for windows powershell: container.send_command(f"""git apply - <<@"{diff_patch}"@""")
container.send_command(";".join(instance["rebuild_cmd"]))
after_patch_testlog = container.send_command(";".join(instance["test_commands"])).output
after_patch_status = run_parser(instance["log_parser"], testlog)

# if you need to save the changes
# container.send_command("git commit -m 'apply new patch'")
# container.commit(image_name="experiment", tag="1")
container.cleanup()
```

### If launch was interrupted, you can collect summary manually

```bash
python -m launch.scripts.collect\
    --workspace  data/test1  --step setup  # or organize
```

### To upload the result to dockerhub

```bash
docker login

python -m launch.scripts.upload_docker\
    --dataset  data/test1/organize.jsonl\
    --clear_after_push 0 # 0 for false and 1 for true
```
