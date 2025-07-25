<p align="center">
  <a href="http://swe-bench-live.github.io">
    <img src="../assets/banner.png" style="height: 10em" alt="SWE-bench-Live" />
  </a>
</p>

# `🚀 RepoLaunch`

*Turning Any Codebase into Testable Sandbox Environment*

RepoLaunch addresses the bottleneck of setting up execution environments by automating the process through an LLM-based agentic workflow.

## Launch Environment
Before getting started, please set your `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variable. We use [tavily](https://www.tavily.com/) for LLM search engine support.

We provide an example input file `test-dataset.jsonl` and a run config `test-config.json` to help you quickly go through the launch process.

```shell
cd launch
pip install -e .

python -m launch.run --config-path test-config.json
```

## Data Schema

### Input

For the input data used to set up the environment, we require the following fields:

| Field        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `repo`       | Full name of the repository                                                 |
| `base_commit`| Commit to check out                                                         |
| `instance_id`| Unique identifier of the instance                                           |
| `language`   | Main language of the repo |
| `created_at` | (Optional) Creation time of the instance, used to support time-aware environment setup |

### Run Config

For the run configuration file, the following fields are supported:

| Field              | Type    |  Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `llm_provider_name`| string  |  Name of the LLM provider (e.g., "OpenAI", "AOAI")                          |
| `print_to_console` | boolean |  Whether to print logs to console                                           |
| `model_config`     | dict  |  Configuration for the LLM model (contains `model_name` and `temperature`)  |
| `workspace_root`   | string  |  Workspace folder for one run                                      |
| `dataset`          | string  |  Path to the dataset file                                                    |
| `instance_id`      | string  |  Specific instance ID to run, null to run all instances in the dataset      |
| `first_N_repos`    | integer |  Limit processing to first N repos (-1 for all repos)                       |
| `max_workers`      | integer |  Number of parallel workers for processing                                   |
| `overwrite`        | boolean |  Whether to overwrite existing results (false will skip existing repos)     |

### Output

The output will be saved in `{playground_folder}/{instance_id}/result.json` and follows the structure below:

| Field            | Description                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------|
| `instance_id`    | Unique identifier of the instance                                                                |
| `base_image`     | Docker base image                            |
| `setup_commands` | List of shell commands used to set up the environment                                            |
| `test_commands`  | List of shell commands used to run the tests                                                     |
| `duration`       | Time taken to run the process (in minutes)         |
| `completed`      | Boolean indicating whether the execution completed successfully                                  |
| `exception`      | Error message or `null` if no exception occurred                                                 |