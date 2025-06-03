#!/usr/bin/env python3

"""Script to collect pull requests and convert them to candidate task instances"""

import argparse
import os
import traceback

from dotenv import load_dotenv
from multiprocessing import Pool
from build_dataset import main as build_dataset
from print_pulls import log_selected_pulls
from fetch_pulls import fetch_pulls
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def split_instances(input_list: list, n: int) -> list:
    """
    Split a list into n approximately equal length sublists

    Args:
        input_list (list): List to split
        n (int): Number of sublists to split into
    Returns:
        result (list): List of sublists
    """
    avg_length = len(input_list) // n
    remainder = len(input_list) % n
    result, start = [], 0

    for i in range(n):
        length = avg_length + 1 if i < remainder else avg_length
        sublist = input_list[start : start + length]
        result.append(sublist)
        start += length

    return result


def construct_data_files(data: dict):
    """
    Logic for combining multiple .all PR files into a single fine tuning dataset

    Args:
        data (dict): Dictionary containing the following keys:
            repos (list): List of repositories to retrieve instruction data for
            path_prs (str): Path to save PR data files to
            path_tasks (str): Path to save task instance data files to
            token (str): GitHub token to use for API requests
    """
    repos, path_prs, path_tasks, max_pulls, cutoff_date, token = (
        data["repos"],
        data["path_prs"],
        data["path_tasks"],
        data["max_pulls"],
        data["cutoff_date"],
        data["token"],
    )
    issue_first = True
    for repo in repos:
        repo = repo.strip(",").strip()
        repo_name = repo.split("/")[1]
        try:
            logger.info(f"[{repo}] 🎾 Start the job using token {token[:20]}****")
            path_pr = os.path.join(path_prs, f"{repo_name}-prs.jsonl")
            if cutoff_date:
                path_pr = path_pr.replace(".jsonl", f"-{cutoff_date}.jsonl")

            assert issue_first, "Only issue-first approach is supported now"

            # Get pull2issue
            path_pull2issue = os.path.join(path_prs, f"{repo_name}-pull2issue-{cutoff_date}.jsonl")
            if os.path.exists(path_pull2issue):
                # and os.path.getsize(path_pull2issue) == 0:
                # Check whether path_pull2issue is a fake empy file
                # logger.info(f"[{repo}] ❌ Found no issues closed by a pull, skipping...")
                logger.info(f"[{repo}] 📁 pull2issue file already exists, skipping...")
            else:
                fetch_pulls(repo, token, path_prs, cutoff_date)

            # Get selected pulls
            log_selected_pulls(repo, path_pr, path_pull2issue, token)
            logger.info(f"[{repo}] ✅ Successfully saved PR data to {path_pr} using issue-first approach")

            # Build task instances
            path_task = os.path.join(path_tasks, f"{repo_name}-task-instances.jsonl")
            build_dataset(path_pr, path_task, token)
            logger.info(f"[{repo}] ✅ Successfully saved task instance data to {path_task}")

        except Exception as e:
            print("-"*80)
            print(f"Something went wrong for {repo}, skipping: {e}")
            print("Here is the full traceback:")
            traceback.print_exc()
            print("-"*80)


def main(
        repos: list,
        repo_list_file: str,
        gh_token_file: str,
        token_ids: list,
        path_prs: str,
        path_tasks: str,
        max_pulls: int = None,
        cutoff_date: str = None,
    ):
    """
    Spawns multiple threads given multiple GitHub tokens for collecting fine tuning data

    Args:
        repos (list): List of repositories to retrieve instruction data for
        path_prs (str): Path to save PR data files to
        path_tasks (str): Path to save task instance data files to
        cutoff_date (str): Cutoff date for PRs to consider in format YYYYMMDD
    """
    if repos is None and repo_list_file:
        repos = list()
        with open(repo_list_file, "r") as f:
            for line in f:
                repos.append(line.strip())

    if not repos:
        raise Exception("Repositories list is empty. Please specify repos using --repos or --repo_list_file.")

    path_prs, path_tasks = os.path.abspath(path_prs), os.path.abspath(path_tasks)
    print(f"Will save PR data to {path_prs}")
    print(f"Will save task instance data to {path_tasks}")
    print(f"Received following number of repos to create task instances for: {len(repos)}")

    # List of GitHub API tokens
    with open(gh_token_file, 'r') as f:
        all_tokens = f.read().splitlines()
    print(f"Found {len(all_tokens)} tokens in {gh_token_file}")
    if token_ids:
        tokens = []
        for token_id in token_ids:
            tokens.append(all_tokens[token_id])
    else:
        tokens = all_tokens
    print(f"Will use {len(tokens)} tokens for creating task instances.")

    if not tokens:
        raise Exception("Missing GITHUB_TOKENS, consider rerunning with GITHUB_TOKENS=$(gh auth token)")
    # tokens = tokens.split(",")
    data_task_lists = split_instances(repos, len(tokens))

    data_pooled = [
        {
            "repos": repos,
            "path_prs": path_prs,
            "path_tasks": path_tasks,
            "max_pulls": max_pulls,
            "cutoff_date": cutoff_date,
            "token": token
        }
        for repos, token in zip(data_task_lists, tokens)
    ]

    with Pool(len(tokens)) as p:
        p.map(construct_data_files, data_pooled)
    
    for token_id in token_ids:
        with open(f"job_status/{token_id}.completion", "w") as f:
            f.write("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos", nargs="+", help="List of repositories (e.g., `sqlfluff/sqlfluff`) to create task instances for"
    )
    parser.add_argument(
        "--repo_list_file", type=str, help=""
    )
    parser.add_argument(
        "--gh_token_file", type=str, help=""
    )
    parser.add_argument("--token_ids", nargs="+", type=int, required=True, help="List of token IDs to use")
    parser.add_argument(
        "--path_prs", type=str, help="Path to folder to save PR data files to"
    )
    parser.add_argument(
        "--path_tasks",
        type=str,
        help="Path to folder to save task instance data files to",
    )
    parser.add_argument(
        "--max_pulls",
        type=int,
        help="Maximum number of pulls to log",
        default=None
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        help="Cutoff date for PRs to consider in format YYYYMMDD",
        default=None,
    )
    args = parser.parse_args()
    main(**vars(args))
