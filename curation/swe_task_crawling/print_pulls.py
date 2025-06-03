#!/usr/bin/env python3

"""Given the `<owner/name>` of a GitHub repo, this script writes the raw information for all the repo's PRs to a single `.jsonl` file."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from fastcore.xtras import obj2dict
from utils import Repo
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_all_pulls(
        repo: Repo,
        output: str,
        max_pulls: int = None,
        cutoff_date: str = None,
    ) -> None:
    """
    Iterate over all pull requests in a repository and log them to a file

    Args:
        repo (Repo): repository object
        output (str): output file name
    """
    cutoff_date = datetime.strptime(cutoff_date, "%Y%m%d") \
        .strftime("%Y-%m-%dT%H:%M:%SZ") \
        if cutoff_date is not None else None

    with open(output, "w") as file:
        for i_pull, pull in enumerate(repo.get_all_pulls()):
            setattr(pull, "resolved_issues", repo.extract_resolved_issues(pull))
            print(json.dumps(obj2dict(pull)), end="\n", flush=True, file=file)
            if max_pulls is not None and i_pull >= max_pulls:
                break
            if cutoff_date is not None and pull.created_at < cutoff_date:
                break

def log_selected_pulls(
        repo: str,
        output: str,
        pull2issue_file: str,
        token: str,
    ) -> None:
    """
    Log selected pull requests to a file
    """
    owner, repo_name = repo.split("/")
    repo_obj = Repo(owner, repo_name, token=token)
    pull2issue = dict()
    with open(pull2issue_file, "r") as fr:
        for line in fr:
            instance = json.loads(line)
            pull2issue[instance["pull"]] = instance["issue"]

    # Get crawled pulls
    seen_prs = set()
    if os.path.exists(output):
        with open(output, "r") as fr:
            for line in fr:
                instance = json.loads(line)
                seen_prs.add(instance["repository"]["pullRequest"]["number"])

    # Write new pulls
    write_mode = "w" if not os.path.exists(output) else "a"
    with open(output, write_mode) as file:
        for pull_number, issue_numbers in pull2issue.items():
            if pull_number in seen_prs:
                continue
            pull = repo_obj.get_pull(pull_number)
            if pull is not None:
                pull = pull.json()["data"]
                # Reformat the pull object to match the following original swebench processing logic
                try:
                    pull.update({
                        "base": {
                            "repo": {
                                "full_name": pull["repository"]["pullRequest"]["baseRepository"]["nameWithOwner"]
                            },
                            "sha": pull["repository"]["pullRequest"]["baseRefOid"]
                        },
                        "diff_url": pull["repository"]["pullRequest"]["url"] + ".diff",
                        "number": pull["repository"]["pullRequest"]["number"],
                        "created_at": pull["repository"]["pullRequest"]["createdAt"],
                        "merged_at": pull["repository"]["pullRequest"]["mergedAt"],
                        "resolved_issues": issue_numbers
                    })
                    print(json.dumps(obj2dict(pull)), end="\n", flush=True, file=file)
                except Exception as e:
                    logger.error(f"[{owner}/{repo_name}] Error processing pull {pull_number}: {e}")


def main(
        repo_name: str,
        output: str,
        pull2issue_file:str,
        token: Optional[str] = None,
    ):
    """
    Logic for logging all pull requests in a repository

    Args:
        repo_name (str): name of the repository
        output (str): output file name
        token (str, optional): GitHub token
    """
    if token is None:
        token = os.environ.get("GH_TOKEN")
    owner, repo = repo_name.split("/")
    repo = Repo(owner, repo, token=token)
    log_selected_pulls(repo_name, output, pull2issue_file, token)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--repo_name", default="keras-team/keras-contrib",type=str, help="Name of the repository")
    # parser.add_argument("--output",default="keras-team__keras-contrib-selected_pulls.jsonl", type=str, help="Output file name")
    # parser.add_argument("--token", type=str,default=os.getenv("GH_TOKEN"), help="GitHub token")
    # parser.add_argument("--pull2issue_file", type=str,default="keras-contrib-pull2issue-20180101.jsonl", help="Pull to issue mapping file")
    # args = parser.parse_args()
    # main(**vars(args))
    main("keras-team/keras-contrib", "keras-team__keras-contrib-selected_pulls.jsonl", "keras-contrib-pull2issue-20180101.jsonl", os.getenv("GH_TOKEN"))
