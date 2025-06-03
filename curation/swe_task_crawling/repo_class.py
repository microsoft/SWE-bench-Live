import logging
import re
import time
from http.client import RemoteDisconnected
from typing import Iterator, Optional
from urllib.error import URLError

import requests
import os
import urllib3
from dateutil import parser
from fastcore.net import HTTP403ForbiddenError, HTTP404NotFoundError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Repo:
    def __init__(self, owner: str, name: str, token: Optional[str] = None):
        """
        Init to retrieve target repository and create ghapi tool

        Args:
            owner (str): owner of target repository
            name (str): name of target repository
            token (str): github token
        """
        self.owner = owner
        self.name = name
        self.token = token
        self.api_url = "https://api.github.com/graphql"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.full_name = f"{self.owner}/{self.name}"

    def call_api(
        self, query: str, variables: dict = None, max_retries: int = 10
    ) -> dict | None:
        """
        API call wrapper with rate limit handling (checks every 5 minutes if rate limit is reset)

        Args:
            query (str): GraphQL query string
            variables (dict): variables for the GraphQL query
        Return:
            values (dict): response object of `query`
        """
        attempt = 0
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    json={"query": query, "variables": variables},
                    headers=self.headers,
                )
                if response.status_code == 200:
                    response_json = response.json()
                    rl = int(response.headers.get("x-ratelimit-remaining"))
                    if rl < 1000:
                        print(
                            f"❗️ Waiting for 10 minutes, remaining calls for token {self.token[:20]}****: {rl}"
                        )
                        time.sleep(60 * 10)
                    if rl < 100:
                        print(
                            f"❗️ Waiting for 1 hour, remaining calls for token {self.token[:20]}****: {rl}"
                        )
                        time.sleep(60 * 60)
                    if "data" in response_json:
                        return response
                    else:
                        raise Exception(
                            f"GraphQL Query failed to return data: {response_json}"
                        )
                elif response.status_code == 403:
                    while True:
                        rl = response.headers.get("x-ratelimit-remaining")
                        logger.error(
                            f"Got 403 error for token {self.token[:20]}****, wait for 5 minutes"
                        )
                        if rl > 0:
                            break
                        time.sleep(60 * 5)
                else:
                    raise Exception(
                        f"GraphQL Query failed to run with status code {response.status_code} for token {self.token[:20]}****. {response.json()}"
                    )
            except (
                requests.exceptions.RequestException,
                URLError,
                RemoteDisconnected,
                urllib3.exceptions.MaxRetryError,
                requests.exceptions.ConnectTimeout,
            ) as e:
                print(f"❗️ 📢 Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(20)
                    attempt += 1
                else:
                    print(
                        f"❗️ 📢 Still got connection error after {max_retries} attempts"
                    )
                    return None
            except HTTP403ForbiddenError as e:
                while True:
                    rl = int(response.headers.get("x-ratelimit-remaining"))
                    logger.error(
                        f"Got 403 error for token {self.token[:20]}****, wait for 5 minutes"
                    )
                    if rl > 0:
                        break
                    time.sleep(60 * 5)

    def get_all_loop(
        self,
        query: str,
        variables: dict,
        data_path: list[str],
        per_page: int = 100,
        num_pages: Optional[int] = None,
        quiet: bool = True,
    ) -> Iterator:
        """
        Return all values from a paginated API endpoint.

        Args:
            query (str): GraphQL query string
            variables (dict): variables for the GraphQL query
            per_page (int): number of values to return per page
            num_pages (int): number of pages to return
            quiet (bool): whether to print progress
        """
        variables.update({"per_page": per_page, "after": None})
        page = 1
        while True:
            try:
                logger.debug(
                    f"Request variables: {variables}"
                )  # Log the request variables
                response = self.call_api(query, variables)
                data = response.json()
                for key in data_path:
                    data = data[key]
                values = data["nodes"]
                logger.debug(f"Page {page} values: {values}")  # Log the values returned
                if not values:
                    break
                yield from values
                if not quiet:
                    rl = int(response.headers.get("x-ratelimit-remaining"))
                    logger.info(
                        f"[{self.owner}/{self.name}] Processed page {page} ({per_page} values per page). "
                        f"Remaining calls: {rl}"
                    )
                if num_pages is not None and page >= num_pages:
                    break
                if not data["pageInfo"]["hasNextPage"]:
                    break
                variables["after"] = data["pageInfo"]["endCursor"]
                page += 1
            except Exception as e:
                logger.error(
                    f"[{self.owner}/{self.name}] Error processing page {page} "
                    f"w/ token {self.token[:10]} - {e}"
                )
                while True:
                    rl = int(response.headers.get("x-ratelimit-remaining"))
                    if rl > 0:
                        break
                    logger.info(
                        f"[{self.owner}/{self.name}] Waiting for rate limit reset "
                        f"for token {self.token[:10]}, checking again in 5 minutes"
                    )
                    time.sleep(60 * 5)
        if not quiet:
            logger.info(
                f"[{self.owner}/{self.name}] Processed {(page-1)*per_page + len(values)} values"
            )

    def get_pull_commits(self, pull_number: int, quiet: bool = True) -> Iterator:
        """
        Get all commits for a pull request.
        """
        query = """
        query($owner: String!, $name: String!, $pull_number: Int!, $after: String) {
            repository(owner: $owner, name: $name) {
                pullRequest(number: $pull_number) {
                    commits(first: 100, after: $after) {
                        pageInfo {
                            endCursor
                            hasNextPage
                        }
                        nodes {
                            commit {
                                message
                                author {
                                    date
                                }
                                url
                            }
                        }
                    }
                }
            }
        }
        """
        variables = {
            "owner": self.owner,
            "name": self.name,
            "pull_number": pull_number,
            "after": None,
        }
        return self.get_all_loop(
            query,
            variables,
            ["data", "repository", "pullRequest", "commits"],
            quiet=quiet,
        )

    def get_issue_comments(self, issue_number: int, quiet: bool = True) -> Iterator:
        """
        Get all comments for an issue.
        """
        query = """
        query($owner: String!, $name: String!, $issue_number: Int!, $after: String) {
            repository(owner: $owner, name: $name) {
                issue(number: $issue_number) {
                    comments(first: 100, after: $after) {
                        pageInfo {
                            endCursor
                            hasNextPage
                        }
                        nodes {
                            body
                            updatedAt
                        }
                    }
                }
            }
        }
        """
        variables = {
            "owner": self.owner,
            "name": self.name,
            "issue_number": issue_number,
            "after": None,
        }
        return self.get_all_loop(
            query, variables, ["data", "repository", "issue", "comments"], quiet=quiet
        )

    def get_pull(self, pull_number: int) -> dict:
        """
        Wrapper for API call to get a single PR

        Args:
            pull_number (int): number of PR to return
        """
        query = """
        query($owner: String!, $name: String!, $pull_number: Int!) {
            repository(owner: $owner, name: $name) {
                pullRequest(number: $pull_number) {
                    number
                    title
                    body
                    baseRefName
                    baseRefOid
                    baseRepository {
                        nameWithOwner
                    }
                    url
                    createdAt
                    mergedAt
                }
            }
        }
        """
        variables = {
            "owner": self.owner,
            "name": self.name,
            "pull_number": pull_number,
        }
        return self.call_api(query, variables)

    def get_issue(self, issue_number: int) -> dict:
        """
        Wrapper for API call to get a single issue

        Args:
            issue_number (int): number of issue to return
        """
        query = """
        query($owner: String!, $name: String!, $issue_number: Int!) {
            repository(owner: $owner, name: $name) {
                issue(number: $issue_number) {
                    number
                    title
                    body
                }
            }
        }
        """
        variables = {
            "owner": self.owner,
            "name": self.name,
            "issue_number": issue_number,
        }
        return self.call_api(query, variables)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GitHub Repo API wrapper.")
    parser.add_argument("--owner", default="flow-project", help="Owner of the repository.")
    parser.add_argument("--name", default="flow", help="Repository name.")
    parser.add_argument("--token", default=os.getenv("GH_TOKEN"), help="GitHub personal access token.")
    parser.add_argument(
        "--pull", type=int,default=153, help="Pull request number to fetch commits."
    )
    parser.add_argument("--issue", type=int,default=290, help="Issue number to fetch comments.")

    args = parser.parse_args()

    repo = Repo(owner=args.owner, name=args.name, token=args.token)
    # if args.pull:
    #     pull = repo.get_pull(args.pull)
    #     print(pull)
    # if args.issue:
    #     issue = repo.get_issue(args.issue)
    #     print(issue)

    # if args.pull:
    #     logger.info(f"Fetching commits for pull request #{args.pull}...")
    #     commits = list(repo.get_pull_commits(args.pull))
    #     logger.info(f"Retrieved {len(commits)} commits.")
    #     for commit in commits:
    #         print(commit)
    

    if args.issue:
        logger.info(f"Fetching comments for issue #{args.issue}...")
        comments = list(repo.get_issue_comments(args.issue))
        logger.info(f"Retrieved {len(comments)} comments.")
        for comment in comments:
            print(comment)
