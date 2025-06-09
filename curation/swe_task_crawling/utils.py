from __future__ import annotations

import logging
import re
import requests
import time
import urllib3

from dateutil import parser
from bs4 import BeautifulSoup
from typing import Callable, Iterator, Optional
from unidiff import PatchSet
from urllib.error import URLError
from http.client import RemoteDisconnected

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
        self.api_url = 'https://api.github.com/graphql'
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.full_name = f"{self.owner}/{self.name}"

    def call_api(self, query: str, variables: dict = None, max_retries: int = 10) -> dict|None:
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
                response = requests.post(self.api_url, json={'query': query, 'variables': variables}, headers=self.headers)
                if response.status_code == 200:
                    response_json = response.json()
                    rl = int(response.headers.get('x-ratelimit-remaining'))
                    if rl < 1000:
                        print(f"❗️ Waiting for 10 minutes, remaining calls for token {self.token[:20]}****: {rl}")
                        time.sleep(60 * 10)
                    if rl < 100:
                        print(f"❗️ Waiting for 1 hour, remaining calls for token {self.token[:20]}****: {rl}")
                        time.sleep(60 * 60)
                    if "data" in response_json:
                        return response
                    else:
                        raise Exception(f"GraphQL Query failed to return data: {response_json}")
                elif response.status_code == 403:
                    while True:
                        rl = response.headers.get('x-ratelimit-remaining')
                        logger.error(f"Got 403 error for token {self.token[:20]}****, wait for 5 minutes")
                        if rl > 0:
                            break
                        time.sleep(60 * 5)
                else:
                    raise Exception(f"GraphQL Query failed to run with status code {response.status_code} for token {self.token[:20]}****. {response.json()}")
            except (requests.exceptions.RequestException, URLError, RemoteDisconnected, urllib3.exceptions.MaxRetryError, requests.exceptions.ConnectTimeout) as e:
                print(f"❗️ 📢 Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(20)
                    attempt += 1
                else:
                    print(f"❗️ 📢 Still got connection error after {max_retries} attempts")
                    return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    while True:
                        rl = int(response.headers.get('x-ratelimit-remaining'))
                        logger.error(f"Got 403 error for token {self.token[:20]}****, wait for 5 minutes")
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
                logger.debug(f"Request variables: {variables}")  # Log the request variables
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
                    rl = int(response.headers.get('x-ratelimit-remaining'))
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
                    rl = int(response.headers.get('x-ratelimit-remaining'))
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
        return self.get_all_loop(query, variables, ["data", "repository", "pullRequest", "commits"], quiet=quiet)

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
        return self.get_all_loop(query, variables, ["data", "repository", "issue", "comments"], quiet=quiet)

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


def extract_problem_statement_and_hints(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Extract problem statement from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    if repo.name == "django":
        return extract_problem_statement_and_hints_django(pull, repo)

    problem_text = ""   # issue title and body
    hint_text = ""      # issue discussions (cutoff at first commit)
    all_hint_text = ""  # all issue discussions

    for issue_number in pull["resolved_issues"]:
        issue = repo.get_issue(issue_number)
        if issue is None:
            continue
        issue = issue.json()["data"]

        title = issue["repository"]["issue"]["title"] if issue["repository"]["issue"]["title"] else ""
        body = issue["repository"]["issue"]["body"] if issue["repository"]["issue"]["body"] else ""
        problem_text += f"{title}\n{body}\n"

        issue_number = issue["repository"]["issue"]["number"]
        single_hint, single_all_hint, commit_urls = _extract_hints(pull, repo, issue_number)
        hint_text += (single_hint + "\n\n")
        all_hint_text += (single_all_hint + "\n\n")

    return problem_text, hint_text, all_hint_text, commit_urls


def _extract_hints(pull: dict, repo: Repo, issue_number: int) -> list[str]:
    """
    Extract hints from comments associated with a pull request (before first commit)

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
        issue_number (int): issue number
    Return:
        issue_hint_comments: issue comments (cutoff at first commit)
        issue_all_comments: issue comments
        commit_urls: list of commit urls
    """
    # Get all commits in PR
    commits = repo.get_pull_commits(pull["number"])

    commits = list(commits)
    if len(commits) == 0:
        # If there are no comments, return no hints
        return "", "", []

    # Get time of first commit in PR
    commit_time = commits[0]["commit"]["author"]["date"]  # str
    commit_time = parser.parse(commit_time).timestamp()

    # Get commit urls
    commit_urls = []
    for commit in commits:
        commit_urls.append(commit["commit"]["url"])

    # Get all comments in issue
    all_issue_comments = repo.get_issue_comments(issue_number)
    all_issue_comments = list(all_issue_comments)
    # Iterate through all comments, only keep comments created before first commit
    issue_hint_comments = list()
    issue_all_comments = list()
    for comment in all_issue_comments:
        comment_time = time.mktime(
            time.strptime(comment["updatedAt"], "%Y-%m-%dT%H:%M:%SZ")
        )  # use updated_at instead of created_at
        if comment_time < commit_time:
            # only include information available before the first commit was created
            issue_hint_comments.append(comment)
        issue_all_comments.append(comment)
    assert len(issue_hint_comments) <= len(issue_all_comments)

    # Keep text from comments
    issue_hint_comments = "\n".join([comment["body"] for comment in issue_hint_comments])
    issue_all_comments = "\n".join([comment["body"] for comment in issue_all_comments])
    # return issue_hint_comments, issue_all_comments
    return issue_hint_comments, issue_all_comments, commit_urls


def wrapped_requests_get(url: str, max_retries: int = 10) -> requests.Response:
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url)
            return response
        except requests.exceptions.HTTPError as e:
            logger.info(f"Resource not found: {e}")
            return None
        except (requests.exceptions.RequestException, URLError, RemoteDisconnected, \
            urllib3.exceptions.MaxRetryError, requests.exceptions.ConnectTimeout)  as e:
            print(f"❗️ 📢 Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(20)
                attempt += 1
            else:
                print(f"❗️ 📢 Still got connection error after {max_retries} attempts")
                return None

def extract_patches(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Get patch and test patch from PR

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """
    # patch = requests.get(pull["diff_url"]).text
    patch = wrapped_requests_get(pull["diff_url"]).text
    patch_test = ""
    patch_fix  = ""
    for hunk in PatchSet(patch):
        if any(
            test_word in hunk.path for test_word in
            ['test', 'tests', 'e2e', 'testing']
        ):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix, patch_test


### MARK: Repo Specific Parsing Functions ###
def extract_problem_statement_and_hints_django(
    pull: dict, repo: Repo
) -> tuple[str, list[str]]:
    """
    Get problem statement and hints from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    text = ""
    all_hints_text = list()
    for issue_number in pull["resolved_issues"]:
        url = f"https://code.djangoproject.com/ticket/{issue_number}"
        # resp = requests.get(url)
        resp = wrapped_requests_get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get problem statement (title + body)
        issue_desc = soup.find("div", {"id": "ticket"})
        title = issue_desc.find("h1", class_="searchable").get_text()
        title = re.sub(r"\s+", " ", title).strip()
        body = issue_desc.find("div", class_="description").get_text()
        body = re.sub(r"\n+", "\n", body)
        body = re.sub(r"    ", "\t", body)
        body = re.sub(r"[ ]{2,}", " ", body).strip()
        text += f"{title}\n{body}\n"

        # Get time of first commit in PR
        commits = repo.get_all_loop(
            repo.api.pulls.list_commits, pull_number=pull["number"]
        )
        commits = list(commits)
        if len(commits) == 0:
            continue
        commit_time = commits[0].commit.author.date
        commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))

        # Get all comments before first commit
        comments_html = soup.find("div", {"id": "changelog"})
        div_blocks = comments_html.find_all("div", class_="change")
        # Loop through each div block
        for div_block in div_blocks:
            # Find the comment text and timestamp
            comment_resp = div_block.find("div", class_="comment")
            timestamp_resp = div_block.find("a", class_="timeline")
            if comment_resp is None or timestamp_resp is None:
                continue

            comment_text = re.sub(r"\s+", " ", comment_resp.text).strip()
            timestamp = timestamp_resp["title"]
            if timestamp.startswith("See timeline at "):
                timestamp = timestamp[len("See timeline at ") :]
            if "/" in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%m/%d/%y %H:%M:%S"))
            elif "," in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%b %d, %Y, %I:%M:%S %p"))
            else:
                raise ValueError(f"Timestamp format not recognized: {timestamp}")

            # Append the comment and timestamp as a tuple to the comments list
            if timestamp < commit_time:
                all_hints_text.append((comment_text, timestamp))

    return text, all_hints_text
