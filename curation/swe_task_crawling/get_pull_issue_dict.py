import argparse
import json
import logging
import os
import time
from datetime import datetime

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_graphql_query(query, token):
    def call_api(query, token):
        request = requests.post(
            "https://api.github.com/graphql",
            json={"query": query},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        if request.status_code == 200:
            response_json = request.json()
            rl = int(request.headers.get("x-ratelimit-remaining"))
            if rl < 1000:
                logger.warning(f"❗️ GraphQL Rate limit remaining: {rl}")
                
            if "data" in response_json:
                return response_json
            else:
                raise Exception(f"GraphQL Query failed to return data: {response_json}")
        elif request.status_code == 403:
            rl = request.headers.get("x-ratelimit-remaining")
            logger.error(f"GraphQL Rate limit remaining: {rl}")
            if rl == 0:
                reset_time = int(request.headers.get("x-ratelimit-reset"))
                sleep_time = max(0, reset_time - time.time()) + 1
            else:
                sleep_time = 60 * 5
            logger.error(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            return call_api(query, token)
        else:
            raise Exception(
                f"GraphQL Query failed to run with status code {request.status_code}. {request.json()}"
            )

    for i in range(3):
        try:
            return call_api(query, token)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(60)
    logger.error("retry failed, giving up...")
    return {}


def get_closed_issue_events(owner, repo_name, token):
    query_template = """
    {{
      repository(owner: "{owner}", name: "{repo_name}") {{
        issues(states: CLOSED, first: 100, after: {after}) {{
          pageInfo {{
            endCursor
            hasNextPage
          }}
          edges {{
            node {{
              number
              title
              body
              timelineItems(last: 100) {{
                nodes {{
                  __typename
                  ... on ClosedEvent {{
                    actor {{
                      login
                    }}
                    createdAt
                    closer {{
                      __typename
                      ... on PullRequest {{
                        number
                        title
                      }}
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

    after_cursor = "null"
    issues = []

    while True:
        query = query_template.format(
            owner=owner, repo_name=repo_name, after=after_cursor
        )
        result = run_graphql_query(query, token)
        if not result:
            break
        issues_data = result["data"]["repository"]["issues"]
        issues.extend(issues_data["edges"])

        if issues_data["pageInfo"]["hasNextPage"]:
            after_cursor = f'"{issues_data["pageInfo"]["endCursor"]}"'
        else:
            break

    return issues


def collect_closed_issues(owner, repo_name, token, path_issue, cutoff_date):
    cutoff_date = datetime.strptime(cutoff_date, "%Y%m%d")

    existing_issues = set()
    if os.path.exists(path_issue):
        with open(path_issue, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_issues.add(instance["issue_number"])

    closed_issues = get_closed_issue_events(owner, repo_name, token)

    for issue in closed_issues:
        issue_node = issue["node"]
        issue_number = issue_node["number"]

        if issue_number in existing_issues:
            continue

        for event in issue_node["timelineItems"]["nodes"]:
            if (
                event["__typename"] == "ClosedEvent"
                and event.get("closer")
                and event["closer"]["__typename"] == "PullRequest"
            ):
                pull_number = event["closer"]["number"]
                closed_at = datetime.strptime(event["createdAt"], "%Y-%m-%dT%H:%M:%SZ")

                if closed_at >= cutoff_date:
                    instance = {
                        "repo": f"{owner}/{repo_name}",
                        "pull_number": pull_number,
                        "issue_number": issue_number,
                    }

                    with open(path_issue, "a") as f:
                        f.write(json.dumps(instance) + "\n")


def fetch_pulls(repo, token, path_pulls, cutoff_date):
    owner, repo_name = repo.split("/")
    path_issue = os.path.join(path_pulls, f"{repo_name}-issues-{cutoff_date}.jsonl")
    path_pull2issue = os.path.join(
        path_pulls, f"{repo_name}-pull2issue-{cutoff_date}.jsonl"
    )

    logger.info(f"[{repo}] 🚀 Fetching closed issues ...")
    collect_closed_issues(owner, repo_name, token, path_issue, cutoff_date)

    if not os.path.exists(path_issue):
        logger.info(f"[{repo}] ❌ Found no issues closed by a pull.")
        with open(path_pull2issue, "w") as f:
            f.write("")
    else:
        pull2issue = {}
        with open(path_issue, "r") as f:
            for line in f:
                instance = json.loads(line)
                if instance["pull_number"] not in pull2issue:
                    pull2issue[instance["pull_number"]] = [instance["issue_number"]]
                else:
                    pull2issue[instance["pull_number"]].append(instance["issue_number"])
        with open(path_pull2issue, "w") as f:
            for key, value in pull2issue.items():
                f.write(json.dumps({"pull": key, "issue": value}) + "\n")
        logger.info(
            f"[{repo}] ⭕️ Successfully saved {len(pull2issue)} closed issue data instances."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch closed issues and their associated pull requests."
    )
    parser.add_argument(
        "--repo", default="flow-project/flow", help="Repository in the format 'owner/repo'."
    )
    parser.add_argument("--token", default=os.getenv("GH_TOKEN"), help="GitHub personal access token.")
    parser.add_argument(
        "--path_pulls", default="./flow_test_file", help="Path to save pull request data."
    )
    parser.add_argument(
        "--cutoff_date", default="20090101", help="Cutoff date in YYYYMMDD format."
    )

    args = parser.parse_args()

    fetch_pulls(args.repo, args.token, args.path_pulls, args.cutoff_date)


if __name__ == "__main__":
    main()
