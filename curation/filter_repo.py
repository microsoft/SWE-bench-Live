"""
Criteria:
- More than [] pulls and issues (total)
- More than [] forks
- The percentage of [] language code should be more than 60%
"""

import asyncio
import json
import re
from itertools import cycle
from typing import Any, Dict

import aiohttp
import fire
from aiohttp import ClientSession
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

GITHUB_API = "https://api.github.com"


def load_tokens(token_file: str):
    with open(token_file, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    if not tokens:
        raise ValueError("Token file is empty.")
    return cycle(tokens)


async def fetch_json(
    session: ClientSession, url: str, token_cycle, console, params=None
):
    while True:
        token = next(token_cycle)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 403:
                    await asyncio.sleep(1)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            console.log(f"[red]Error fetching {url}: {e}[/red]")
            return {}


async def fetch_pr_count_from_link_header(
    session: ClientSession, url: str, token_cycle, console
) -> int:
    while True:
        token = next(token_cycle)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            async with session.get(
                url, headers=headers, params={"state": "all", "per_page": 1}
            ) as resp:
                if resp.status == 403:
                    await asyncio.sleep(1)
                    continue
                resp.raise_for_status()
                link = resp.headers.get("Link")
                if not link:
                    return len(await resp.json())  # maybe 0 or 1 item
                match = re.search(r'&page=(\d+)>; rel="last"', link)
                return int(match.group(1)) if match else 0
        except Exception as e:
            console.log(f"[red]Error fetching pr count from {url}: {e}[/red]")
            return 0

async def fetch_issue_count_from_search_api(
    session: ClientSession, owner_repo: str, token_cycle, console
) -> int:
    """
    Fetches the total count of all Issues (open and closed) for a repository 
    using the GitHub Search API, which provides a 'total_count' field.
    """
    search_url = "https://api.github.com/search/issues"
    query_params = {
        "q": f"repo:{owner_repo} type:issue",
        "per_page": 1, 
    }
    while True:
        token = next(token_cycle)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.json",
        }  
        try:
            async with session.get(
                search_url, headers=headers, params=query_params
            ) as resp:
                if resp.status == 403:
                    await asyncio.sleep(1)
                    continue
                
                resp.raise_for_status()
                data = await resp.json()

                total_count = data.get("total_count", 0)
                return total_count

        except Exception as e:
            console.log(f"[red]Error fetching issue count for {owner_repo}: {e}[/red]")
            return 0

async def filter_repo(
    repo: Dict[str, Any],
    session: ClientSession,
    token_cycle,
    criteria,
    progress,
    task_id,
    console,
):
    full_name = repo.get("full_name")
    if not full_name:
        progress.update(task_id, advance=1)
        return None

    forks = repo.get("forks_count", 0)
    if forks < criteria["min_forks"]:
        progress.update(task_id, advance=1)
        return None

    owner_repo_url = f"{GITHUB_API}/repos/{full_name}"
    pulls_url = f"{owner_repo_url}/pulls"
    langs_url = f"{owner_repo_url}/languages"

    issues_count, pulls_count, langs_data = await asyncio.gather(
        fetch_issue_count_from_search_api(session, full_name, token_cycle, console),
        fetch_pr_count_from_link_header(session, pulls_url, token_cycle, console),
        fetch_json(session, langs_url, token_cycle, console),
    )
    
    total = issues_count + pulls_count
    if total < criteria["min_total_pr_issues"]:
        progress.update(task_id, advance=1)
        return None

    total_bytes = sum(langs_data.values())
    target_bytes = langs_data.get(criteria["language"], 0)
    percentage = (target_bytes / total_bytes) * 100 if total_bytes > 0 else 0

    progress.update(task_id, advance=1)

    if percentage < criteria["min_lang_percent"]:
        return None

    return repo


async def filter_main(
    input_file: str,
    output_file: str,
    tokens_file: str,
    min_total_pr_issues: int = 200,
    min_forks: int = 200,
    language: str = "Python",
    min_lang_percent: float = 60.0,
    max_workers: int = 5,
):
    console = Console()
    token_cycle = load_tokens(tokens_file)

    with open(input_file, "r", encoding="utf-8") as f:
        repos = [json.loads(line) for line in f]

    console.print(
        f"[bold cyan]Filtering {len(repos)} repos using {max_workers} workers...[/bold cyan]"
    )

    criteria = {
        "min_total_pr_issues": min_total_pr_issues,
        "min_forks": min_forks,
        "language": language,
        "min_lang_percent": min_lang_percent,
    }

    connector = aiohttp.TCPConnector(limit=max_workers)
    timeout = aiohttp.ClientTimeout(total=60)

    results = []

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TextColumn("[blue]{task.completed}/{task.total} done"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Filtering Repositories...", total=len(repos))

        async with ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(max_workers)

            async def sem_task(repo):
                async with semaphore:
                    return await filter_repo(
                        repo, session, token_cycle, criteria, progress, task_id, console
                    )

            tasks = [sem_task(repo) for repo in repos]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for repo in results:
            f.write(json.dumps(repo) + "\n")

    console.print(
        f"\n[bold green]Done.[/bold green] [white]{len(results)}[/white] repositories saved to [italic]{output_file}[/italic]"
    )


def run_filter(*args, **kwargs):
    asyncio.run(filter_main(*args, **kwargs))


if __name__ == "__main__":
    fire.Fire(run_filter)
