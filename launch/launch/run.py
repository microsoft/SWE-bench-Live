"""
Orchestrates the execution of repository launches across multiple instances.

This module provides functionality to process SWE-bench instances in parallel,
setting up environments and executing launches with progress tracking.
"""
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
import traceback

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from launch.core.entry import setup, organize
from launch.utilities.config import Config, load_config
from launch.utilities.utils import prepare_workspace, safe_read_result
from launch.scripts import collect

lock = threading.Lock()


def setup_instance(instance, config, workspace_root):
    """
    Process a single SWE-bench instance by launching its environment.
    
    Args:
        instance (dict): SWE-bench instance data containing repo and commit info
        config (Config): Configuration object with launch settings
        workspace_root (Path): Root directory for workspace creation
        
    Returns:
        tuple: (status, instance_id, error_message)
            - status: "success", "fail", or "skip"
            - instance_id: identifier for the instance
            - error_message: error details if failed, None if successful
    """
    instance[
        "commit_url"
    ] = f"https://github.com/{instance['repo']}/tree/{instance['base_commit']}"

    
    instance_path = workspace_root / "playground" / instance["instance_id"] 
    result_path = instance_path / "result.json"

    
    if not config.overwrite and os.path.exists(result_path):
        result_path = instance_path / "result.json"
        result = (result_path).read_text()
        if result.strip():
            result = json.loads(result)
            if result["completed"]:
                return "success", instance["instance_id"], None
            elif result.get("exception", "") == "Launch failed":
                return "fail", instance["instance_id"], "Launch failed"

    try:
        workspace = prepare_workspace(workspace_root, instance, config)
        result = safe_read_result(setup(instance, workspace), result_path, lock)
        if result["completed"]:
            return "success", instance["instance_id"], None
        else:
            return (
                "fail",
                instance["instance_id"],
                result.get("exception", "Unknown error"),
            )
    except Exception as e:
        # in case unexpected error escapes previous clean-up
        # workspace may not exist if prepare_workspace() failed early
        try:
            if "workspace" in locals() and getattr(workspace, "repo_root", None):
                repo_path = workspace.repo_root.resolve()
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, ignore_errors=True)
        except Exception:
            # best-effort cleanup; don't mask the original exception
            pass
        return "fail", instance["instance_id"], str(e) + str(traceback.format_exc())


def organize_instance(instance, config, workspace_root):
    """
    Process a single SWE-bench instance by launching its environment.
    
    Args:
        instance (dict): SWE-bench instance data containing repo and commit info
        config (Config): Configuration object with launch settings
        workspace_root (Path): Root directory for workspace creation
        
    Returns:
        tuple: (status, instance_id, error_message)
            - status: "success", "fail", or "skip"
            - instance_id: identifier for the instance
            - error_message: error details if failed, None if successful
    """
    instance[
        "commit_url"
    ] = f"https://github.com/{instance['repo']}/tree/{instance['base_commit']}"

    instance_path = workspace_root / "playground" / instance["instance_id"] 
    result_path = instance_path / "result.json"

    
    if not config.overwrite and os.path.exists(result_path):
        result = (result_path).read_text()
        if result.strip():
            result = json.loads(result)
            if result.get("organize_completed", False):
                return "success", instance["instance_id"], None
            elif result.get("exception", "") == "Organize failed":
                return "fail", instance["instance_id"], "Organize failed"

    try:
        workspace = prepare_workspace(workspace_root, instance, config)
        result = safe_read_result(organize(instance, workspace), result_path, lock)
        if result["organize_completed"]:
            return "success", instance["instance_id"], None
        else:
            return (
                "fail",
                instance["instance_id"],
                result.get("exception", "Unknown error"),
            )
    except Exception as e:
        # in case unexpected error escapes previous clean-up
        if os.path.exists(instance_path / "repo"):
            shutil.rmtree(instance_path / "repo", ignore_errors=True)
        return "fail", instance["instance_id"], str(e) + str(traceback.format_exc())



def run_setup(config: Config, dataset: list):
    """
    Main function to run launches for multiple instances with parallel processing.
    
    Args:
        config_path (str): Path to the configuration JSON file
    """

    console = Console()
    workspace_root = Path(config.workspace_root)

    if config.first_N_repos > 0:
        dataset = dataset[: config.first_N_repos]
        console.print(f"[yellow]Processing first {config.first_N_repos} repositories only[/yellow]")

    if config.instance_id:
        dataset = [
            instance
            for instance in dataset
            if instance["instance_id"] == config.instance_id
        ]

    console.rule("[bold green] Starting Launching Repositories...")
    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[green]Success: {task.fields[success]}[/green] | [red]Fail: {task.fields[fail]}[/red]"
        ),
        BarColumn(),
        TextColumn(f"Total: {len(dataset)}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(dataset)} instances",
            total=len(dataset),
            success=0,
            fail=0,
        )

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    setup_instance, instance, config, workspace_root
                ): instance
                for instance in dataset
            }

            for future in as_completed(futures): 
                try:
                    status, instance_id, error = future.result(timeout=10800)  # 3 hours timeout
                    if status == "skip":
                        console.print(
                            f"[yellow]Skipped[/yellow] {instance_id}: {error or ''}"
                        )
                    elif status == "fail":
                        with lock:
                            progress.update(
                                task, advance=0, fail=progress.tasks[0].fields["fail"] + 1
                            )
                        console.print(f"[red]Failed[/red] {instance_id}: {error}")
                    elif status == "success":
                        with lock:
                            progress.update(
                                task,
                                advance=0,
                                success=progress.tasks[0].fields["success"] + 1,
                            )
                        console.print(f"[green]Success![/green] {instance_id}")
                except TimeoutError:
                    # Find the instance_id for this future
                    instance_id = futures.get(future, {}).get("instance_id", "unknown")
                    with lock:
                        progress.update(
                            task, advance=0, fail=progress.tasks[0].fields["fail"] + 1
                        )
                    console.print(f"[red]Timeout[/red] {instance_id}: Task exceeded 3 hour timeout")
                    future.cancel()  # Cancel the timed-out task
                progress.update(task, advance=1)

    console.rule("[bold green] Finished setting up all instances!")

    # Log which repositories were processed
    if config.first_N_repos > 0:
        processed_repos = list(set([instance["repo"] for instance in dataset]))
        console.print(f"[blue]Processed repositories ({len(processed_repos)}):[/blue]")
        for i, repo in enumerate(sorted(processed_repos), 1):
            console.print(f"  {i}. {repo}")
        console.print(f"[yellow]Total instances processed: {len(dataset)}[/yellow]")


def run_organize(config: Config, dataset: list):
    """
    Main function to run launches for multiple instances with parallel processing.
    
    """

    console = Console()
    workspace_root = Path(config.workspace_root)

    if config.first_N_repos > 0:
        dataset = dataset[: config.first_N_repos]
        console.print(f"[yellow]Processing first {config.first_N_repos} repositories only[/yellow]")

    if config.instance_id:
        dataset = [
            instance
            for instance in dataset
            if instance["instance_id"] == config.instance_id
        ]

    console.rule("[bold green] Starting Organizing Launch Info...")
    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[green]Success: {task.fields[success]}[/green] | [red]Fail: {task.fields[fail]}[/red]"
        ),
        BarColumn(),
        TextColumn(f"Total: {len(dataset)}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(dataset)} instances",
            total=len(dataset),
            success=0,
            fail=0,
        )

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    organize_instance, instance, config, workspace_root
                ): instance
                for instance in dataset
            }

            for future in as_completed(futures): 
                try:
                    status, instance_id, error = future.result(timeout=10800)  # 3 hours timeout
                    if status == "skip":
                        console.print(
                            f"[yellow]Skipped[/yellow] {instance_id}: {error or ''}"
                        )
                    elif status == "fail":
                        with lock:
                            progress.update(
                                task, advance=0, fail=progress.tasks[0].fields["fail"] + 1
                            )
                        console.print(f"[red]Failed[/red] {instance_id}: {error}")
                    elif status == "success":
                        with lock:
                            progress.update(
                                task,
                                advance=0,
                                success=progress.tasks[0].fields["success"] + 1,
                            )
                        console.print(f"[green]Success![/green] {instance_id}")
                except TimeoutError:
                    # Find the instance_id for this future
                    instance_id = futures.get(future, {}).get("instance_id", "unknown")
                    with lock:
                        progress.update(
                            task, advance=0, fail=progress.tasks[0].fields["fail"] + 1
                        )
                    console.print(f"[red]Timeout[/red] {instance_id}: Task exceeded 3 hour timeout")
                    future.cancel()  # Cancel the timed-out task
                progress.update(task, advance=1)

    console.rule("[bold green] Finished organizing all instances!")

    # Log which repositories were processed with their status
    if config.first_N_repos > 0:
        processed_instances = list(set([instance["instance_id"] for instance in dataset]))
        console.print(f"[blue]Processed repositories ({len(processed_instances)}):[/blue]")
        for i, instance_id in enumerate(sorted(processed_instances), 1):
            console.print(f"  {i}. {instance_id}")

def run_launch(config_path):
    config: Config = load_config(config_path)
    with open(config.dataset, "r") as f:
        dataset = [json.loads(line) for line in f]
        instance_ids: list[str] = [instance["instance_id"] for instance in dataset]
    if config.mode["setup"]:
        run_setup(config, dataset)
        collect.main(config.workspace_root, platform = config.platform, step = "setup", instance_ids = instance_ids)
    if config.mode["organize"]:
        if not os.path.exists(f"{config.workspace_root}/setup.jsonl"):
            raise RuntimeError(f"{config.workspace_root}/setup.jsonl NOT FOUND. You need to finish the setup step first.")
        with open(f"{config.workspace_root}/setup.jsonl") as f:
            dataset = [json.loads(line) for line in f]
        run_organize(config, dataset)
        collect.main(config.workspace_root, platform = config.platform, step = "organize", instance_ids = instance_ids)
    return



def main():
    """
    Entry point for the repo-launch command line tool.
    """
    import argparse

    argparser = argparse.ArgumentParser(
        description="RepoLaunch - Turn any codebase into a testable sandbox environment"
    )
    argparser.add_argument(
        "--config-path", type=str, required=True, help="Path to configuration file"
    )
    args = argparser.parse_args()

    run_launch(args.config_path)


if __name__ == "__main__":
    main()
