"""
Orchestrates the execution of repository launches across multiple instances.

This module provides functionality to process SWE-bench instances in parallel,
setting up environments and executing launches with progress tracking.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from launch.entry import launch
from launch.utilities.config import load_config
from launch.utilities.utils import check_workspace_exists, prepare_workspace

lock = threading.Lock()


def process_instance(instance, config, workspace_root):
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

    if not config.overwrite and check_workspace_exists(workspace_root, instance):
        result_path = workspace_root / instance["instance_id"] / "result.json"
        result = json.loads((result_path).read_text())
        if result["completed"]:
            return "success", instance["instance_id"], None
        else:
            exception = result.get("exception", "")
            return "fail", instance["instance_id"], exception

    try:
        workspace = prepare_workspace(workspace_root, instance, config)
        launch(instance, workspace)
        result = json.loads(workspace.result_path.read_text())
        if result["completed"]:
            return "success", instance["instance_id"], None
        else:
            exception = result.get("exception", "")
            return (
                "fail",
                instance["instance_id"],
                result.get("exception", "Unknown error"),
            )
    except Exception as e:
        return "fail", instance["instance_id"], str(e)


def run_launch(config_path):
    """
    Main function to run launches for multiple instances with parallel processing.
    
    Args:
        config_path (str): Path to the configuration JSON file
    """
    config = load_config(config_path)

    with open(config.dataset, "r") as f:
        dataset = [json.loads(line) for line in f]
    if config.first_N_repos > 0:
        dataset = dataset[: config.first_N_repos]

    if config.instance_id:
        dataset = [
            instance
            for instance in dataset
            if instance["instance_id"] == config.instance_id
        ]

    console = Console()
    workspace_root = Path(config.workspace_root)

    console.rule("[bold green]Starting Launching Repositories...")
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
                    process_instance, instance, config, workspace_root
                ): instance
                for instance in dataset
            }

            for future in as_completed(futures):
                status, instance_id, error = future.result()
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
                progress.update(task, advance=1)

    console.rule("[bold green] Finished all instances!")


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
