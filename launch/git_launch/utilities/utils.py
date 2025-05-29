"""
Utility functions for workspace and repository management.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from git_launch.utilities.config import Config
from git_launch.utilities.llm import LLMProvider
from git_launch.utilities.logger import setup_logger


@dataclass
class WorkSpace:
    """
    Workspace container for a SWE-bench instance with all necessary components.
    
    Attributes:
        instance_id (str): Unique identifier for the instance
        repo_root (Path): Path to the cloned repository
        instance_path (Path): Path to instance metadata file
        result_path (Path): Path to store execution results
        logger (logging.Logger): Logger for this instance
        llm (LLMProvider): LLM provider for agent interactions
        llm_log_folder (Path): Directory for LLM interaction logs
        date (str): Creation date of the instance (optional)
        language (str): Programming language of the repository
    """
    instance_id: str
    repo_root: Path
    instance_path: Path  # TODO what is this for?
    result_path: Path
    logger: logging.Logger
    llm: LLMProvider
    llm_log_folder: Path
    date: str = None
    language: str = "python"


def prepare_repo(instance: dict, repo_root: Path) -> Path:
    """
    Prepares the repository by cloning it from GitHub and checking out the specified commit.
    Args:
        instance (dict): The instance containing repository information.
        repo_root (Path): The root directory where the repository will be cloned.
    """
    url = f'https://github.com/{instance["repo"]}.git'
    base_commit = instance["base_commit"]

    if repo_root.exists():
        return repo_root

    repo = Repo.clone_from(url, repo_root)
    repo.git.reset("--hard", base_commit)

    return repo_root


def check_workspace_exists(workspace_root: Path, instance: dict) -> bool:
    """Check if the workspace for the given instance already exists."""
    instance_folder = workspace_root / instance["instance_id"]
    result_path = instance_folder / "result.json"
    instance_path = instance_folder / "instance.json"
    if (
        (not instance_folder.exists())
        or (not result_path.exists())
        or (not instance_path.exists())
    ):
        return False
    return True


def prepare_workspace(
    workspace_root: Path, instance: dict, config: Config
) -> WorkSpace:
    """
    Prepare a complete workspace for processing a SWE-bench instance.
    
    Args:
        workspace_root (Path): Root directory for all workspaces
        instance (dict): SWE-bench instance data
        config (Config): Configuration settings
        
    Returns:
        WorkSpace: Fully configured workspace ready for processing
    """
    instance_folder = workspace_root / instance["instance_id"]
    instance_folder.mkdir(parents=True, exist_ok=True)
    result_path = instance_folder / "result.json"
    instance_path = instance_folder / "instance.json"
    llm_log_folder = instance_folder / "llm"
    llm_log_folder.mkdir(parents=True, exist_ok=True)
    llm = LLMProvider(
        llm_provider=config.llm_provider_name,
        log_folder=llm_log_folder,
        **config.model_config,
    )
    with open(instance_path, "w") as f:
        json.dump(instance, f, indent=2)

    repo_root = prepare_repo(instance, instance_folder / "repo")
    log_file = instance_folder / "setup.log"
    logger = setup_logger(
        instance["instance_id"], log_file, printing=config.print_to_console
    )
    return WorkSpace(
        instance_id=instance["instance_id"],
        language=instance["language"] if "language" in instance else "python",
        repo_root=repo_root,
        instance_path=instance_path,
        result_path=result_path,
        logger=logger,
        llm_log_folder=llm_log_folder,
        llm=llm,
    )
