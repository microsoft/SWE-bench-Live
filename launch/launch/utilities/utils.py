"""
Utility functions for workspace and repository management.
"""
import json
import logging
from dataclasses import dataclass
import os
from pathlib import Path
import threading

from launch.utilities.config import Config
from launch.utilities.get_repo_structure import view_repo_structure
from launch.utilities.llm import LLMProvider
from launch.utilities.logger import setup_logger, clean_logger
import subprocess

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
    repo_structure: str
    date: str = None
    language: str = "python"
    platform: str = "linux"
    max_trials: str = 3
    max_steps_setup: int = 20
    max_steps_verify: int = 20
    max_steps_organize: int = 20
    timeout: int = 30
    image_prefix: str = "repolaunch/dev"
    
    def cleanup(self) -> None:
        """Clean up workspace resources."""
        try:
            clean_logger(self.logger)
        except Exception as e:
            print(f"Failed to clean logger: {e}")


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

    # Clone repo using subprocess
    subprocess.run(
        ["git", "clone", url, str(repo_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Reset to base_commit using subprocess
    subprocess.run(
        ["git", "reset", "--hard", base_commit],
        cwd=str(repo_root),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

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
    workspace_root: Path, instance: dict, config: Config, log_file: str="setup.log"
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
    instance_folder = workspace_root / "playground" / instance["instance_id"]
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
    
    repo_structure = None
    if os.path.exists(result_path):
        with open(result_path) as f:
            history = f.read()
        if history.strip():
            history = json.loads(history)
            repo_structure = history.get("repo_structure", None)

    repo_root = prepare_repo(instance, instance_folder / "repo")
    if not repo_structure:
        repo_structure = view_repo_structure(repo_root)
    log_file = instance_folder / log_file
    logger = setup_logger(
        instance["instance_id"], log_file, printing=config.print_to_console
    )
    
    language = instance.get("language", "python").lower()
    logger.info(f"Using language: {language}")
    
    return WorkSpace(
        instance_id=instance["instance_id"],
        language=language,
        repo_root=repo_root,
        image_prefix=config.image_prefix,
        instance_path=instance_path,
        result_path=result_path,
        repo_structure=repo_structure,
        logger=logger,
        llm_log_folder=llm_log_folder,
        llm=llm,
        platform=config.platform,
        max_trials=config.max_trials,
        max_steps_setup=config.max_steps_setup,
        max_steps_verify=config.max_steps_verify,
        max_steps_organize=config.max_steps_organize,
        timeout=config.timeout
    )



def safe_read_result(result: str, result_path: Path, lock: threading.Lock) -> dict:
    '''
    Though this function looks ugly,
    it is used to guarantee result.json is saved.
    Because due to some minor bugs in Python thread concurrency,
    result.json is not saved in the 'save' step successfully sometimes.
    '''
    with lock:
        if result_path.exists():
            saved_result = result_path.read_text()
            if saved_result.strip():
                return json.loads(saved_result)
    if not result.strip():
        return {
            "completed": False, 
            "organize_completed": False, 
            "exception": "Result Empty Error!"
        }
    with lock:
        with open(result_path, "w") as f:
            f.write(result)
    return json.loads(result)