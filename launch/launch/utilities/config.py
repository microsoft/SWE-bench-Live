"""
Configuration management for launch operations.
"""
import json
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration settings for repository launch operations.
    
    Attributes:
        llm_provider_name (str): Name of the LLM provider (e.g., "AOAI")
        print_to_console (bool): Whether to print logs to console
        model_config (dict): Configuration for the LLM model
        workspace_root (str): Root directory for workspace creation
        dataset (str): Path to the dataset file
        instance_id (str): Specific instance ID to run, None for all instances
        first_N_repos (int): Limit processing to first N repos (-1 for all)
        max_workers (int): Number of parallel workers for processing
        overwrite (bool): Whether to overwrite existing results
    """
    llm_provider_name: str
    print_to_console: bool
    model_config: dict
    workspace_root: str
    dataset: str
    instance_id: str  # instance id to run, if None, will run all instances in the dataset
    first_N_repos: int = -1  # -1 means all repos
    max_workers: int = 5
    overwrite: bool = (
        False  # whether to overwrite existing results, False will skip existing repos
    )


def load_config(config_path: str) -> Config:
    """
    Load the configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Config: An instance of the Config class containing the loaded configuration.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)

    return Config(
        llm_provider_name=config_data.get("llm_provider_name", "AOAI"),
        print_to_console=config_data.get("print_to_console", True),
        model_config=config_data.get(
            "model_config",
            {
                "model_name": "gpt-4o-20241120",
                "temperature": 0.0,
            },
        ),
        workspace_root=config_data.get("workspace_root"),
        dataset=config_data.get("dataset"),
        first_N_repos=config_data.get("first_N_repos", -1),
        max_workers=config_data.get("max_workers", 5),
        overwrite=config_data.get("overwrite", False),
        instance_id=config_data.get("instance_id", None),
    )
