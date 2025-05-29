"""
Core launch functionality for setting up and executing repository environments.
"""
import pprint

from git_launch.agent.state import AgentState
from git_launch.utilities.get_repo_structure import view_repo_structure
from git_launch.utilities.utils import WorkSpace
from git_launch.workflow import define_workflow


def launch(instance: dict, workspace: WorkSpace):
    """
    Launch the environment setup workflow for a SWE-bench instance.
    
    Args:
        instance (dict): SWE-bench instance containing repo and task information
        workspace (WorkSpace): Prepared workspace with repo, logger, and LLM provider
    """
    repo_structure = view_repo_structure(workspace.repo_root)
    workflow = define_workflow()
    logger = workspace.logger
    initial_state = AgentState.create(
        instance=instance,
        llm=workspace.llm,
        logger=logger,
        language=workspace.language,
        repo_root=workspace.repo_root.resolve(),
        repo_structure=repo_structure,
        result_path=workspace.result_path,
        date=instance.get("created_at", None),
    )

    for event in workflow.stream(initial_state, stream_mode="values", subgraphs=True):
        logger.debug(pprint.pformat(event))
