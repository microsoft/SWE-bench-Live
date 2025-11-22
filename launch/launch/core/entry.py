"""
Core launch functionality for setting up and executing repository environments.
"""
import pprint

from launch.agent.state import AgentState
from launch.utilities.utils import WorkSpace
from launch.core.workflow import define_setup_workflow, define_organize_workflow


def setup(instance: dict, workspace: WorkSpace):
    """
    Launch the environment setup workflow for a SWE-bench instance.
    
    Args:
        instance (dict): SWE-bench instance containing repo and task information
        workspace (WorkSpace): Prepared workspace with repo, logger, and LLM provider
        
    Returns:
        AgentState: Final state after workflow completion
    """
    workflow = define_setup_workflow(
        max_trials = workspace.max_trials, 
        max_steps_setup = workspace.max_steps_setup, 
        max_steps_verify = workspace.max_steps_verify, 
        timeout = workspace.timeout
    )
    logger = workspace.logger
    initial_state = AgentState.create(
        instance=instance,
        llm=workspace.llm,
        logger=logger,
        language=workspace.language,
        repo_root=workspace.repo_root.resolve(),
        repo_structure=workspace.repo_structure,
        image_prefix=workspace.image_prefix,
        result_path=workspace.result_path,
        date=instance.get("created_at", None),
        platform=workspace.platform,
    )

    final_state = initial_state
    for event in workflow.stream(initial_state, stream_mode="values", subgraphs=True):
        logger.debug(pprint.pformat(event))
        if isinstance(event, tuple):
            metadata, final_state = event
        else:
            final_state = event
    
    result = final_state.get("result", "")
    if not result:
        print("Warning! Result not found!")
    return result


def organize(instance: dict, workspace: WorkSpace):
    """
    Launch the environment setup workflow for a SWE-bench instance.
    
    Args:
        instance (dict): SWE-bench instance containing repo and task information
        workspace (WorkSpace): Prepared workspace with repo, logger, and LLM provider
        
    Returns:
        AgentState: Final state after workflow completion
    """
    workflow = define_organize_workflow(
        max_steps = workspace.max_steps_organize,
        timeout = workspace.timeout
    )
    logger = workspace.logger
    initial_state = AgentState.create(
        instance=instance,
        llm=workspace.llm,
        logger=logger,
        language=workspace.language,
        repo_root=workspace.repo_root.resolve(),
        repo_structure=workspace.repo_structure,
        image_prefix=workspace.image_prefix,
        result_path=workspace.result_path,
        date=instance.get("created_at", None),
        platform=workspace.platform,
    )

    final_state = initial_state
    for event in workflow.stream(initial_state, stream_mode="values", subgraphs=True):
        logger.debug(pprint.pformat(event))
        if isinstance(event, tuple):
            metadata, final_state = event
        else:
            final_state = event
    
    result = final_state.get("result", "")
    if not result:
        print("Warning! Result not found!")
    return result
