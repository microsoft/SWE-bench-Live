"""
Defines the workflow graph for repository environment setup and verification.
"""
import json
import os
import time
from functools import partial

from langgraph.graph import END, START, StateGraph

from launch.agent.base_image import select_base_image
from launch.agent.locate import locate_related_file
from launch.agent.setup import setup, start_bash_session
from launch.agent.state import AgentState, auto_catch
from launch.agent.verify import verify
from launch.utilities.language_handlers import get_language_handler


@auto_catch
def save_result(state: AgentState) -> dict:
    """
    Save the launch result to a JSON file and commit successful setup to Docker image.
    
    Args:
        state (AgentState): Current agent state containing results and session info
        
    Returns:
        dict: Updated state with session set to None
    """
    instance_id = state["instance"]["instance_id"]
    logger = state["logger"]
    path = state["result_path"]
    start_time = state["start_time"]
    duration = time.time() - start_time

    # transform to minutes
    duration = int(duration / 60)

    logger.info(f"Duration: {duration} minutes")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    exception = state.get("exception", None)
    exception = str(exception) if exception else None

    if not exception and not state.get("success", False):
        exception = "Launch failed"

    with open(path, "w") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "base_image": state["base_image"],
                    "setup_commands": state["setup_commands"],
                    "test_commands": state["test_commands"],
                    "duration": duration,
                    "completed": state.get("success", False),
                    "exception": exception,
                },
                indent=2,
            )
        )
    logger.info("Result saved to: " + str(path))
    if state["exception"]:
        logger.error(f"!!! Exception: {state['exception']}")

    session = state["session"]

    # Clean up language-specific resources
    language = state["language"]
    language_handler = get_language_handler(language)
    server = state["pypiserver"]  # Keep name for backward compatibility
    
    try:
        language_handler.cleanup_environment(session, server)
    except Exception as e:
        logger.warning(f"Failed to cleanup language environment: {e}")

    if state.get("success", False):
        logger.info("Setup completed successfully, now commit into swebench image.")

        ARCH = "x86_64"
        NAMESPACE = "starryzhang"

        key = f"sweb.eval.{ARCH}.{instance_id.lower()}"
        key = f"{NAMESPACE}/{key}".replace("__", "_1776_")

        # try:
        #     session.commit(image_name=key, push=False)
        #     logger.info(f"Image {key} committed successfully.")
        # except Exception as e:
        #     logger.error(f"Failed to commit image: {e}")

    try:
        session.cleanup()
    except Exception as e:
        logger.error(f"Failed to cleanup session: {e}")

    return {
        "session": None,
    }


def define_workflow(max_trials: int = 1, max_steps: int = 20):
    """
    Define the workflow graph for repository environment setup.
    
    Args:
        max_trials (int): Maximum number of setup/verify retry attempts
        max_steps (int): Maximum steps allowed for setup and verify agents
        
    Returns:
        Compiled workflow graph ready for execution
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("locate_related_file", locate_related_file)
    graph.add_node("select_base_image", select_base_image)
    graph.add_node("start_bash_session", start_bash_session)
    setup_agent = partial(setup, max_steps)
    graph.add_node("setup", setup_agent)
    verify_agent = partial(verify, max_steps)
    graph.add_node("verify", verify_agent)
    graph.add_node("save_result", save_result)

    graph.add_edge(START, "locate_related_file")
    graph.add_edge("locate_related_file", "select_base_image")
    graph.add_edge("select_base_image", "start_bash_session")
    graph.add_edge("start_bash_session", "setup")
    graph.add_edge("setup", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda x: x.get("success") or x["trials"] == max_trials or x["exception"],
        {True: "save_result", False: "setup"},
    )
    graph.add_edge("save_result", END)
    return graph.compile()
