import json
import os
import shutil
import time
from launch.agent.state import AgentState, auto_catch
from launch.utilities.language_handlers import get_language_handler

#@auto_catch
def save_organize_result(state: AgentState) -> dict:
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
        exception = "Organize failed"

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

        key = state["image_prefix"]
        tag = f"{instance_id}_{state["platform"]}"
        try:
            session.commit(image_name=key, tag=tag, push=False)
            logger.info(f"Image {key}:{tag} committed successfully.")
            state["docker_image"] = f"{key}:{tag}"
        except Exception as e:
            raise Exception(f"Failed to commit image: {e}. If timeout please commit and clean the container manually.")

    # in case unexpected error escapes previous clean-up
    if os.path.exists(state["repo_root"]):
        shutil.rmtree(state["repo_root"], ignore_errors=True)
    try:
        session.cleanup()
    except Exception as e:
        logger.error(f"Failed to cleanup session: {e}")

    if os.path.exists(path):
        with open(path) as f:
            history = f.read()
            if history.strip():
                history = json.loads(history)
            else:
                history = {}
    else:
        history = {}
    
    result = json.dumps(
            {
                **history,
                "instance_id": instance_id,
                "docker_image": state.get("docker_image", state["instance"].get("docker_image", "")),
                "rebuild_commands": state.get("setup_commands", []),
                "test_commands": state.get("test_commands", []),
                "test_status": state.get("test_status", {}),
                "pertest_command": state.get("pertest_command", {}),
                "log_parser": state.get("parser", ""),
                "unittest_generator": state.get("unittest_generator", ""),
                "organize_duration": duration,
                "organize_completed": state.get("success", False),
                "exception": exception,
                "repo_structure": state["repo_structure"],
                "docs": state["docs"],
            },
            indent=2,
        )
    with open(path, "w") as f:
        f.write(result)
    time.sleep(10)
    logger.info("Result saved to: " + str(path))

    return {
        "session": None,
        "result": result,
    }
