"""
Defines the workflow graph for repository environment setup and verification.
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from launch.agent.setup.base_image import select_base_image
from launch.agent.locate import locate_related_file
from launch.agent.setup.setup import setup, start_bash_session
from launch.agent.state import AgentState
from launch.agent.setup.verify import verify
from launch.agent.setup.save import save_setup_result


def define_setup_workflow(max_trials: int = 3, max_steps_setup: int = 20, max_steps_verify: int = 20, timeout: int = 30):
    """
    Define the workflow graph for repository environment setup.
    
    Args:
        max_trials (int): Maximum number of setup/verify retry attempts
        max_steps_setup (int): Maximum steps allowed for setup 
        max_steps_verify (int): Maximum steps allowed for verify 
        timeout (int): timeout after ? minutes only for setup step
        
    Returns:
        Compiled workflow graph ready for execution
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("locate_related_file", locate_related_file)
    graph.add_node("select_base_image", select_base_image)
    graph.add_node("start_bash_session", start_bash_session)
    setup_agent = partial(setup, 
                          max_steps = max_steps_setup,
                          timeout = timeout)
    graph.add_node("setup", setup_agent)
    verify_agent = partial(verify, 
                           max_steps = max_steps_verify)
    graph.add_node("verify", verify_agent)
    graph.add_node("save_result", save_setup_result)

    graph.add_edge(START, "locate_related_file")
    graph.add_edge("locate_related_file", "select_base_image")
    graph.add_edge("select_base_image", "start_bash_session")
    graph.add_edge("start_bash_session", "setup")
    graph.add_edge("setup", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda x: "return" if bool(x.get("success", False) or (x["trials"] == max_trials) or x.get("exception", False)) else "continue",
        {"return": "save_result", "continue": "setup"},
    )
    graph.add_edge("save_result", END)
    return graph.compile()


# ======================================================================== #

from launch.agent.organize.rebuild import reload_container
from launch.agent.organize.rebuild import organize_setup
from launch.agent.organize.testall import organize_test_cmd
from launch.agent.organize.parselog import generate_log_parser
from launch.agent.organize.testone import organize_unit_test
from launch.agent.organize.save import save_organize_result

def define_organize_workflow(max_steps: int = 20, timeout: int = 30):
    """
    Define the workflow graph for repository environment setup.
    
    Args:
        max_trials (int): Maximum number of setup/verify retry attempts
        max_steps_setup (int): Maximum steps allowed for setup 
        max_steps_verify (int): Maximum steps allowed for verify 
        timeout (int): timeout after ? minutes only for setup step
        
    Returns:
        Compiled workflow graph ready for execution
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("locate_related_file", locate_related_file)
    rebuild_agent = partial(organize_setup, 
                          max_steps = max_steps,
                          timeout = timeout)
    testall_agent = partial(organize_test_cmd, 
                           max_steps = max_steps,
                           timeout = timeout)
    parselog_agent = partial(generate_log_parser,
                            max_steps = max_steps,
                            timeout = timeout)
    testone_agent = partial(organize_unit_test, 
                           max_steps = max_steps,
                           timeout = timeout)
    graph.add_node("container", reload_container)
    graph.add_node("rebuild", rebuild_agent)
    graph.add_node("testall", testall_agent)
    graph.add_node("parselog", parselog_agent)
    graph.add_node("testone", testone_agent)
    graph.add_node("save_result", save_organize_result)

    graph.add_conditional_edges(
        START,
        lambda x: "container" if bool(x.get("docs", "")) else "locate_related_file",
        {"container": "container", "locate_related_file": "locate_related_file"},
    )
    graph.add_edge("locate_related_file", "container")
    graph.add_edge("container", "rebuild")
    
    graph.add_conditional_edges(
        "rebuild",
        lambda x: "return" if (not bool(x.get("success", False))) or bool(x.get("exception", False)) else "continue",
        {"return": "save_result", "continue": "testall"},
    )
    
    graph.add_conditional_edges(
        "testall",
        lambda x: "return" if (not bool(x.get("success", False))) or bool(x.get("exception", False)) else "continue",
        {"return": "save_result", "continue": "parselog"},
    )
    
    graph.add_conditional_edges(
        "parselog",
        lambda x: "return" if (not bool(x.get("success", False))) or bool(x.get("exception", False)) else "continue",
        {"return": "save_result", "continue": "testone"},
    )
    graph.add_edge("testone", "save_result")
    graph.add_edge("save_result", END)
    
    return graph.compile()

'''
START
|    \
|     \
|      \
|      locate
|      /
|     /
|    /
container
|
|
|
rebuild
|    \
|     \
|      \
|      testall
|      /    |
|     /     parselog
|    /      |
save<-----testone
|
|
|
END
'''