"""
Environment setup agent for repository testing environment preparation.
"""
import json
import shutil
import time
from typing import Any, Literal, ClassVar  

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from launch.agent.action_parser import ActionParser
from launch.agent.prompt import ReAct_prompt
from launch.agent.state import AgentState, auto_catch
from launch.core.runtime import SetupRuntime
from launch.utilities.language_handlers import get_language_handler


system_msg = """You are a developer. Your task is to install dependencies and set up a environment that is able to run the tests of the project.

- You start with an initial Docker container based on {base_image}.
- You interact with a Bash session inside this container.
- Project files are located in /testbed within the container, and your current working directory of bash is already set to /testbed.
- No need to clone the project again.

The final objective is to successfully run the tests of the project.

{language_instructions}
"""

# Omit the following requirement for now:
#   -> You are not allowed to edit code files in the project.


class SetupAction(BaseModel):
    prompt: ClassVar[dict] = {
        "linux":
        """
        Command: run a command in the bash, reply with following format, your command should not require sudo or interactive input:
            <command>bash command</command>
            e.g. install build-essential: <command>apt-get install -y build-essential</command>
            e.g. view file content: <command>cat README.md</command>
        Search: search the web for if you need some information, generate query and reply with following format:
            <search>the search query</search>
            e.g. <search>how to fix 'No module named setuptools'</search>
            e.g. <search>how to install python3 on ubuntu</search>
            e.g. <search>how to create development environment for python3</search>
        Stop: stop the setup loop once you think the setup is complete, reply with following format:
            <stop></stop>
        """,
        "windows":
        """
        Command: run a command in the windows powershell, reply with following format, your command should not require admin privilage or interactive input:
            <command>powershell command</command>
            e.g. install build-essential: <command>choco install -y ...</command>
            e.g. view file content: <command>cat README.md</command>
        Search: search the web for if you need some information, generate query and reply with following format:
            <search>the search query</search>
            e.g. <search>how to fix 'No module named setuptools'</search>
            e.g. <search>how to install python3 on ubuntu</search>
            e.g. <search>how to create development environment for python3</search>
        Stop: stop the setup loop once you think the setup is complete, reply with following format:
            <stop></stop>
        """,
    }

    action: Literal["command", "search", "stop"] = Field(
        "command", description="The action type"
    )
    args: Any = Field(None, description="The action arguments")


class SetupObservation(BaseModel):
    """Observation for the setup action"""

    content: str = Field("", description="The content of the observation")
    is_stop: bool = Field(False, description="Whether stop the setup loop")


class SetupActionParser(ActionParser):
    """Parser for setup agent actions."""
    
    def parse(self, response: str) -> SetupAction | None:
        """Parse setup action from LLM response text."""
        response = self.clean_response(response)
        
        command = self.extract_tag_content(response, "command")
        if command:
            return SetupAction(action="command", args=command)
            
        search = self.extract_tag_content(response, "search")
        if search:
            return SetupAction(action="search", args=search)
            
        if "<stop>" in response and "</stop>" in response:
            return SetupAction(action="stop", args=None)
            
        return None


def parse_setup_action(response: str) -> SetupAction | None:
    """Parse setup action from LLM response text."""
    parser = SetupActionParser()
    return parser.parse(response)


def observation_for_setup_action(
    state: AgentState, action: SetupAction | None
) -> SetupObservation:
    """
    Execute setup action and return observation.
    
    Args:
        state (AgentState): Current agent state
        action (SetupAction | None): Action to execute
        
    Returns:
        SetupObservation: Result of action execution
    """
    if not action or not action.action:
        content = f"""\
Please using following format after `Action: ` to make a valid action choice:
{SetupAction.prompt[state['platform']]}
"""
        return SetupObservation(content=content, is_stop=False)
    if action.action == "command":
        session = state["session"]
        result = session.send_command(action.args)
        return SetupObservation(content=result.to_observation(), is_stop=False)
    if action.action == "search":
        result = state["search_tool"].invoke(action.args)
        return SetupObservation(content=json.dumps(result), is_stop=False)
    if action.action == "stop":
        return SetupObservation(content="", is_stop=True)


@auto_catch
def start_bash_session(state: AgentState) -> dict:
    """
    Start a Docker container with bash session for repository testing.
    
    Args:
        state (AgentState): Agent state containing base image and instance info
        
    Returns:
        dict: Updated state with session and pypiserver
    """
    base_image = state["base_image"]
    repo_root = state["repo_root"]
    logger = state["logger"]
    logger.info(f"Starting bash session in container based on image: {base_image}")
    
    # Docker retry configuration
    max_docker_retries = 3
    docker_retry_delay = 30  # seconds
    session = None
    
    for attempt in range(max_docker_retries):
        try:
            logger.info(f"Docker attempt {attempt + 1}/{max_docker_retries}")
            session = SetupRuntime.from_base_image(base_image, state["instance"], platform = state["platform"])
            logger.info(f"Session started successfully: {session}")
            break
        except Exception as e:
            logger.warning(f"Docker attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_docker_retries - 1:
                logger.info(f"Waiting {docker_retry_delay} seconds before retry...")
                time.sleep(docker_retry_delay)
            else:
                logger.error(f"All {max_docker_retries} Docker attempts failed. Last error: {str(e)}")
                raise e

    # clean up repository in the host
    shutil.rmtree(repo_root, ignore_errors=True)
    logger.info(f"Repo root in the host cleaned up: {repo_root}")

    # Setup language-specific environment
    language = state["language"]
    print(f"Setting up environment for language: {language}")
    language_handler = get_language_handler(language)
    
    logger.info(f"Setting up environment for language: {language}")
    server = language_handler.setup_environment(session, state["date"])
    if server:
        logger.info(f"Language-specific server started")
    else:
        logger.info("No language-specific server needed")

    assert (
        session is not None
    ), "Session is None, please check the whether the docker is running"
    return {
        "pypiserver": server,  # Keep name for backward compatibility
        "session": session,
    }


SETUP_CONVERSATION_WINDOW = 40


@auto_catch
def setup(state: AgentState, max_steps: int, timeout: int = 30) -> dict:
    """
    ReAct agent for environment setup through conversational command execution.
    
    Args:
        max_steps (int): Maximum number of setup steps allowed
        state (AgentState): Current agent state with session and tools
        
    Returns:
        dict: Updated state with setup messages and commands
    """
    llm = state["llm"]
    logger = state["logger"]
    repo_structure = state["repo_structure"]

    logger.info(f"setup state: {state.get("success" , "false")}, {state["trials"]}, {state["exception"]} ... ")
    
    # Get language-specific instructions
    language = state["language"]
    language_handler = get_language_handler(language)
    language_instructions = language_handler.get_setup_instructions(state["base_image"], platform=state["platform"])
    hints = "\n\n"
    action_hints = state["instance"].get("hints", "")
    action_hints = f"\nAdditional hints from user that may help you set up / test the repo: <check>{action_hints}</check>.\n" if action_hints else ""
    hints += action_hints
    setup_cmds = state["instance"].get("setup_cmds", "")
    setup_cmds_hints = f"\nHints: this is the build commands used to build this repo other developers used in other platforms that may help you understand how to build the program. <command>{setup_cmds}</command>" if setup_cmds else ""
    hints += setup_cmds_hints
    test_cmds = state["instance"].get("test_cmds", "")
    test_cmd_hints = f"\nHints: this is the test command used for this repo other developers used in other platforms that may help you verify whether your build is successful. <command>{test_cmds}</command>" if test_cmds else ""
    hints += test_cmd_hints
    platform_hints = ""
    if state["platform"] == "windows":
        platform_hints = f"\n\nNote: This is a windows server image. Use windows powershell command.\n"
    hints += platform_hints

    logger.info("-" * 10 + "Start setup conversation" + "-" * 10)
    messages = [
        SystemMessage(system_msg.format(
            base_image=state["base_image"],
            language_instructions=language_instructions
        )),
        HumanMessage(
            ReAct_prompt.format(
                tools=SetupAction.prompt[state["platform"]],
                project_structure=repo_structure,
                docs=state["docs"],
            ) + hints
        ),
    ]
    # logger.info(f"### Initial messages: {messages}")
    messages.extend(state["verify_messages"])
    if bool(state["verify_messages"]):
        messages.append(
            HumanMessage(
                f"Test cases did not run successfully. The setup of the repository is not successful so far... Please try again to setup dependencies, build the repo and run tests!"
            )
        )
    prefix_messages = len(messages)
    commands = state.get("setup_commands", [])
    step = 0
    start_time = time.time()
    while step < max_steps:
        if time.time() - start_time > timeout * 60:
            logger.info(f"Reached global timeout of {timeout} minutes")
            break
        step += 1
        # uses a window to avoid exceed context
        commands_history = HumanMessage(
            f"\nThe previous commands which you have run to try to set up the repository:```\n{commands}```\nFollowing are the last {SETUP_CONVERSATION_WINDOW} messages:\n"
        )
        if len(messages) < SETUP_CONVERSATION_WINDOW + prefix_messages:
            input_messages = (
                messages[:prefix_messages]
                + [commands_history]
                + messages[prefix_messages:]
            )
        else:
            input_messages = (
                messages[:prefix_messages]
                + [commands_history]
                + messages[-SETUP_CONVERSATION_WINDOW:]
            )

        response = llm.invoke(input_messages)

        logger.info("\n" + response.pretty_repr())
        messages.append(response)
        action = parse_setup_action(response.content)
        if action and action.action == "command":
            commands.append(action.args)
        observation = observation_for_setup_action(state, action)
        if observation.is_stop:
            break
        message = HumanMessage(f"Observation:\n{observation.content}")
        # print(observation.content)
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End setup conversation" + "-" * 10)
    return {
        "messages": messages,
        "setup_messages": messages[prefix_messages:],
        "setup_commands": commands,
        "commands": commands,
    }
