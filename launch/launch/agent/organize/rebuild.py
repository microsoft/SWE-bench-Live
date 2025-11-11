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


# system_msg = """You are a developer. You have already setup all dependencies and build the repository in the current folder.
# However, for the maintainance of the project, you need to organize the minimal commands to re-install ONLY modified packages and build the projects again after edits to the source code / package list.

# - You are inside a docker container with source code already inside the container under the current directory called /testbed
# - The dependencies of the repository have already been set up by you before.
# - The full history commands that you used to try to set up the repo: {commands}

# You can send commands in the container for several times to try to test the commands to re-build the repo and expolre the repo freely if you need more information.
# You do not need to include the commands to run test cases because we will do it later.

# The final objective is: 
#     to "find the minimal commands to re-install ONLY modified packages AND re-build the project" again after package list / source code edits and "output your minimal re-install & re-build commands in one line".
# You need to finish it in {steps} steps.
# """
system_msg = """You are a developer. You have already set up all dependencies and successfully built the repository in the current folder.
Now, for project maintenance, you must organize the minimal commands required to re-install only modified packages and re-build the project after any edits to the source code or package list.

Environment details:
- You are inside a Docker container with the source code already present at /testbed.
- All dependencies have been previously installed by you.
- The complete command history of your setup process is available as {commands}.

Your workflow process:
1. You may execute commands inside the container multiple times to inspect files, verify changes, or explore the repo.
2. You do not need to include test case execution commands.
3. When you think you have found the minimal rebuild commands, submit them using <submit>...</submit>
4. The system will AUTOMATICALLY verify your submitted commands by executing them.
5. If verification passes (commands execute successfully), the process completes.
6. If verification fails, you will receive the error output and must try different commands.
7. Repeat steps 3-6 until you find working minimal rebuild commands.

Final objective:
Find the minimal set of commands required to:
- Re-install only modified packages after edits to the package list or source code.
- Re-build the project efficiently.

Output requirements:
Submit your minimal re-install and re-build commands in a single line using <submit>.
The system will automatically test them and provide feedback if they fail.
You must complete this in {steps} steps."""

# Omit the following requirement for now:
#   -> You are not allowed to edit code files in the project.


class SetupAction(BaseModel):
    '''
        Command: run a command in the command line, reply with following format, your command should not require sudo/admin privilage or interactive input:
            <command>...</command>
            e.g. <command>python main.py</command>
        Search: search the web if you need some information, generate query and reply with following format:
            <search>...</search>
            e.g. <search>how to fix 'No module named setuptools'</search>
        Submit: stop the exploration loop once you find the minimal commands to re-install modified packages and re-build the repo. Submit your minimal commands in one line, link multiple commands with ";"
            <submit>...</submit>
            e.g. <submit>./gradlew resolveAllDependencies ; ./gradlew check</submit>
            Of course you can submit with empty commands (an enter \n) if the repo really does not require any re-install and re-build: <submit>\n</submit>
    '''

    action: Literal["command", "search", "submit"] = Field(
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
            
        submit = self.extract_tag_content(response, "submit")
        if submit:
            return SetupAction(action="submit", args=submit)
            
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
    if (not action) or (not action.action):
        content = f"""\
Please using following format after `Action: ` to make a valid action choice:
{SetupAction.__doc__}
"""
        return SetupObservation(content=content, is_stop=False)
    if action.action == "command":
        session = state["session"]
        result = session.send_command(action.args)
        return SetupObservation(content=result.to_observation(), is_stop=False)
    if action.action == "search":
        result = state["search_tool"].invoke(action.args)
        return SetupObservation(content=json.dumps(result), is_stop=False)
    if action.action == "submit":
        return SetupObservation(content=action.args, is_stop=True)


@auto_catch
def reload_container(state: AgentState) -> dict:
    """
    Start a Docker container with bash session for repository testing.
    
    Args:
        state (AgentState): Agent state containing base image and instance info
        
    Returns:
        dict: Updated state with session and pypiserver
    """
    repo_root = state["repo_root"]
    logger = state["logger"]
    session = SetupRuntime.from_launch_image(
        image_name = state["instance"]["docker_image"],
        instance_id = state["instance"]["instance_id"], 
        platform = state["platform"]
    )

    # clean up repository in the host
    shutil.rmtree(repo_root, ignore_errors=True)
    logger.info(f"Repo root in the host cleaned up: {repo_root}")

    # Setup language-specific environment
    language = state["language"]
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


def analyze_verification_with_llm(llm, submitted_commands: str, verification_output: str) -> bool:
    """
    Use LLM to analyze verification results and determine if rebuild was successful.
    
    Args:
        llm: The language model to use for analysis
        submitted_commands (str): The commands that were submitted for verification
        verification_output (str): The output from executing the commands
        return_code (int): The return code from command execution
        
    Returns:
        bool: True if LLM determines rebuild was successful, False otherwise
    """
    analysis_prompt = f"""You are an expert developer analyzing the results of rebuild command execution.

SUBMITTED COMMANDS:
{submitted_commands}

EXECUTION OUTPUT:
{verification_output}

Your task is to determine whether the rebuild commands executed SUCCESSFULLY or FAILED.

Consider the following:
- Error messages in the output
- Warning messages vs critical errors
- Whether the build/installation actually completed
- Whether dependencies were properly installed
- Whether the project was successfully built

Respond with EXACTLY one of these two words:
- SUCCESS: If the rebuild commands executed successfully and the project is properly built
- FAILURE: If the rebuild commands failed or the project is not properly built

Your response:"""

    try:
        response = llm.invoke([HumanMessage(analysis_prompt)])
        analysis = response.content.strip().upper()
        
        # Return True if LLM says SUCCESS, False otherwise
        return analysis == "SUCCESS"
    except Exception as e:
        # Fallback to return code check if LLM analysis fails
        return analysis == "FAILURE"

@auto_catch
def reload_container(state: AgentState) -> dict:
    """
    Start a Docker container with bash session for repository testing.
    
    Args:
        state (AgentState): Agent state containing base image and instance info
        
    Returns:
        dict: Updated state with session and pypiserver
    """
    repo_root = state["repo_root"]
    logger = state["logger"]
    session = SetupRuntime.from_launch_image(
        image_name = state["instance"]["docker_image"],
        instance_id = state["instance"]["instance_id"], 
        platform = state["platform"]
    )

    # clean up repository in the host
    shutil.rmtree(repo_root, ignore_errors=True)
    logger.info(f"Repo root in the host cleaned up: {repo_root}")

    # Setup language-specific environment
    language = state["language"]
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

@auto_catch
def organize_setup(state: AgentState, max_steps: int, timeout: int = 30) -> dict:
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

    logger.info(f"setup state: {state.get("success" , "false")}, {state["trials"]}, {state["exception"]} ... ")
    hints = "\n\n"
    history_cmds = state["instance"].get("setup_cmds", [])
    history_cmds += state["instance"].get("test_cmds", [])
    platform_hints = ""
    if state["platform"] == "windows":
        platform_hints = f"\n\nNote: This is a windows server image. Use windows powershell command.\n"
    hints += platform_hints

    logger.info("-" * 10 + "Start rebuild conversation" + "-" * 10)
    messages = [
        SystemMessage(system_msg.format(
            commands=history_cmds,
            steps=max_steps,
        )),
        HumanMessage(
            ReAct_prompt.format(
                tools=SetupAction.__doc__,
                project_structure=state["repo_structure"],
                docs=state["docs"],
            ) + hints
        ),
    ]
    prefix_messages = len(messages)
    step = 0
    commands = []
    answer = None
    start_time = time.time()
    while step < max_steps:
        if time.time() - start_time > timeout * 60:
            logger.info(f"Reached global timeout of {timeout} minutes")
            break
        step += 1
        # uses a window to avoid exceed context
        commands_history = HumanMessage(
            f"\nThe previous commands you have run:```\n{commands}```\nFollowing are the last {SETUP_CONVERSATION_WINDOW} messages:\n"
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
            # Agent submitted commands, now automatically verify them
            submitted_commands = observation.content
            logger.info(f"Agent submitted commands: {submitted_commands}")
            logger.info("Automatically verifying submitted commands...")
            
            # Execute the submitted commands to verify they work
            session = state["session"]
            verification_result = session.send_command(submitted_commands)
            verification_output = verification_result.to_observation()
            
            # Use LLM to analyze verification results instead of just checking return code
            verification_success = analyze_verification_with_llm(
                llm, submitted_commands, verification_output
            )
            
            if verification_success:
                # Verification passed according to LLM analysis
                logger.info("Verification PASSED - LLM determined commands executed successfully")
                answer = submitted_commands
                break
            else:
                # Verification failed according to LLM analysis
                logger.info("Verification FAILED - LLM determined commands failed")
                verification_message = HumanMessage(
                    f"ENFORCED VERIFICATION FAILED:\n"
                    f"Your submitted commands: {submitted_commands}\n"
                    f"Execution output:\n{verification_output}\n"
                    f"An expert analysis of the output indicates the rebuild was not successful. "
                    f"Please analyze the error and provide different rebuild commands that will work. "
                    f"Be sure to verify on your own before submission."
                )
                logger.info("\n" + verification_message.pretty_repr())
                messages.append(verification_message)
                continue
                
        message = HumanMessage(f"Observation:\n{observation.content}")
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End rebuild organization conversation" + "-" * 10)
    return {
        "session": state["session"],
        "messages": messages,
        "commands": commands,
        "setup_messages": messages[prefix_messages:],
        "setup_commands": [answer] if answer else [],
        "success": (answer is not None)
    }

