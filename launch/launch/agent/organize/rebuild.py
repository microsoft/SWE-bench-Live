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


system_msg = """You are a developer. You have already setup all dependencies and build the repository in the current folder.
However, for the maintainance of the project, you need to organize the minimal commands to re-install ONLY modified packages and build the projects again after edits to the source code / package list.

- You are inside a docker container with source code already inside the container under the current directory called /testbed
- The dependencies of the repository have already been set up by you before.
- The full history commands that you used to try to set up the repo: {commands}

You can send commands in the container for several times to try to test the commands to re-build the repo and expolre the repo freely if you need more information.
You do not need to include the commands to run test cases because we will do it later.

The final objective is: 
    to "find the minimal commands to re-install ONLY modified packages AND re-build the project" again after package list / source code edits and "output your minimal re-install & re-build commands in one line".
You need to finish it in {steps} steps.
"""

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




SETUP_CONVERSATION_WINDOW = 40


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
    state["session"] = SetupRuntime.from_launch_image(
        image_name = state["instance"]["docker_image"],
        instance_id = state["instance"]["instance_id"], 
        platform = state["platform"]
    )

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
            answer = observation.content
            break
        message = HumanMessage(f"Observation:\n{observation.content}")
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End rebuild organization conversation" + "-" * 10)
    return {
        "session": state["session"],
        "messages": messages,
        "commands": commands,
        "setup_messages": messages[prefix_messages:],
        "setup_commands": [answer],
        "success": (answer is not None),
    }

