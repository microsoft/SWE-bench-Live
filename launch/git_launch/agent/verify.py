"""
Environment verification agent for testing repository setup correctness.
"""
from typing import Any, Literal

from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from git_launch.agent.prompt import ReAct_prompt
from git_launch.agent.state import AgentState, auto_catch
from git_launch.runtime import SetupRuntime

system_msg = """You are a developer. Your task is to verify whether the environment for the given project is set up correctly. Your colleague has set up a Docker environment for the project. You need to verify if it can successfully run the tests of the project.
- You interact with a Bash session inside this container.
- The container is based on {base_image}.
- The setup commands that your colleague has run are {setup_commands}
- Project files are located in /testbed within the container, and your current working directory of bash is already set to /testbed.
- Use the same test framework as your colleague, because that aligns with the setup stage.
- Only test commands, skip linting/packaging/publishing commands.
- Do not change the state of the environment, your task is to verify not to fix it. If you see issues, report it not fix it.
- You can tolerate a few test cases failures—as long as most tests pass, it's good enough. """

system_msg += """\
## Important Note:

Your test command must output detailed pass/fail status for each test item. This is mandatory. For example, with pytest, use the -rA option to get output like:
```
...
PASSED tests/test_resources.py::test_fetch_centromeres
PASSED tests/test_vis.py::test_to_ucsc_colorstring
```
Since we need to parse the test output to extract a test item → status mapping, **this requirement is mandatory**. If you observed that your test command does not produce such detailed output, you must adjust it accordingly.

In summary, your goal is:
1. Write the test commands that could output detailed pass/fail status for each test item, you can iterate until it does. (this is mandatory, DO NOT ignore this requirement!!! This is your obligation to correctly identify the test commands to run the test suite of the project, and find a way to output detailed pass/fail status)
2. Run the test command to verify if the environment is set up correctly. If not, report any observed issues. If you think the setup is correct, report none issue.
"""


class VerifyAction(BaseModel):
    """
    Command: run a command in the bash, reply with following format, your command should not require sudo or interactive input:
        <command>...</command>
        e.g. run pytest with detailed output turned on: <command>pytest -rA</command>
        e.g. <command>tox -- -rA</command>
    Issue: stop the verify loop once you think the setup is complete, and reply with the issue of the setup:
        <issue>...</issue>
        e.g. <issue>some dependency is missing, run `pytest` failed</issue>
        e.g. <issue>None</issue> if you think the setup is correct (remember to tolerate a few test cases failures as long as most tests pass)
    """

    action: Literal["command", "issue"] = Field(
        "command", description="The action type"
    )
    args: Any = Field(None, description="The action arguments")


class VerifyObservation(BaseModel):
    """Observation for the setup action"""

    content: str = Field("", description="The content of the observation")


def parse_verify_action(response: str) -> VerifyAction | None:
    """
    Parse verification action from LLM response text.
    
    Args:
        response (str): Raw LLM response text
        
    Returns:
        VerifyAction | None: Parsed action or None if parsing failed
    """
    if "<command>" in response:
        command = response.split("<command>")[1].split("</command>")[0].strip()
        return VerifyAction(action="command", args=command)
    elif "<issue>" in response:
        issue = response.split("<issue>")[1].split("</issue>")[0].strip().lower()
        return VerifyAction(action="issue", args=issue)
    else:
        return None


def observation_for_verify_action(
    action: VerifyAction | None, session: SetupRuntime
) -> VerifyObservation:
    """
    Execute verification action and return observation.
    
    Args:
        action (VerifyAction | None): Action to execute
        session (SetupRuntime): Runtime session for command execution
        
    Returns:
        VerifyObservation: Result of action execution
    """
    if not action:
        content = f"""\
Please using following format after `Action: ` to make a valid action choice:
{VerifyAction.__doc__}
"""
        return VerifyObservation(content=content, success=False)
    if action.action == "command":
        result = session.send_command(action.args)
        return VerifyObservation(content=result.to_observation(), success=False)
    if action.action == "issue":
        if action.args == "none":
            return VerifyObservation(content="", success=True)
        else:
            return VerifyObservation(content=action.args, success=False)


VERIFY_CONVERSATION_WINDOW = 10


@auto_catch
def verify(max_steps: int, state: AgentState):
    """
    ReAct agent for environment verification through test command execution.
    
    Args:
        max_steps (int): Maximum number of verification steps allowed
        state (AgentState): Current agent state with setup results
        
    Returns:
        dict: Updated state with verification results and success status
    """
    if state["exception"]:
        raise state["exception"]

    session = state["session"]
    llm = state["llm"]
    logger = state["logger"]
    setup_commands = state["setup_commands"]
    logger.info("-" * 10 + "Start verify conversation" + "-" * 10)
    messages = [
        SystemMessage(
            system_msg.format(
                base_image=state["base_image"], setup_commands=setup_commands
            )
        ),
        HumanMessage(
            ReAct_prompt.format(
                tools=VerifyAction.__doc__,
                project_structure=state["repo_structure"],
                docs=state["docs"],
            )
        ),
    ]
    prefix_messages = len(messages)
    commands = []
    step = 0
    success = False
    issue = None
    while step < max_steps:
        step += 1
        # uses a window to avoid exceed context
        if len(messages) < VERIFY_CONVERSATION_WINDOW + prefix_messages:
            input_messages = messages
        else:
            input_messages = (
                messages[:prefix_messages] + messages[-VERIFY_CONVERSATION_WINDOW:]
            )
        response = llm.invoke(input_messages)
        # for reasoning model
        if "<think>" in response.content:
            response.content = response.content.split("</think>")[1]
        # print(response.pretty_repr())
        logger.info(response.pretty_repr())
        messages.append(response)
        action = parse_verify_action(response.content)
        if action.action == "command":
            commands.append(action.args)
        observation = observation_for_verify_action(action, session)
        message = HumanMessage(f"Observation:\n{observation.content}")
        # print(message.pretty_repr())
        logger.info(message.pretty_repr())
        messages.append(message)
        if action.action == "issue":
            if observation.content == "":
                success = True
                logger.info("The setup is successful")
                break
            issue = observation.content
            logger.info(f"Verification failed due to: {issue}")
            break

    trials = state["trials"] + 1
    logger.info("-" * 10 + "End verify conversation" + "-" * 10)
    return {
        "messages": messages,
        "verify_messages": messages[prefix_messages:],
        "test_commands": commands,
        "commands": commands,
        "trials": trials,
        "success": success,
        "issue": issue,
    }
