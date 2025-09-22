"""
Environment verification agent for testing repository setup correctness.
"""
import json
import time
from typing import Any, Literal

from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from launch.agent.action_parser import ActionParser
from launch.agent.prompt import ReAct_prompt
from launch.agent.state import AgentState, auto_catch
from launch.core.runtime import SetupRuntime

from launch.scripts.parser import run_get_pertest_cmd

system_msg: str = """
You are a developer. You have already setup all dependencies and build the repository in the current folder.

- You are inside a docker container with the source code already inside the container under the current directory called /testbed
- The dependencies of the repository have already been set up by you before.
- The command to build the repository: {setup_cmd}
- The command to run all the test cases in the repository: {test_cmd} 
- The python script to parse the test output: <python>{parser}</python>
- The list of testcases discovered by the parser: {test_status}

In summary, your task is:
1. Proceed to figure out whether each test case can be run and output the result of fail/pass/skip to console separately.
You can try to run some shell commands to explore how to specify one signle test case to run and print its pass/fail status to console.
Note: If one test suite outputs a grouped result, try your best to run each testcase separately. If you cannot do that, running one test suite at a time is also acceptable.
2. Once you have found the commands to run one specific testcase and print test result to console, write a python function to receive the input of the list of testcase names and output the mapping of the commands to run each testcase with verbose output in the format:
{{
"Test_parseArgs_new/unrecognized_flag" : "go test -v -run 'Test_parseArgs_new/unrecognized_flag'",
"Test_homeDir/none_available" : "go test -v -run 'Test_homeDir/none_available'",
}}

You need to do it in {steps} steps.
"""


class VerifyAction(BaseModel):
    """
Command: run a command in the shell, reply with following format, your command should not require sudo/admin privilage or interactive input:
    <command>...</command>
    e.g. <command>dotnet test /testbed/Test/Mono.Cecil.Tests.csproj --no-build -c Debug --filter "FullyQualifiedName=Mono.Cecil.Tests.TypeParserTests.ConstField"</command>
    e.g. <command>mvn test  -Dtest=ProjectBuildListTest#testGetByTaskSegment  -Dsurefire.printSummary=true  -Dsurefire.useFile=false  -DtrimStackTrace=false</command>
    e.g. <command>pytest -v tests/test_valid_requests.py::test_http_parser['/testbed/tests/requests/valid/025.http']</command>
Search: search the web if you need some information, generate query and reply with following format:
    <search>...</search>
    e.g. <search>how to fix 'No module named setuptools'</search>
Generate: Write a python script to generate a function to map each test case to the command to execute it separately with test result output to console  
    <python>def get_pertest_cmd(testcase_list:list[str])->dict[str,str]:\n\treturn</python>
    The "testcase_list" argument is the list of passed testcase names. The system will pass the testcase list into the function you define and give you execution results, so you only need to define the get_pertest_cmd function
    The example generating command: <python>get_pertest_cmd(testcase_list:list[str])->dict[str,str]:
    result: dict[str, str] = {}
    for testcase in testcase_list:
        safe_testcase = testcase.replace("::", "#")
        result[testcase] = f"mvn test  -Dtest={safe_testcase}  -Dsurefire.printSummary=true  -Dsurefire.useFile=false  -DtrimStackTrace=false"
    return result</python>
Submit: Stop the exploration if you find the commands to run each testcase separately with verbose output and you think your python script to generate the command list is correct.
    You do not need to output anything at this step because we would re-use your last python script and generated command list.
    Only output <submit>success</submit>
    If the repo really cannot specify one testcase to run, output <submit>failure</submit> honestly.
    """

    action: Literal["command", "search", "python", "submit"] = Field(
        "command", description="The action type"
    )
    args: Any = Field(None, description="The action arguments")


class SetupObservation(BaseModel):
    """Observation for the setup action"""

    content: str = Field("", description="The content of the observation")
    is_stop: bool = Field(False, description="Whether stop the setup loop")


class VerifyActionParser(ActionParser):
    """Parser for setup agent actions."""
    
    def parse(self, response: str) -> VerifyAction | None:
        """Parse setup action from LLM response text."""
        response = self.clean_response(response)
        
        submit = self.extract_tag_content(response, "submit")
        if submit:
            return VerifyAction(action="submit", args=submit)

        script = self.extract_tag_content(response, "python")
        if script:
            return VerifyAction(action="python", args=script)

        command = self.extract_tag_content(response, "command")
        if command:
            return VerifyAction(action="command", args=command)
            
        search = self.extract_tag_content(response, "search")
        if search:
            return VerifyAction(action="search", args=search)
            
        return None


def parse_verify_action(response: str) -> VerifyAction | None:
    """Parse setup action from LLM response text."""
    parser = VerifyActionParser()
    return parser.parse(response)



VERIFY_CONVERSATION_WINDOW = 40


@auto_catch
def organize_unit_test(state: AgentState, max_steps: int, timeout: int = 30) -> dict:
    """
    ReAct agent for environment verification through test command execution.
    
    Args:
        max_steps (int): Maximum number of verification steps allowed
        state (AgentState): Current agent state with setup results
        
    Returns:
        dict: Updated state with verification results and success status
    """

    pertest_command: str = "{{}}"
    test_status: dict[str,str] = state["test_status"]
    testcase_list: list[str] = [i for i in test_status.keys()]
    parser: str = ""

    def observation_for_verify_action(
        state: AgentState, action: VerifyAction | None
    ) -> SetupObservation:
        """
        Execute setup action and return observation.
        
        Args:
            state (AgentState): Current agent state
            action (VerifyAction | None): Action to execute
            
        Returns:
            SetupObservation: Result of action execution
        """
        nonlocal testcase_list, parser, pertest_command

        if not action or not action.action:
            content = f"""Please using following format after `Action: ` to make a valid action choice: \n{VerifyAction.__doc__}"""
            return SetupObservation(content=content, is_stop=False)
        if action.action == "command":
            session = state["session"]
            result = session.send_command(action.args)
            return SetupObservation(content=result.to_observation(), is_stop=False)
        if action.action == "python":
            parser = action.args
            generation_result: dict[str, str] = run_get_pertest_cmd(action.args, testcase_list)
            pertest_command = json.dumps(generation_result)
            test_commands = " ; ".join(generation_result.values())
            session = state["session"]
            execution_result = session.send_command(test_commands)
            execution_result = execution_result.to_observation()
            result = f"""
This is the "testcase name :: per-testcase excution command" mapping returned from your python script: 
{generation_result}
==========================
This is the shell execution result of all of the pertest execution commands returned from your python script:
{execution_result}
==========================
Please judge the correctness of your per-testcase execution commands (the commands should run one specified testcase per command, and run the specified testcase successfully with pass/fail/skip result).
If correct, submit with <submit>success</submit>;
If not correct, please explore the correct command to run a specific testcase again and write the python script to generate all per-testcase commands again;
If you think it is impossible to run each testcase separately, give up by outputting <submit>failure</submit>.
"""
            return SetupObservation(content=result, is_stop=False)
        if action.action == "search":
            result = state["search_tool"].invoke(action.args)
            return SetupObservation(content=json.dumps(result), is_stop=False)
        if action.action == "submit":
            if ("success" in action.args) and (len(json.loads(pertest_command)) == 0):
                observation = "You submit your answer with <submit>success</submit>. But we cannot find any correct per-testcase execution commands in history. Please explore the correct per-testcase commands again and write the python script to generate all per-testcase commands again. If you find it is impossible to run a specific testcase, output <submit>failure</submit> instead."
                return SetupObservation(content=observation, is_stop=False)
            return SetupObservation(content=action.args, is_stop=True)



    if state["exception"]:
        raise state["exception"]

    hints = "\n\n"
    session = state["session"]
    llm = state["llm"]
    logger = state["logger"]

    logger.info(f"setup state: {state.get("success" , "false")}, {state["trials"]}, {state["exception"]} ... ")
    hints = "\n\n"
    platform_hints = ""
    if state["platform"] == "windows":
        platform_hints = f"\n\nNote: This is a windows server image. Use windows powershell command.\n"
    hints += platform_hints
        
    messages = [
        SystemMessage(
            system_msg.format(
               setup_cmd=state["setup_commands"],
               test_cmd=state["test_commands"],
               parser=state["parser"],
               test_status=state["test_status"],
               steps=max_steps,
            )
        ),
        HumanMessage(
            ReAct_prompt.format(
                tools=VerifyAction.__doc__,
                project_structure=state["repo_structure"],
                docs=state["docs"],
            ) + hints
        ),
    ]
    prefix_messages = len(messages)
    commands = []
    step = 0
    start_time = time.time()
    success = False
    logger.info("-" * 10 + "Start unit test conversation" + "-" * 10)
    while step < max_steps:
        if time.time() - start_time > timeout * 60:
            logger.info(f"Reached global timeout of {timeout} minutes")
            break
        step += 1
        # uses a window to avoid exceed context
        if len(messages) < VERIFY_CONVERSATION_WINDOW + prefix_messages:
            input_messages = messages
        else:
            input_messages = (
                messages[:prefix_messages] + messages[-VERIFY_CONVERSATION_WINDOW:]
            )
        response = llm.invoke(input_messages)

        logger.info("\n" + response.pretty_repr())
        messages.append(response)
        action = parse_verify_action(response.content)
        observation = observation_for_verify_action(state, action)
        if observation.is_stop:
            success = "success" in observation.content
            break
        if action and action.action == "command":
            commands.append(action.args)
        message = HumanMessage(f"Observation:\n{observation.content}")
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End verify conversation" + "-" * 10)
    if success:
        try:
            pertest_command_dict = json.loads(pertest_command)
        except:
            pertest_command_dict = {}
    else:
        pertest_command_dict = {}
    return {
        "messages": messages,
        "commands": commands,
        "unittest_generator": parser if success else "",
        "pertest_command": pertest_command_dict,
        # "success": success, # We decide not to count the success of this optional step into overall success
    }
