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

from launch.scripts.parser import run_parser

system_msg: str = """You are a developer. You have already setup all dependencies and build the repository in the current folder.
However, for the maintainance of the project, you need to organize the minimal commands to run all test commands from scratch again and get the verbose output of "test_case -- pass/fail/skip" pairs. Then you need to write a python script to parse the test output.

- You are inside a docker container with the source code already inside the container under the current directory called /testbed
- The dependencies of the repository have already been set up by you before.
- The full history commands that you used to try to set up and test the repo: {commands}. These tiral commands may be insufficient to discover all testcases. You need to find as many testcases to run as possible.
- You have organized the minimal commands to re-build the repository again after edits to the source code: {setup_cmd} and have run the them, so you don't need to build the repository again. You should only output commands to run test cases.

## Note

Your test command must output detailed pass/fail status for each test item. This is mandatory. For example, with pytest, use the -rA option to get output like:
```
PASSED tests/test_resources.py::test_fetch_centromeres
PASSED tests/test_vis.py::test_to_ucsc_colorstring
```
or 
```
tests/test_resources.py::test_fetch_centromeres.............OK
tests/test_vis.py::test_to_ucsc_colorstring......... ERROR
```

Since we need to parse the test output to extract a test item -> status mapping, **this requirement is mandatory**. 
If you observed that your test command does not produce such detailed output (test_case_name -> pass/fail/skip mapping), you must adjust it accordingly.
If test results are written to a file not print to stdout, then find the file and output its content to console (with cat command etc.) to verify. 
Do not print only the head and tail of the output file to console (do not use head 50 / tail 50), we need to see ALL testcase results.


## Note
If one test suite outputs a grouped result, try your best to reveal the result of EACH testcase under this test suite. 
If you cannot figure out, taking a test suite as one testcase is also acceptable. In this case, mark a suite with some testcases failed / error as fail, with all test cases skipped as skip, otherwise as pass.

## Note
In some corner cases the re-build commands already run all testcases directly. 
Since later we would run re-build commands + test commands together, if you can find the testcase - status mapping in the result files of your re-build commands, you only need to submit shell commands to VIEW / REVEAL the mapping (and of course the parser script) at this step.

## Note
When trying the python script as the parser, you only need to output the parsing function parser(log:str)->dict[str, str].
Each parser trial would return script execution results, and you can iterate until it meets the requirements.


In summary, your goal is:
1. Try to find the test commands that could run ALL testcases (or as much as possible) from scratch and output detailed fail/pass/skip status for each testcase, you can iterate until it does. (this is mandatory!!!)
2. Try to write a short python script to parse the test output to get a python dict mapping each test case name to fail/pass/skip status strictly in the python dict format:
{{
    "tests/test_resources.py::test_fetch_centromeres": "pass",
    "tests/test_vis.py::test_to_ucsc_colorstring": "fail",
    "tests/model/model_utils/test_visual.py::test_visual_full[True]": "skip",
}}

You need to finish it in {steps} steps.
"""


class VerifyAction(BaseModel):
    """
Command: run a command in the shell, reply with following format, your command should not require sudo/admin privilage or interactive input:
    <command>...</command>
    e.g. <command>pytest -rA</command>
    e.g. <command>tox -- -rA</command>
Search: search the web if you need some information, generate query and reply with following format:
    <search>...</search>
    e.g. <search>how to fix 'No module named setuptools'</search>
Parse: parse the test output with python script, wrap your python script in  
    <python>def parser(log:str)->dict[str, str]:\n\rreturn</python>
    The "log" argument is the test case output, the system will pass the concatenated history message into the function you define and give you execution results, so you only need to submit the python script
    The history log would contain much noise, so you'd better use regex to parse the log, you must only use built-in python libs
    The first example parse script: <python>def parser(log: str) -> dict[str, str]:
    import re
    result: dict[str, str] = {}
    for line in log.splitlines():
        # match test lines like: tests/foo/bar.py::test_name PASSED ...
        m = re.search(r'(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS)', line)
        if m:
            test, status = m.groups()
            status = status.upper()
            if status == "PASSED":
                result[test] = "pass"
            elif status in ("FAILED", "XFAIL", "XPASS"):
                result[test] = "fail"
            elif status == "SKIPPED":
                result[test] = "skip"
    return result</python>
    The second example parse script: <python>def parser(log: str) -> dict[str, str]:
    import re
    from typing import Dict
    test_header_re = re.compile(r"^\s*-{3,}\s*$|^\s*Test set:\s+(.+?)\s*$")
    # Typical line: ------------------------------------------------------------------------------- Tests run: 11, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 2.025 s -- in org.asynchttpclient.ws.WebSocketWriteFutureTest
    summary_re = re.compile(
        r"Tests run:\s*(\d+)\s*,\s*Failures:\s*(\d+)\s*,\s*Errors:\s*(\d+)\s*,\s*Skipped:\s*(\d+)",
        re.IGNORECASE,
    )

    results: Dict[str, str] = {}
    current_suite: str | None = None

    for line in log.splitlines():
        # Detect the 'Test set:' header and capture suite name
        m = test_header_re.match(line)
        if m:
            suite = m.group(1)
            if suite:  # it's a 'Test set:' line (not a dashed separator)
                current_suite = suite.strip()
            continue

        if current_suite is None:
            continue  # not inside a suite block yet

        # Parse the first summary line encountered after the current suite header
        s = summary_re.search(line)
        if s:
            tests_run = int(s.group(1))
            failures = int(s.group(2))
            errors = int(s.group(3))
            skipped = int(s.group(4))

            if failures > 0 or errors > 0:
                status = "fail"
            elif tests_run == 0 and skipped > 0:
                status = "skip"
            else:
                status = "pass"

            results[current_suite] = status
            current_suite = None  # reset until next 'Test set:' header

    return results</python>

Submit: Stop the exploration if you find BOTH the minimal commands to run all test cases and expose the per-testcase results to console AND the minimal python script to parse test output into test_name : status dict.
    You only need to submit the minimal commands to run all test cases and expose the per-testcase results to console in one line in the format:
    <submit>your final test commands in one line separated by ;</submit>
    We would re-use the last python parser script you wrote, so you do not need to output it again.
    For example:
    <submit>./mvnw test -B -Dsurefire.printSummary=true ; cat flink-core/target/surefire-reports/*.txt</submit>
    <submit>./eng/common/cibuild.sh -configuration Release -prepareMachine ; cat ./artifacts/TestResults/Release/AspNet.Security.OAuth.Providers.Tests_net9.0_x64.xml</submit>
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
def organize_test_cmd(state: AgentState, max_steps: int, timeout: int = 30) -> dict:
    """
    ReAct agent for environment verification through test command execution.
    
    Args:
        max_steps (int): Maximum number of verification steps allowed
        state (AgentState): Current agent state with setup results
        
    Returns:
        dict: Updated state with verification results and success status
    """

    test_output: str = ""
    test_status: str = ""
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
        nonlocal test_output

        if not action or not action.action:
            content = f"""Please using following format after `Action: ` to make a valid action choice: \n{VerifyAction.__doc__}"""
            return SetupObservation(content=content, is_stop=False)
        if action.action == "command":
            session = state["session"]
            result = session.send_command(action.args)
            return SetupObservation(content=result.to_observation(), is_stop=False)
        if action.action == "python":
            result = run_parser(action.args, test_output)
            return SetupObservation(content=json.dumps(result), is_stop=False)
        if action.action == "search":
            result = state["search_tool"].invoke(action.args)
            return SetupObservation(content=json.dumps(result), is_stop=False)
        if action.action == "submit":
            if not parser:
                content = "Warning: It seems you have not written python scripts to parse the test output. Please try to write a python script to parse the test output! If you really cannot get per-testcase status, write an empty python function <python>def parser(log:str)->dict[str, str]:\n\treturn dict()</python> to indicate failure at the next step, and then submit again at the second step."
                return SetupObservation(content=content, is_stop=False)
            return SetupObservation(content=action.args, is_stop=True)

    if state["exception"]:
        raise state["exception"]

    hints = "\n\n"
    session = state["session"]
    llm = state["llm"]
    logger = state["logger"]
    setup_commands = state["setup_commands"]

    logger.info(f"setup state: {state.get("success" , "false")}, {state["exception"]} ... ")
    hints = "\n\n"
    history_cmds = state["instance"].get("setup_cmds", [])
    history_cmds += state["instance"].get("test_cmds", [])
    platform_hints = ""
    if state["platform"] == "windows":
        platform_hints = f"\n\nNote: This is a windows server image. Use windows powershell command.\n"
    hints += platform_hints
        
    messages = [
        SystemMessage(
            system_msg.format(
               commands=history_cmds,
               setup_cmd=setup_commands,
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
    answer = None
    start_time = time.time()
    logger.info("-" * 10 + "Start test conversation" + "-" * 10)
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
            answer = observation.content
            break
        if action and action.action == "command":
            commands.append(action.args)
            test_output += observation.content
        if action and action.action == "python":
            parser = action.args
            test_status = observation.content
        message = HumanMessage(f"Observation:\n{observation.content}")
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End verify conversation" + "-" * 10)
    try:
        test_status = json.loads(test_status)
    except:
        test_status = None
    return {
        "messages": messages,
        "verify_messages": messages[prefix_messages:],
        "test_commands": [answer],
        "commands": commands,
        "parser": parser,
        "test_status": test_status,
        "success": bool(answer and parser and test_status),
    }
