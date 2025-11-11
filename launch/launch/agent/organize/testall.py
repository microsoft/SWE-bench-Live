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
from launch.utilities.language_handlers import get_language_handler

from launch.scripts.parser import run_parser

system_msg: str = """You are a developer. You have already setup all dependencies and build the repository in the current folder.
Your task is to organize the MINIMAL test commands to run ALL test cases and write a python script to parse the output and extract test case statuses.

## Environment
- You are inside a docker container with the source code at /testbed
- Dependencies are already installed
- Previous setup commands: {commands}
- Re-build commands (already executed): {setup_cmd}
- You only need to output test commands now

## THREE STEPS TO COMPLETE

### STEP 1: Find Minimal Test Command & Save Output to File
Find a test command that prints out all test cases. **IMPORTANT: Whenever possible, try to output in structured format (JSON, XML, or other machine-readable format).**

**Preferred: Use structured output (JSON/XML/etc) if framework supports it:**
- Submit a test command that saves results in JSON or XML format to a file
- Structured formats make parsing much easier and more reliable
**Fallback: If NO structured format is available, use verbose text output:**
- Submit a test command that captures all test output to a file with verbose logging

**Requirements for Step 1:**
- The command must run ALL test cases (not a subset)
- Output must expose EVERY test case with its status (pass/fail/skip)
- Output must be saved to a file (use `> file.txt` or `| tee file.txt`)
- **Strongly prefer structured formats (JSON/XML) over raw logs for easier and more reliable parsing**

{test_cmd_hints}

### STEP 2: Print Out All Logs from the File
After Step 1 completes, submit a command to print out all the logs/results from the file.
- Example: `cat jest-results.json`
- Example: `cat test-output.log`
- Example: `cat report.json`

This step extracts the content from the file saved in Step 1.

### STEP 3: Write Python Parser
Write a python script to extract each test case and its result (pass/fail/skip) from the logs. 
Make sure it actually run the parser to validate.

The parser function should:
- Take the log output as input
- Parse JSON/XML if available, otherwise parse text logs with regex
- Extract test case names and their status (pass/fail/skip)
- Return a dictionary mapping test names to status strings

## Important Notes
- Step 1: Run tests and save to file (do NOT print to console in this step)
- Step 2: Print the file contents (this is what parser will receive)
- Step 3: Parse the output to extract test -> status mapping
- If test framework groups results by suite, extract per-test status where possible
- Suppress unnecessary console noise (spinners, warnings) when saving to file in Step 1

## Your Three Goals
1. Submit test command that saves output to a file (Step 1)
2. Submit command to print the file contents (Step 2)
3. Write python script to extract test_name to status mapping from the output (Step 3)

## Output Format for Submission
Submit: THREE submissions required to complete:

    SUBMISSION 1 - Test command (STEP 1): Run tests and save output to file
    <submit>COMMAND_TO_RUN_TESTS_AND_SAVE_OUTPUT</submit>
    Examples:
        <submit>npm run test:unit -- --json --outputFile=jest-results.json</submit>
        <submit>pytest tests/ --json-report --json-report-file=report.json</submit>
        <submit>./mvnw test -B -Dsurefire.printSummary=false 2>&1 | tee test-output.log</submit>
        <submit>go test ./... -v 2>&1 | tee test-output.log</submit>
    
    SUBMISSION 2 - Print command (STEP 2): Print the file contents
    <submit>COMMAND_TO_PRINT_THE_OUTPUT_FILE</submit>
    Examples:
        <submit>cat jest-results.json</submit>
        <submit>cat report.json</submit>
        <submit>cat test-output.log</submit>
        <submit>cat flink-core/target/surefire-reports/*.txt</submit>
    
    SUBMISSION 3 - Validate parser (STEP 3): Write and execute the parser
    Write and execute a Python parser to extract test cases and their status from the output.
    The parser will be automatically validated when you execute it.
    <python>def parser(log: str) -> dict[str, str]:
    # Extract test cases and their status from the printed output
    import re
    results = {{}}
    for line in log.splitlines():
        m = re.search(r'(\S+)\s+(PASSED|FAILED|SKIPPED)', line)
        if m:
            test, status = m.groups()
            results[test] = 'pass' if 'PASSED' in status else 'fail' if 'FAILED' in status else 'skip'
    return results</python>
    
    When parser is executed and validated successfully, you should invoke submit action and the process finishes.

You need to finish in {steps} steps.
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

Submit: TWO submissions required (parser is auto-stored):
    
    SUBMISSION 1 - Test command (STEP 1): Run tests and save output to file
    <submit>npm run test:unit -- --json --outputFile=jest-results.json</submit>
    <submit>pytest tests/ --json-report --json-report-file=report.json</submit>
    <submit>go test ./... -v 2>&1 | tee test-output.log</submit>
    
    SUBMISSION 2 - Print command (STEP 2): Print the file contents
    <submit>cat jest-results.json</submit>
    <submit>cat report.json</submit>
    <submit>cat test-output.log</submit>
    
    EXECUTION 3 - Execute parser (STEP 3): Write and execute parser to finish
    After you validate your log parser, invoke submit action to finish the process
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
    test_command: str = ""  # Store STEP 1: test command
    print_command: str = ""  # Store STEP 2: print/cat command
    submitted_steps: int = 0  # Track submission count (1 or 2)

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
        nonlocal test_output, test_status, test_command, print_command, submitted_steps

        if not action or not action.action:
            content = f"""Please using following format after `Action: ` to make a valid action choice: \n{VerifyAction.__doc__}"""
            return SetupObservation(content=content, is_stop=False)
        if action.action == "command":
            session = state["session"]
            result = session.send_command(action.args)
            test_output += result.output # This is full (unstripped) command output
            return SetupObservation(content=result.to_observation(), is_stop=False) # content is trucated history
        if action.action == "python":
            if print_command != "":
                session = state["session"]
                test_output = session.send_command(print_command).output        
            result = run_parser(action.args, test_output)
            test_status = json.dumps(result, indent = True)
            truncated_result = test_status
            if len(truncated_result) > 40000:
                truncated_result = truncated_result[:40000] + "\n...result truncated due to length...\n"
            content = f"""
Below is the execution result of your Python script:
{truncated_result}
Please judge whether:
(1) Your Python script extracts the statuses of all testcases and 
(2) Your script and test command have made best effort to reveal the status of each unit testcase under a test suite / set if there's test suite / set. The testcase names in your output should be split to the finest granularity.
If not successful, please adjust your test command, run your new test command in container, and adjust your Python script to extract test status.
If successful, please submit.
"""
            return SetupObservation(content=content, is_stop=False)
        if action.action == "search":
            result = state["search_tool"].invoke(action.args)
            return SetupObservation(content=json.dumps(result), is_stop=False)
        if action.action == "submit":
            submitted_steps += 1
            
            # SUBMISSION 1: Test command (STEP 1) - runs tests and saves to file
            if submitted_steps == 1:
                test_command = action.args
                content = f"""Received STEP 1 test command:
{test_command}

Now please explore STEP 2 and submit: a command to print/cat the output file.
For example:
<submit>cat jest-results.json</submit>
or
<submit>cat test-output.log</submit>
"""
                return SetupObservation(content=content, is_stop=False)
            
            # SUBMISSION 2: Print command (STEP 2) - prints file contents
            elif submitted_steps == 2:
                print_command = action.args
                content = f"""Received STEP 2 print command:
{print_command}

Now please write a Python parser to extract test cases and their status from the output.
Make sure to run the parser to validate. The parser will be automatically stored when you execute it.
Use: <python>def parser(log:str)->dict[str, str]: ... return results</python>
"""
                return SetupObservation(content=content, is_stop=False)
            
            # SUBMISSION 3: Check if parser was executed and results obtained
            elif submitted_steps == 3:
                if parser and test_status:
                    # Parser was executed and produced results
                    content = "Parser validated. Test analysis complete."
                    return SetupObservation(content=content, is_stop=True)
                else:
                    # Parser was not executed or produced no results
                    content = "Note: Parser have not yet been validated. Please verify your parser runs correctly before submission to finish the process."
                    return SetupObservation(content=content, is_stop=False)

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
    
    language = state["language"]
    language_handler = get_language_handler(language)
    test_cmd_hints = language_handler.get_test_cmd_instructions()
    messages = [
        SystemMessage(
            system_msg.format(
               commands=history_cmds,
               setup_cmd=setup_commands,
               test_cmd_hints=test_cmd_hints,
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
        if action and action.action == "python":
            parser = action.args
        message = HumanMessage(f"Observation:\n{observation.content}")
        logger.info("\n" + message.pretty_repr())
        messages.append(message)

    logger.info("-" * 10 + "End verify conversation" + "-" * 10)
    try:
        test_status = json.loads(test_status)
    except:
        test_status = None
    
    # # Combine test and print commands into final list
    # final_commands = []
    # if test_command:
    #     final_commands.append(test_command)
    # if print_command:
    #     final_commands.append(print_command)
    
    # # If we got both test and print commands, use them
    # # Otherwise fall back to answer (for backwards compatibility)
    # if final_commands:
    #     test_commands = final_commands
    # else:
    #     test_commands = [answer] if answer else []
    
    return {
        "messages": messages,
        "verify_messages": messages[prefix_messages:],
        "test_commands": [test_command],
        "print_commands": [print_command],
        "commands": commands,
        "parser": parser,
        "test_status": test_status,
        "success": bool((test_command or answer) and parser and test_status),
    }