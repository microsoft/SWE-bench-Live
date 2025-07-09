# custimized for out task
ReAct_prompt = """
You run in a loop of Thought, Action, Observation.
At the end of the loop you should use Action to stop the loop.

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.
> Important Note: Each step, reply with only **one** (Thought, Action) pair.
> Important Note: Do not reply **Observation**, it will be provided by the system.

Your available actions are:
{tools}

Observation will be the result of running those actions.


Project Structure: the structure of the project, including files and directories.
Related Files: the content of related files of the project that may help you understand the project.
Thought: you should always think about what to do
Action: <command>your bash command here</command> or <search>your search query</search> or other actions available
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times) ...
Thought: I think the setup should be fine
Action: <stop>stop the setup</stop>
Answer: the final result

Begin!
Project Structure: {project_structure}
Related Files: {docs}
"""

EnvironmentSpec = """
A working development environment should:
- Correctly exported environment variables
- Have successfully compiled/installed the project **from source**
- Able to run all tests without errors

For develop environment, following points can be omitted:
- Lint errors (e.g. mypy type checking or linting)
- Warnings
"""
