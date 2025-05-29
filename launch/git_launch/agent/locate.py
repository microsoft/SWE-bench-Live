"""
Repository analysis agent for locating environment setup documentation.
"""
import os

from langchain.schema import HumanMessage

from git_launch.agent.state import AgentState, auto_catch
from git_launch.utilities.get_repo_structure import view_repo_structure

prompt = """Given this repository structure:
------ BEGIN REPOSITORY STRUCTURE ------
{structure}
------ END REPOSITORY STRUCTURE ------

List the most relevant files for setting up a development environment, including:
0. CI/CD configuration files
1. README files
2. Documentation
3. Installation guides
4. Development setup guides

Only list files that are critical for understanding project dependencies and setup requirements.
Format each file with its relative path (relative to project root) to be wrapped with tag <file> </file>, one per line."""


determine_prompt = """Given a file of the repository, determine if it is relevant for setting up a development environment for the repository or providing information about how to set up dev env (how to setup, install, test, etc.). This determines whether the file's content is fed to the LLM and helps it set up the environment.

### File:
{file}

### Reply with the following format:

<rel>Yes</rel>

or

<rel>No</rel>

Choose either Yes or No, Yes means this file IS relevant for setting up a dev env for the repository.
"""

THRESHOLD = 128 * 1000 * 3

@auto_catch
def locate_related_file(state: AgentState) -> AgentState:
    """
    Analyze repository structure to identify files relevant for environment setup.
    
    Uses LLM to scan repository structure and determine which files contain
    information about dependencies, installation, and development setup.
    
    Args:
        state (AgentState): Current agent state with repo structure
        
    Returns:
        AgentState: Updated state with documentation content and related files
    """
    llm = state["llm"]
    logger = state["logger"]
    locate_prompt = HumanMessage(
        content=prompt.format(structure=state["repo_structure"])
    )
    if len(locate_prompt.content) > THRESHOLD:
        locate_prompt = HumanMessage(
            content=prompt.format(structure=view_repo_structure(state["repo_root"], 1))
        )
    
    response = llm.invoke([locate_prompt])
    if "<think>" in response.content:
        response.content = response.content.split("</think>")[1]
    potential_files = [
        line.split("<file>")[1].split("</file>")[0].strip()
        for line in response.content.split("\n")
        if line.strip() and "<file>" in line
    ]
    potential_files = [
        file
        for file in potential_files
        if os.path.exists(os.path.join(state["repo_root"], file))
    ]
    potential_files = list(set(potential_files))

    logger.info(f"Potential files: {potential_files}")
    logger.info("Start determine relevance of these files...")
    related_files = []

    docs = "------ BEGIN RELATED FILES ------\n"
    for file in potential_files:
        path = os.path.join(state["repo_root"], file)
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            continue
        with open(path, "r") as f:
            try:
                content = f.read()
            except Exception as e:
                logger.info(f"Error reading file {file}: {e}")
                continue
        
        if len(content) > THRESHOLD:
            content = content[:THRESHOLD]

        file_info = f"""------ START FILE {file} ------
{content}
------ END FILE {file} ------"""
        determine_input = HumanMessage(content=determine_prompt.format(file=file_info))
        try:
            response = llm.invoke([determine_input])
        except Exception:
            logger.error(f"Error determining file: {file}")
            continue
        if "<think>" in response.content:
            response.content = response.content.split("</think>")[1]
        logger.info(f"File: {file} - {response.content}")
        if "<rel>Yes</rel>" in response.content:
            docs += f"File: {file}\n```\n"
            docs += content + "\n"
            docs += "```\n"
            related_files.append(file)
    docs += "------ END RELATED FILES ------\n"

    logger.info(f"Located related files: {related_files}")

    return {
        "messages": [locate_prompt, response],
        "docs": docs,
        # We do not require the full repo structure later
        "repo_structure": view_repo_structure(state["repo_root"], 1),
    }


if __name__ == "__main__":
    from git_launch.agent.state import AgentState

    state = AgentState()
    locate_related_file(state)
