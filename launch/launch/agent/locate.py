"""
Repository analysis agent for locating environment setup documentation.
"""
import os

try:  # LangChain >= 0.3.26
    from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from langchain.schema import HumanMessage  # type: ignore

from launch.agent.state import AgentState, auto_catch
from launch.utilities.get_repo_structure import view_repo_structure
from launch.agent.utils import message_content_to_str

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

THRESHOLD = 128 * 1000 * 2

@auto_catch
def locate_related_file(state: AgentState) -> dict:
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
    response_text = message_content_to_str(response.content)
    potential_files = [
        line.split("<file>")[1].split("</file>")[0].strip()
        for line in response_text.split("\n")
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
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(THRESHOLD)
        except Exception as e:
            logger.info(f"Error reading file {file}: {e}")
            continue

        file_info = f"""------ START FILE {file} ------
{content}
------ END FILE {file} ------"""
        determine_input = HumanMessage(content=determine_prompt.format(file=file_info))
        try:
            response = llm.invoke([determine_input])
            response_text = message_content_to_str(response.content)
        except Exception:
            logger.error(f"Error determining file: {file}")
            continue
        logger.info(f"File: {file} - {response_text}")
        if "<rel>Yes</rel>" in response_text:
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
    from launch.agent.state import AgentState

    state = AgentState()
    locate_related_file(state)
