"""
Agent state management for repository setup workflow.
"""
import operator
import time
from functools import wraps
from logging import Logger
from typing import Annotated, Callable, List, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from typing_extensions import Literal, Self, TypedDict

from git_launch.runtime import SetupRuntime
from git_launch.utilities.timemachine import PyPiServer
from git_launch.utilities.llm import LLMProvider


class State(TypedDict):
    exception: Exception | None


LANGUAGE = Literal["python", "rust", "javascript", "bash"]


class AgentState(State):
    """
    Comprehensive state container for the repository setup agent workflow.
    
    Contains all necessary information, tools, and state tracking for processing
    a SWE-bench instance through environment setup and verification stages.
    """
    instance: dict
    llm: LLMProvider
    language: LANGUAGE
    logger: Logger
    messages: Annotated[
        List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]], add_messages
    ]
    search_tool: TavilySearchResults
    setup_messages: Annotated[
        List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]], add_messages
    ]
    verify_messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    setup_commands: Annotated[List[str], operator.add]
    test_commands: List[str]
    commands: Annotated[List[str], operator.add]
    repo_root: str
    repo_structure: str
    result_path: str
    date: str | None
    docs: List[str] | None
    base_image: str | None
    session: SetupRuntime | None
    pypiserver: PyPiServer | None
    current_issue: str | None
    success: bool | None
    start_time: float | None
    trials: int
    debug: bool

    @classmethod
    def create(
        cls,
        instance: str,
        llm: LLMProvider,
        logger: Logger,
        language: LANGUAGE,
        repo_root: str,
        repo_structure: str,
        result_path: str,
        date: str | None = None,
        max_search_results: int = 3,
        debug: bool = False
    ) -> Self:
        """
        Create a new AgentState instance with default values.
        
        Args:
            instance (str): SWE-bench instance data
            llm (LLMProvider): LLM provider for agent interactions
            logger (Logger): Logger for this instance
            language (LANGUAGE): Programming language of the repository
            repo_root (str): Path to the repository root
            repo_structure (str): String representation of repository structure
            result_path (str): Path to store execution results
            date (str, optional): Creation date of the instance
            max_search_results (int): Maximum search results for web search
            debug (bool): Enable debug mode
            
        Returns:
            Self: Initialized AgentState instance
        """
        return cls(
            instance=instance,
            llm=llm,
            language=language,
            logger=logger,
            messages=[],
            search_tool=TavilySearchResults(max_results=max_search_results),
            setup_messages=[],
            verify_messages=[],
            setup_commands=[],
            test_commands=[],
            commands=[],
            repo_root=repo_root,
            repo_structure=repo_structure,
            result_path=result_path,
            date=date,
            docs=None,
            base_image=None,
            session=None,
            start_time=time.time(),
            verified=None,
            pypiserver=None,
            trials=0,
            exception=None,
            debug=debug,
        )


# TODO: more strict type hints
def auto_catch(func: Callable) -> Callable:
    """
    Decorator to automatically catch exceptions in workflow functions.
    
    Args:
        func (Callable): Function to wrap with exception handling
        
    Returns:
        Callable: Wrapped function that returns exception in state on error
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # import traceback

            # print(traceback.format_exc())
            return {"exception": e}

    return wrapper


if __name__ == "__main__":
    # keys of AgentState
    print(AgentState.__annotations__.keys())
