"""
Agent state management for repository setup workflow.
"""
import json
import operator
import os
import time
import traceback
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

from launch.core.runtime import SetupRuntime
from launch.utilities.timemachine import PyPiServer
from launch.utilities.llm import LLMProvider


class State(TypedDict):
    exception: Exception | None


LANGUAGE = Literal["python", "rust", "javascript", "bash", "java", "c", "c++", "go", "c#"]


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
    print_commands: List[str]
    commands: Annotated[List[str], operator.add]
    repo_root: str
    repo_structure: str
    result_path: str
    date: str | None
    docs: str | None
    base_image: str | None
    session: SetupRuntime | None
    pypiserver: PyPiServer | None
    current_issue: str | None
    success: bool | None
    start_time: float | None
    trials: int
    debug: bool
    platform: str
    image_prefix: str
    parser: str | None
    test_status: dict[str, str] | None
    pertest_command: dict[str, str] | None
    unittest_generator: str | None
    original_parser: str | None
    original_test_status: dict[str, str] | None
    result: str

    @classmethod
    def create(
        cls,
        instance: dict,
        llm: LLMProvider,
        logger: Logger,
        language: LANGUAGE,
        repo_root: str,
        repo_structure: str,
        image_prefix: str,
        result_path: str,
        date: str | None = None,
        max_search_results: int = 3,
        debug: bool = False,
        platform: str = "linux",
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

        docs = None
        if os.path.exists(result_path):
            with open(result_path) as f:
                history = f.read()
            if history.strip():
                history = json.loads(history)
                docs = history.get("docs", None)

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
            print_commands=[],
            commands=[],
            repo_root=repo_root,
            repo_structure=repo_structure,
            image_prefix=image_prefix,
            result_path=result_path,
            date=date,
            docs=docs,
            base_image=None,
            session=None,
            start_time=time.time(),
            pypiserver=None,
            current_issue=None,
            success=None,
            trials=0,
            exception=None,
            debug=debug,
            platform=platform,
            parser=None,
            test_status=None,
            pertest_command=None,
            unittest_generator=None,
            original_parser=None,
            original_test_status=None,
            result="",
        )


def auto_catch(func: Callable[..., dict]) -> Callable[..., dict]:
    """
    Decorator to automatically catch exceptions in workflow functions.
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function that returns exception in state on error
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"exception": Exception(str(e) + "\n\n" + str(traceback.format_exc()))}

    return wrapper


if __name__ == "__main__":
    # keys of AgentState
    print(AgentState.__annotations__.keys())
