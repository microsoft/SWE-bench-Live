"""
LLM provider abstraction for various language model services.
"""
import os
from functools import wraps
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


def logged_invoke(invoke_func):
    """
    Decorator to log LLM interactions to files.
    
    Args:
        invoke_func: LLM invoke function to wrap
        
    Returns:
        Wrapped function that logs inputs and outputs
    """
    @wraps(invoke_func)
    def wrapper(self, messages: List[BaseMessage]) -> BaseMessage:  
        if self.log_folder is None:
            response: BaseMessage = invoke_func(self, messages)
            return response
        
        log_folder = self.log_folder  # Dynamically get the log folder from the instance
        os.makedirs(log_folder, exist_ok=True)

        try:
            existing_files = [
                f for f in os.listdir(log_folder) if f.split(".")[0].isdigit()
            ]
            existing_numbers = [int(name.split(".")[0]) for name in existing_files]
            next_number = max(existing_numbers) + 1 if existing_numbers else 0
        except (OSError, ValueError):
            next_number = 0
        log_file_path = os.path.join(log_folder, f"{next_number}.md")

        response: BaseMessage = invoke_func(self, messages)

        with open(log_file_path, "w") as f:
            f.write("##### LLM INPUT #####\n")
            f.write("\n".join([m.pretty_repr() for m in messages]))
            f.write("\n##### LLM OUTPUT #####\n")
            f.write(response.pretty_repr())
        return response
    return wrapper


class LLMProvider:
    """
    Unified interface for different LLM providers with logging and retry capabilities.
    
    Supports Azure OpenAI, OpenAI, and Anthropic models with automatic logging
    of interactions and built-in retry logic for robustness.
    """
    def __init__(self, llm_provider: str, log_folder: str | None = "./llm_logs", **kwargs):
        """
        Initialize LLM provider with specified backend.
        
        Args:
            llm_provider (str): Provider name ("AOAI", "OpenAI", "Anthropic")
            log_folder (str | None): Directory for interaction logs, None disables logging
            **kwargs: Model configuration parameters (model_name, temperature, etc.)
        """
        self.llm_provider = llm_provider
        self.log_folder = log_folder

        llm_instance_map = {
            "AOAI": AzureOpenAIModel,
            "OpenAI": OpenAIModel,
            "Anthropic": AnthropicModel,
        }
        if self.llm_provider not in llm_instance_map:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        self.llm_instance = llm_instance_map[self.llm_provider](**kwargs)

    @logged_invoke
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=5, max=10, jitter=3)
    )
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with messages, includes automatic retry and logging.
        
        Args:
            messages (List[BaseMessage]): List of conversation messages
            
        Returns:
            BaseMessage: LLM response message
        """
        return self.llm_instance.invoke(messages)


class OpenAIModel:
    """OpenAI model implementation with API key authentication."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        """
        Initialize OpenAI model.
        
        Args:
            model_name (str): Name of the OpenAI model
            temperature (float): Sampling temperature for responses
        """
        self.model_name = model_name
        self.temperature = temperature

        from langchain_openai import ChatOpenAI
        
        # Use environment variable OPENAI_API_KEY for authentication
        self.llm = ChatOpenAI(
            # base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
        )
    
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke OpenAI model with messages.
        
        Args:
            messages (List[BaseMessage]): Conversation messages
            
        Returns:
            BaseMessage: Model response
        """
        return self.llm.invoke(messages)


class AnthropicModel:   
    """Anthropic model implementation with API key authentication."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        """
        Initialize Anthropic model.
        
        Args:
            model_name (str): Name of the Anthropic model
            temperature (float): Sampling temperature for responses
        """
        self.model_name = model_name
        self.temperature = temperature

        from langchain_anthropic import ChatAnthropic
        
        # Use environment variable ANTHROPIC_API_KEY for authentication
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
        )
    
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke Anthropic model with messages.
        
        Args:
            messages (List[BaseMessage]): Conversation messages
            
        Returns:
            BaseMessage: Model response
        """
        return self.llm.invoke(messages)


class AzureOpenAIModel:
    """Azure OpenAI model implementation with token-based authentication."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        """
        Initialize Azure OpenAI model.
        
        Args:
            model_name (str): Name of the Azure OpenAI model
            temperature (float): Sampling temperature for responses
        """
        self.model_name = model_name
        self.temperature = temperature

        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(  # Directly initialize the instance
            model=model_name,
            temperature=temperature,
        )
    
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke Azure OpenAI model with messages.
        
        Args:
            messages (List[BaseMessage]): Conversation messages
            
        Returns:
            BaseMessage: Model response
        """
        return self.llm.invoke(messages)

if __name__ == "__main__":
    llm_provider = "AOAI"
    model_config = {        
        "model_name": "gpt-4o-20241120",
        "temperature": 0.0,
    }
    llm = LLMProvider(llm_provider, log_folder="./llm_logs", **model_config)
    messages = [HumanMessage(content="What is the capital of France?")]
    res = llm.invoke(messages)
    print(res)
