"""
LLM provider abstraction for various language model services.
"""
import os
from functools import wraps
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential_jitter



class LLMProvider:
    """
    Unified interface for different LLM providers with logging and retry capabilities.
    
    Supports Azure OpenAI, OpenAI, and Anthropic models with automatic logging
    of interactions and built-in retry logic for robustness.
    """
    def __init__(self, llm_provider: str, model: str):
        """
        Initialize LLM provider with specified backend.
        
        Args:
            llm_provider (str): Provider name ("AOAI", "OpenAI", "Anthropic")
            model: model name, reasoning models suggested
        """
        self.llm_provider = llm_provider

        llm_instance_map = {
            "AOAI": AzureOpenAIModel,
            "OpenAI": OpenAIModel,
            "Anthropic": AnthropicModel,
        }
        if self.llm_provider not in llm_instance_map:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        self.llm_instance = llm_instance_map[self.llm_provider](model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=5, max=120, jitter=3)
    )
    def invoke(self, messages: List[BaseMessage]):
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
    def __init__(self, model_name: str):
        """
        Initialize OpenAI model.
        
        Args:
            model_name (str): Name of the OpenAI model
        """
        self.model_name = model_name

        from langchain_openai import ChatOpenAI
        
        # Use environment variable OPENAI_API_KEY for authentication
        self.llm = ChatOpenAI(
            model=model_name,
        )
    
    def invoke(self, messages: List[BaseMessage]):
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
    def __init__(self, model_name: str):
        """
        Initialize Anthropic model.
        
        Args:
            model_name (str): Name of the Anthropic model
        """
        self.model_name = model_name

        from langchain_anthropic import ChatAnthropic
        
        # Use environment variable ANTHROPIC_API_KEY for authentication
        self.llm = ChatAnthropic(
            model=model_name,
        )
    
    def invoke(self, messages: List[BaseMessage]):
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
    def __init__(self, model_name: str):
        """
        Initialize Azure OpenAI model.
        
        Args:
            model_name (str): Name of the Azure OpenAI model
        """
        self.model_name = model_name

        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(  # Directly initialize the instance
            model=model_name,
        )
    
    def invoke(self, messages: List[BaseMessage]):
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
    }
    llm = LLMProvider(llm_provider, **model_config)
    messages = [HumanMessage(content="What is the capital of France?")]
    res = llm.invoke(messages)
    print(res)


