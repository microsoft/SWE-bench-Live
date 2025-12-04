"""
Base Docker image selection agent for repository environment setup.
"""
try:  # LangChain >= 0.3.26
    from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for older LangChain
    from langchain.schema import HumanMessage  # type: ignore

from launch.agent.state import AgentState, auto_catch
from launch.agent.utils import message_content_to_str
from launch.utilities.language_handlers import get_language_handler


@auto_catch
def select_base_image(state: AgentState) -> dict:
    """
    Select appropriate base Docker image based on repository analysis.
    
    Uses LLM to analyze repository documentation and select the most suitable
    base image from language-specific candidate images.
    
    Args:
        state (AgentState): Current agent state with repo docs and language info
        
    Returns:
        AgentState: Updated state with selected base image
    """
    llm = state["llm"]
    logger = state["logger"]
    language = state["language"]
    
    # Get language handler and candidate images
    language_handler = get_language_handler(language)
    candidate_images = language_handler.base_images
    messages = [
        HumanMessage(
            content=f"""Based on related file:
{state['docs']}

Please recommend a suitable base Docker image. Consider:
1. The programming language and version requirements
2. Common system dependencies
3. Use official images when possible

Select a base image from the following candidate list:
{candidate_images}
Wrap the image name in a block like <image>ubuntu:20.04</image> to indicate your choice.
"""
        )
    ]
    base_image = None
    trials = 0
    while not base_image or trials < 5:
        trials += 1
        response = llm.invoke(messages)
        response_text = message_content_to_str(response.content)
        if "<image>" in response_text:
            image = response_text.split("<image>")[1].split("</image>")[0]
            if image in candidate_images:
                base_image = image
                break
            messages.append(response)
            messages.append(
                HumanMessage(
                    content=f"""The image you selected({image}) is not in the candidate list: {candidate_images}. Please select again."""
                )
            )
        else:
            messages.append(response)
            messages.append(
                HumanMessage(
                    content="""Please wrap the image name in a block like <image>ubuntu:20.04</image> to indicate your choice."""
                )
            )

    logger.info(f"Selected base image: {base_image}")
    return {
        "messages": messages,
        "base_image": base_image,
    }
