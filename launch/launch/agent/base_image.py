"""
Base Docker image selection agent for repository environment setup.
"""
from langchain.schema import HumanMessage

from launch.agent.state import AgentState, auto_catch
from launch.utilities.language_handlers import get_language_handler


def _version_key(image: str) -> tuple[int, ...]:
    """
    Provide a tuple that increases with the semantic version embedded in the tag.
    Falls back to (0,) if no numeric components are found.
    """
    _, _, version_part = image.partition(":")
    if not version_part:
        version_part = image
    tokens = [int(token) for token in re.findall(r"\d+", version_part)]
    return tuple(tokens) if tokens else (0,)


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
    max_trials = 5
    last_response_text = ""
    while trials < max_trials and not base_image:
        trials += 1
        response = llm.invoke(messages)
        if "<image>" in response.content:
            image = response.content.split("<image>")[1].split("</image>")[0]
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
            logger.info(
                "Base image response missing <image> tag, retrying. Raw response: %s",
                response_text[:500],
            )
            messages.append(response)
            messages.append(
                HumanMessage(
                    content="""Please wrap the image name in a block like <image>ubuntu:20.04</image> to indicate your choice."""
                )
            )

    if not base_image:
        fallback = max(candidate_images, key=_version_key) if candidate_images else None
        base_image = fallback
        logger.warning(
            "Base image selection failed after %s trials, defaulting to %s. Last response: %s",
            max_trials,
            base_image,
            last_response_text[:500],
        )
    else:
        logger.info(f"Selected base image: {base_image}")
    return {
        "messages": messages,
        "base_image": base_image,
    }
