"""
Utility helpers shared across agent modules.
"""
from __future__ import annotations

from typing import Any


def message_content_to_str(content: Any) -> str:
    """
    Normalize LangChain message content into a plain string.

    Newer LangChain versions return message.content as either a raw string
    or a list of structured blocks (e.g., [{"type": "text", "text": "..."}]).
    This helper flattens those structures so legacy string-based parsing
    logic keeps working.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    # Fallback to the full dict representation to preserve info
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "".join(parts)

    return str(content)

