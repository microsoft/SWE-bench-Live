"""Normalize command text that was serialized through HTML before evaluation."""

from __future__ import annotations

from html import unescape


def normalize_published_shell_command(command: str) -> str:
    """Restore standard HTML character references in a published shell command.

    Dataset rows are sometimes captured from HTML-rendered content. Keeping
    ``&gt;`` or ``&amp;`` verbatim changes shell syntax; on Windows PowerShell treats
    the leading ampersand as an operator and rejects the command before tests
    start. ``html.unescape`` is intentionally limited to standard character
    references and leaves ordinary command text untouched.
    """
    return unescape(command)