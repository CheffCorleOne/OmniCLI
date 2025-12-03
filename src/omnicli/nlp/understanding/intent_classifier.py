"""Very simple intent classifier stub.

In future this will be backed by a transformer model fine-tuned on NL2Bash-style data.
"""

from __future__ import annotations


def classify_intent(user_input: str) -> str:
    """Return a coarse intent label for a given user input.

    This is intentionally naive for the first iteration.
    """
    text = user_input.lower()
    if "бэкап" in text or "backup" in text:
        return "backup"
    if "разверни" in text or "deploy" in text:
        return "deploy"
    if "мониторинг" in text or "monitor" in text:
        return "setup_monitoring"
    if "почему" in text or "why" in text:
        return "diagnose"
    return "general"



