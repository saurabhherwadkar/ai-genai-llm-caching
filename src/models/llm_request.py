# -------------------------------------------------------------------
# llm_request.py
# Defines the data model for an incoming LLM request.
# -------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlmRequest:
    """Represents a request to be sent to the LLM provider.

    Attributes:
        prompt: The user prompt text to send to the LLM.
        model: The model identifier to use for generation.
        temperature: Sampling temperature for response randomness.
        max_tokens: Maximum number of tokens in the response.
        system_message: Optional system-level instruction for the LLM.
    """

    prompt: str  # The user prompt text
    model: str = "gpt-4"  # Target LLM model identifier
    temperature: float = 0.7  # Controls response randomness
    max_tokens: int = 1024  # Maximum response length in tokens
    system_message: Optional[str] = None  # Optional system instruction
    metadata: dict = field(default_factory=dict)  # Additional request metadata
