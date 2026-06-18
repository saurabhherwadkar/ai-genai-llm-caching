# -------------------------------------------------------------------
# llm_response.py
# Defines the data model for an LLM response.
# -------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlmResponse:
    """Represents a response received from the LLM provider.

    Attributes:
        content: The generated text content from the LLM.
        model: The model that generated this response.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Total tokens consumed (prompt + completion).
        cached: Whether this response was served from cache.
        similarity_score: The similarity score if served from cache.
    """

    content: str  # The generated text from the LLM
    model: str  # The model that produced this response
    prompt_tokens: int = 0  # Input token count
    completion_tokens: int = 0  # Output token count
    total_tokens: int = 0  # Total token count
    cached: bool = False  # True if response was served from cache
    similarity_score: float = 0.0  # Similarity score when cache hit
    cache_key: Optional[str] = None  # Cache key if applicable
    metadata: dict = field(default_factory=dict)  # Additional response metadata
