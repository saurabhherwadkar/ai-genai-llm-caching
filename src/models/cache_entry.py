# -------------------------------------------------------------------
# cache_entry.py
# Defines the data model for a cache entry storing request-response pairs.
# -------------------------------------------------------------------

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class CacheEntry:
    """Represents a cached request-response pair with embedding data.

    Attributes:
        request_prompt: The original prompt that was sent to the LLM.
        response_content: The LLM-generated response text.
        model: The model used to generate the response.
        embedding: Vector embedding of the prompt for semantic similarity.
        created_at: Timestamp when this entry was cached.
        hit_count: Number of times this cache entry has been retrieved.
        ttl_seconds: Time-to-live for this entry in seconds.
    """

    request_prompt: str  # The original user prompt
    response_content: str  # The cached LLM response text
    model: str  # Model that generated the response
    embedding: List[float] = field(default_factory=list)  # Semantic embedding vector
    created_at: datetime = field(default_factory=datetime.utcnow)  # Creation timestamp
    hit_count: int = 0  # Number of cache hits for this entry
    ttl_seconds: int = 86400  # Time-to-live in seconds (default 24h)
    cache_key: Optional[str] = None  # Unique identifier for this entry

    def is_expired(self) -> bool:
        """Check if this cache entry has exceeded its time-to-live.

        Returns:
            True if the entry is expired, False otherwise.
        """
        # Calculate elapsed time since entry creation
        elapsed_seconds = (datetime.utcnow() - self.created_at).total_seconds()

        # Return True if elapsed time exceeds the TTL
        return elapsed_seconds > self.ttl_seconds

    def increment_hit_count(self) -> None:
        """Increment the hit counter for this cache entry."""
        self.hit_count += 1
