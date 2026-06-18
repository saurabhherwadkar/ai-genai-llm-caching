# -------------------------------------------------------------------
# cache_manager.py
# High-level cache manager that coordinates storage and similarity
# matching to serve cached responses or store new ones.
# -------------------------------------------------------------------

import hashlib
import logging
from typing import Dict, Any, Optional

from src.models.cache_entry import CacheEntry
from src.models.llm_request import LlmRequest
from src.models.llm_response import LlmResponse
from src.similarity.similarity_engine import SimilarityEngine
from src.cache.memory_cache import MemoryCache
from src.cache.sqlite_cache import SqliteCache

# Module-level logger for cache management operations
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages the LLM response cache with similarity-based lookup.

    Coordinates between the storage backend and the similarity engine
    to determine whether a new request can be served from cache or
    needs to be forwarded to the LLM provider.

    Attributes:
        storage: The cache storage backend (memory or sqlite).
        similarity_engine: Engine for computing similarity scores.
        ttl_seconds: Default time-to-live for new cache entries.
    """

    def __init__(
        self,
        cache_config: Dict[str, Any],
        similarity_config: Dict[str, Any],
    ):
        """Initialize the cache manager with storage and similarity.

        Args:
            cache_config: Configuration for cache storage backend.
            similarity_config: Configuration for similarity thresholds and model.
        """
        # Extract cache configuration values
        backend_type = cache_config.get("backend", "memory")
        self.ttl_seconds = cache_config.get("ttl_seconds", 86400)

        # Initialize the appropriate storage backend
        if backend_type == "sqlite":
            database_path = cache_config.get("database_path", "data/llm_cache.db")
            self.storage = SqliteCache(database_path=database_path)
            logger.info("Using SQLite cache backend at: %s", database_path)
        else:
            max_entries = cache_config.get("max_memory_entries", 1000)
            self.storage = MemoryCache(max_entries=max_entries)
            logger.info("Using in-memory cache backend (max: %d)", max_entries)

        # Initialize the similarity engine for match detection
        self.similarity_engine = SimilarityEngine(similarity_config)
        logger.info("Cache manager initialized successfully")

    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key from the prompt text.

        Uses SHA-256 hashing of the normalized prompt for consistent keys.

        Args:
            prompt: The prompt text to generate a key for.

        Returns:
            A hexadecimal string hash serving as the cache key.
        """
        # Normalize the prompt before hashing for consistency
        normalized_prompt = prompt.lower().strip()

        # Generate SHA-256 hash of the normalized prompt
        hash_digest = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()

        return hash_digest

    def lookup(self, request: LlmRequest) -> Optional[LlmResponse]:
        """Look up a cached response for the given request.

        Checks if any previously cached prompt is sufficiently similar
        (syntactically or semantically) to return a cached response.

        Args:
            request: The incoming LLM request to look up.

        Returns:
            An LlmResponse from cache if a match is found, None otherwise.
        """
        logger.info("Cache lookup for prompt: %.50s...", request.prompt)

        # Retrieve all non-expired cache entries from storage
        all_entries = self.storage.get_all_entries()

        # Use the similarity engine to find the best matching entry
        match_result = self.similarity_engine.find_best_match(
            request.prompt, all_entries
        )

        # No match found, return None to signal a cache miss
        if match_result is None:
            logger.info("Cache MISS - no similar prompt found")
            return None

        # Unpack the match result
        matched_entry, similarity_score, match_type = match_result

        # Increment the hit counter on the matched entry
        matched_entry.increment_hit_count()
        self.storage.update_entry(matched_entry)

        # Build and return the cached response
        cached_response = LlmResponse(
            content=matched_entry.response_content,
            model=matched_entry.model,
            cached=True,
            similarity_score=similarity_score,
            cache_key=matched_entry.cache_key,
            metadata={
                "match_type": match_type,
                "hit_count": matched_entry.hit_count,
                "original_prompt": matched_entry.request_prompt,
            },
        )

        logger.info(
            "Cache HIT (%s) with score %.4f", match_type, similarity_score
        )

        return cached_response

    def store(self, request: LlmRequest, response: LlmResponse) -> CacheEntry:
        """Store a new request-response pair in the cache.

        Generates an embedding for the prompt and persists the entry
        in the configured storage backend.

        Args:
            request: The original LLM request.
            response: The LLM response to cache.

        Returns:
            The created CacheEntry instance.
        """
        # Generate a unique cache key for this prompt
        cache_key = self._generate_cache_key(request.prompt)

        # Generate a semantic embedding for future similarity lookups
        embedding = self.similarity_engine.generate_embedding(request.prompt)

        # Create the cache entry with all relevant data
        entry = CacheEntry(
            request_prompt=request.prompt,
            response_content=response.content,
            model=response.model,
            embedding=embedding,
            ttl_seconds=self.ttl_seconds,
            cache_key=cache_key,
        )

        # Persist the entry in the storage backend
        self.storage.store_entry(entry)

        logger.info("Stored new cache entry with key: %.16s...", cache_key)
        return entry

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retrieve statistics about the current cache state.

        Returns:
            Dictionary containing cache size, hit counts, and other metrics.
        """
        all_entries = self.storage.get_all_entries()

        # Compute aggregate statistics across all entries
        total_hits = sum(entry.hit_count for entry in all_entries)
        expired_count = sum(1 for entry in all_entries if entry.is_expired())

        return {
            "total_entries": len(all_entries),
            "expired_entries": expired_count,
            "active_entries": len(all_entries) - expired_count,
            "total_hits": total_hits,
        }

    def clear_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of expired entries that were removed.
        """
        # Delegate to storage backend for removal
        removed_count = self.storage.remove_expired()
        logger.info("Cleared %d expired cache entries", removed_count)
        return removed_count
