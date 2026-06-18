# -------------------------------------------------------------------
# llm_cache_service.py
# Main service orchestrator that ties together the cache, similarity
# engine, and LLM client into a unified request pipeline.
# -------------------------------------------------------------------

import logging
from typing import Dict, Any

from src.cache.cache_manager import CacheManager
from src.llm.llm_client import LlmClient
from src.llm.openai_client import OpenAiClient
from src.llm.mock_client import MockLlmClient
from src.models.llm_request import LlmRequest
from src.models.llm_response import LlmResponse
from src.exceptions import LlmClientError

# Module-level logger for the cache service
logger = logging.getLogger(__name__)


class LlmCacheService:
    """Main service that orchestrates cached LLM request handling.

    For each incoming request, the service:
    1. Checks if a similar prompt exists in the cache.
    2. If cache hit: returns the cached response (no LLM call).
    3. If cache miss: forwards to the LLM, caches the response, then returns it.

    Attributes:
        cache_manager: Handles cache storage and similarity lookups.
        llm_client: The LLM provider client for non-cached requests.
        total_requests: Counter for total requests processed.
        cache_hits: Counter for requests served from cache.
        cache_misses: Counter for requests forwarded to the LLM.
    """

    def __init__(
        self,
        cache_config: Dict[str, Any],
        similarity_config: Dict[str, Any],
        llm_config: Dict[str, Any],
    ):
        """Initialize the LLM cache service.

        Args:
            cache_config: Configuration for cache storage backend.
            similarity_config: Configuration for similarity thresholds.
            llm_config: Configuration for the LLM provider client.
        """
        # Initialize the cache manager with storage and similarity config
        self.cache_manager = CacheManager(cache_config, similarity_config)
        logger.info("Cache manager initialized")

        # Initialize the appropriate LLM client based on provider setting
        self.llm_client = self._create_llm_client(llm_config)
        logger.info("LLM client initialized")

        # Initialize request counters for metrics tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_llm_client(self, llm_config: Dict[str, Any]) -> LlmClient:
        """Create the appropriate LLM client based on configuration.

        Factory method that instantiates the correct client implementation
        based on the configured provider.

        Args:
            llm_config: Dictionary containing provider type and credentials.

        Returns:
            An LlmClient implementation instance.

        Raises:
            ValueError: If the configured provider is not supported.
        """
        # Determine which provider to use from configuration
        provider = llm_config.get("provider", "mock")

        # Instantiate the correct client implementation
        if provider == "openai":
            return OpenAiClient(llm_config)
        elif provider == "mock":
            return MockLlmClient(llm_config)
        else:
            error_message = f"Unsupported LLM provider: {provider}"
            logger.error(error_message)
            raise ValueError(error_message)

    def process_request(self, request: LlmRequest) -> LlmResponse:
        """Process an LLM request with caching.

        This is the main entry point for all LLM requests. It first checks
        the cache for similar prompts, and only calls the LLM if no match found.

        Args:
            request: The incoming LLM request to process.

        Returns:
            An LlmResponse, either from cache or from the LLM provider.
        """
        # Increment total request counter
        self.total_requests += 1

        logger.info(
            "Processing request #%d: %.50s...",
            self.total_requests,
            request.prompt,
        )

        # Step 1: Check the cache for a similar prompt
        cached_response = self.cache_manager.lookup(request)

        if cached_response is not None:
            # Cache hit - return the cached response without calling the LLM
            self.cache_hits += 1
            logger.info(
                "Serving from cache (hit rate: %.1f%%)",
                self._get_hit_rate(),
            )
            return cached_response

        # Step 2: Cache miss - forward the request to the LLM provider
        self.cache_misses += 1
        logger.info("Cache miss - forwarding to LLM provider")

        try:
            # Send the request to the LLM provider
            llm_response = self.llm_client.send_request(request)

            # Step 3: Store the response in the cache for future use
            self.cache_manager.store(request, llm_response)
            logger.info("Response cached for future similarity matches")

            return llm_response

        except LlmClientError as client_error:
            # Log the error and re-raise for the caller to handle
            logger.error("LLM request failed: %s", client_error)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve service-level statistics.

        Returns:
            Dictionary containing request counts and hit rate metrics.
        """
        # Get cache storage statistics
        cache_stats = self.cache_manager.get_cache_stats()

        # Combine service metrics with cache storage metrics
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": self._get_hit_rate(),
            **cache_stats,
        }

    def _get_hit_rate(self) -> float:
        """Calculate the current cache hit rate percentage.

        Returns:
            Hit rate as a percentage (0.0 to 100.0).
        """
        # Avoid division by zero on first request
        if self.total_requests == 0:
            return 0.0

        return (self.cache_hits / self.total_requests) * 100.0

    def clear_expired_cache(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of expired entries removed.
        """
        return self.cache_manager.clear_expired()
