# -------------------------------------------------------------------
# test_llm_cache_service.py
# Unit tests for the LlmCacheService orchestrator.
# -------------------------------------------------------------------

import pytest

from src.llm_cache_service import LlmCacheService
from src.models.llm_request import LlmRequest


class TestLlmCacheService:
    """Tests for the LlmCacheService class."""

    def setup_method(self):
        """Set up test fixtures with mock LLM provider."""
        # Configure with mock provider to avoid real API calls
        self.cache_config = {
            "backend": "memory",
            "max_memory_entries": 100,
            "ttl_seconds": 3600,
        }
        self.similarity_config = {
            "syntactic_threshold": 0.85,
            "semantic_threshold": 0.80,
            "embedding_model": "all-MiniLM-L6-v2",
        }
        self.llm_config = {
            "provider": "mock",
            "model": "mock-gpt-4",
        }

        # Initialize the service under test
        self.service = LlmCacheService(
            self.cache_config, self.similarity_config, self.llm_config
        )

    def test_first_request_is_cache_miss(self):
        """First request for a new prompt should be a cache miss."""
        request = LlmRequest(prompt="What is Python programming?")
        response = self.service.process_request(request)

        # First request should not be cached
        assert response.cached is False
        assert response.content is not None
        assert len(response.content) > 0

    def test_duplicate_request_is_cache_hit(self):
        """Sending the same prompt twice should result in a cache hit."""
        request = LlmRequest(prompt="What is Python programming?")

        # First request - cache miss
        first_response = self.service.process_request(request)
        assert first_response.cached is False

        # Second request - same prompt should hit cache
        second_response = self.service.process_request(request)
        assert second_response.cached is True
        assert second_response.similarity_score > 0.85

    def test_similar_request_is_cache_hit(self):
        """A syntactically similar prompt should hit the cache."""
        # First request
        request_a = LlmRequest(prompt="What is machine learning and how does it work?")
        self.service.process_request(request_a)

        # Similar request (missing question mark)
        request_b = LlmRequest(prompt="What is machine learning and how does it work")
        response_b = self.service.process_request(request_b)

        # Should be served from cache
        assert response_b.cached is True

    def test_unrelated_request_is_cache_miss(self):
        """An unrelated prompt should be a cache miss."""
        # First request about ML
        request_a = LlmRequest(prompt="What is machine learning?")
        self.service.process_request(request_a)

        # Unrelated request about cooking
        request_b = LlmRequest(prompt="How to make pasta from scratch?")
        response_b = self.service.process_request(request_b)

        # Should NOT be served from cache
        assert response_b.cached is False

    def test_stats_track_hits_and_misses(self):
        """Service stats should accurately track hits and misses."""
        # Send first request (miss)
        self.service.process_request(LlmRequest(prompt="What is AI?"))
        # Send duplicate request (hit)
        self.service.process_request(LlmRequest(prompt="What is AI?"))
        # Send different request (miss)
        self.service.process_request(LlmRequest(prompt="What is biology?"))

        stats = self.service.get_stats()
        assert stats["total_requests"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2

    def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly."""
        # Two misses (distinct prompts), then two exact duplicate hits
        self.service.process_request(LlmRequest(prompt="What is quantum physics and relativity?"))
        self.service.process_request(LlmRequest(prompt="How do you bake chocolate chip cookies at home?"))
        self.service.process_request(LlmRequest(prompt="What is quantum physics and relativity?"))
        self.service.process_request(LlmRequest(prompt="How do you bake chocolate chip cookies at home?"))

        stats = self.service.get_stats()
        # 2 hits out of 4 requests = 50.0%
        assert abs(stats["hit_rate_percent"] - 50.0) < 0.5

    def test_invalid_provider_raises_error(self):
        """An unsupported provider should raise ValueError."""
        bad_config = {"provider": "nonexistent"}
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LlmCacheService(self.cache_config, self.similarity_config, bad_config)

    def test_clear_expired_cache(self):
        """clear_expired_cache should return count of removed entries."""
        # With no expired entries, should return 0
        removed = self.service.clear_expired_cache()
        assert removed == 0

    def test_semantic_cache_hit(self):
        """Semantically similar prompts should hit the cache."""
        # Store a request about machine learning
        request_a = LlmRequest(prompt="What is machine learning and how does it work?")
        self.service.process_request(request_a)

        # Ask the same thing with different wording but same meaning
        request_b = LlmRequest(prompt="Explain how machine learning works and what it is")
        response_b = self.service.process_request(request_b)

        # Should find it via semantic similarity
        assert response_b.cached is True
        assert response_b.similarity_score > 0.0
