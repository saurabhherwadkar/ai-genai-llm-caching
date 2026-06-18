# -------------------------------------------------------------------
# test_cache_manager.py
# Unit tests for the CacheManager class.
# -------------------------------------------------------------------

import pytest
import os
import tempfile

from src.cache.cache_manager import CacheManager
from src.models.llm_request import LlmRequest
from src.models.llm_response import LlmResponse


class TestCacheManager:
    """Tests for the CacheManager class."""

    def setup_method(self):
        """Set up test fixtures with in-memory cache backend."""
        # Use memory backend to avoid file system side effects in tests
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
        # Initialize the cache manager under test
        self.cache_manager = CacheManager(self.cache_config, self.similarity_config)

    def test_store_creates_cache_entry(self):
        """Storing a request-response pair should create a cache entry."""
        request = LlmRequest(prompt="What is Python?")
        response = LlmResponse(content="Python is a programming language.", model="gpt-4")

        # Store the entry
        entry = self.cache_manager.store(request, response)

        # Verify the entry was created with correct data
        assert entry.request_prompt == "What is Python?"
        assert entry.response_content == "Python is a programming language."
        assert entry.cache_key is not None
        assert len(entry.embedding) > 0

    def test_lookup_returns_none_for_empty_cache(self):
        """Lookup on an empty cache should return None."""
        request = LlmRequest(prompt="What is Python?")
        result = self.cache_manager.lookup(request)
        assert result is None

    def test_lookup_finds_exact_match(self):
        """Lookup should find an exact match in the cache."""
        # Store a request-response pair
        request = LlmRequest(prompt="What is Python?")
        response = LlmResponse(content="Python is a language.", model="gpt-4")
        self.cache_manager.store(request, response)

        # Look up the same prompt
        lookup_request = LlmRequest(prompt="What is Python?")
        cached_response = self.cache_manager.lookup(lookup_request)

        # Verify cache hit
        assert cached_response is not None
        assert cached_response.cached is True
        assert cached_response.content == "Python is a language."

    def test_lookup_finds_syntactically_similar_prompt(self):
        """Lookup should find syntactically similar prompts."""
        # Store a request
        request = LlmRequest(prompt="What is machine learning and how does it work?")
        response = LlmResponse(content="ML is a subset of AI.", model="gpt-4")
        self.cache_manager.store(request, response)

        # Look up a similar prompt (missing question mark)
        similar_request = LlmRequest(prompt="What is machine learning and how does it work")
        cached_response = self.cache_manager.lookup(similar_request)

        # Should find the cached entry
        assert cached_response is not None
        assert cached_response.cached is True

    def test_lookup_finds_semantically_similar_prompt(self):
        """Lookup should find semantically similar prompts."""
        # Store a request
        request = LlmRequest(prompt="What is machine learning and how does it work?")
        response = LlmResponse(content="ML is about learning from data.", model="gpt-4")
        self.cache_manager.store(request, response)

        # Look up a semantically similar prompt (different wording)
        semantic_request = LlmRequest(prompt="Explain how machine learning works and what it is")
        cached_response = self.cache_manager.lookup(semantic_request)

        # Should find the cached entry via semantic similarity
        assert cached_response is not None
        assert cached_response.cached is True
        assert cached_response.similarity_score > 0.0

    def test_lookup_misses_for_unrelated_prompt(self):
        """Lookup should return None for completely unrelated prompts."""
        # Store a request about ML
        request = LlmRequest(prompt="What is machine learning?")
        response = LlmResponse(content="ML is...", model="gpt-4")
        self.cache_manager.store(request, response)

        # Look up an unrelated prompt
        unrelated_request = LlmRequest(prompt="How to bake a chocolate cake?")
        cached_response = self.cache_manager.lookup(unrelated_request)

        # Should not find a match
        assert cached_response is None

    def test_cache_stats_empty(self):
        """Stats on empty cache should show zeros."""
        stats = self.cache_manager.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0

    def test_cache_stats_after_store(self):
        """Stats should reflect stored entries."""
        request = LlmRequest(prompt="Test prompt")
        response = LlmResponse(content="Test response", model="gpt-4")
        self.cache_manager.store(request, response)

        stats = self.cache_manager.get_cache_stats()
        assert stats["total_entries"] == 1

    def test_generate_cache_key_consistency(self):
        """Same prompt should always generate the same cache key."""
        key_a = self.cache_manager._generate_cache_key("Hello world")
        key_b = self.cache_manager._generate_cache_key("Hello world")
        assert key_a == key_b

    def test_generate_cache_key_case_insensitive(self):
        """Cache key generation should be case-insensitive."""
        key_a = self.cache_manager._generate_cache_key("Hello World")
        key_b = self.cache_manager._generate_cache_key("hello world")
        assert key_a == key_b
