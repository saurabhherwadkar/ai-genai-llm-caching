# -------------------------------------------------------------------
# test_memory_cache.py
# Unit tests for the MemoryCache storage backend.
# -------------------------------------------------------------------

import pytest
from datetime import datetime, timedelta

from src.cache.memory_cache import MemoryCache
from src.models.cache_entry import CacheEntry


class TestMemoryCache:
    """Tests for the MemoryCache class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a memory cache with a small capacity for testing
        self.cache = MemoryCache(max_entries=5)

    def _create_entry(self, key: str, prompt: str = "test", ttl: int = 3600) -> CacheEntry:
        """Helper to create a CacheEntry for testing.

        Args:
            key: The cache key for the entry.
            prompt: The request prompt text.
            ttl: Time-to-live in seconds.

        Returns:
            A CacheEntry instance configured for testing.
        """
        return CacheEntry(
            cache_key=key,
            request_prompt=prompt,
            response_content=f"Response for {prompt}",
            model="gpt-4",
            embedding=[0.1, 0.2, 0.3],
            ttl_seconds=ttl,
        )

    def test_store_and_retrieve_entry(self):
        """Stored entries should be retrievable by key."""
        entry = self._create_entry("key1", "test prompt")
        self.cache.store_entry(entry)

        retrieved = self.cache.get_entry("key1")
        assert retrieved is not None
        assert retrieved.request_prompt == "test prompt"

    def test_get_entry_returns_none_for_missing_key(self):
        """get_entry should return None for non-existent keys."""
        result = self.cache.get_entry("nonexistent")
        assert result is None

    def test_get_all_entries_returns_all(self):
        """get_all_entries should return all stored entries."""
        self.cache.store_entry(self._create_entry("key1", "prompt1"))
        self.cache.store_entry(self._create_entry("key2", "prompt2"))

        entries = self.cache.get_all_entries()
        assert len(entries) == 2

    def test_lru_eviction_on_capacity(self):
        """Oldest entry should be evicted when capacity is exceeded."""
        # Fill the cache to capacity (max_entries=5)
        for i in range(5):
            self.cache.store_entry(self._create_entry(f"key{i}", f"prompt{i}"))

        # Add one more entry to trigger eviction
        self.cache.store_entry(self._create_entry("key5", "prompt5"))

        # The first entry (key0) should have been evicted
        assert self.cache.get_entry("key0") is None
        # The newest entry should still be present
        assert self.cache.get_entry("key5") is not None
        # Total entries should remain at capacity
        assert len(self.cache.get_all_entries()) == 5

    def test_update_entry_modifies_existing(self):
        """update_entry should modify the stored entry."""
        entry = self._create_entry("key1", "test prompt")
        self.cache.store_entry(entry)

        # Modify and update the entry
        entry.hit_count = 5
        self.cache.update_entry(entry)

        # Verify the update persisted
        retrieved = self.cache.get_entry("key1")
        assert retrieved.hit_count == 5

    def test_remove_expired_removes_old_entries(self):
        """remove_expired should remove entries past their TTL."""
        # Create an entry with TTL of 0 (immediately expired)
        expired_entry = CacheEntry(
            cache_key="expired",
            request_prompt="old prompt",
            response_content="old response",
            model="gpt-4",
            embedding=[0.1],
            created_at=datetime.utcnow() - timedelta(hours=2),
            ttl_seconds=1,
        )
        self.cache.store_entry(expired_entry)
        self.cache.store_entry(self._create_entry("active", "new prompt"))

        # Remove expired entries
        removed = self.cache.remove_expired()

        # Only the expired entry should be removed
        assert removed == 1
        assert self.cache.get_entry("expired") is None
        assert self.cache.get_entry("active") is not None

    def test_clear_removes_all_entries(self):
        """clear should empty the entire cache."""
        self.cache.store_entry(self._create_entry("key1"))
        self.cache.store_entry(self._create_entry("key2"))
        self.cache.clear()

        entries = self.cache.get_all_entries()
        assert len(entries) == 0

    def test_access_updates_lru_order(self):
        """Accessing an entry should move it to most-recently-used position."""
        # Fill cache to capacity
        for i in range(5):
            self.cache.store_entry(self._create_entry(f"key{i}"))

        # Access the first entry to move it to the end
        self.cache.get_entry("key0")

        # Add a new entry to trigger eviction
        self.cache.store_entry(self._create_entry("key5"))

        # key0 was accessed so it shouldn't be evicted; key1 should be
        assert self.cache.get_entry("key0") is not None
        assert self.cache.get_entry("key1") is None
