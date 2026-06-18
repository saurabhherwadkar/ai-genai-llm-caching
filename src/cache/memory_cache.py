# -------------------------------------------------------------------
# memory_cache.py
# In-memory cache storage backend using a dictionary with LRU eviction.
# -------------------------------------------------------------------

import logging
from collections import OrderedDict
from typing import List, Optional

from src.models.cache_entry import CacheEntry

# Module-level logger for memory cache operations
logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache storage backend with LRU eviction.

    Stores cache entries in an ordered dictionary that provides O(1)
    lookups and maintains insertion order for LRU eviction when the
    maximum capacity is reached.

    Attributes:
        max_entries: Maximum number of entries before eviction.
        entries: Ordered dictionary storing cache entries by key.
    """

    def __init__(self, max_entries: int = 1000):
        """Initialize the in-memory cache.

        Args:
            max_entries: Maximum number of entries to store before evicting.
        """
        self.max_entries = max_entries  # Set the capacity limit
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()  # LRU storage

        logger.info("Memory cache initialized with max entries: %d", max_entries)

    def store_entry(self, entry: CacheEntry) -> None:
        """Store a cache entry in memory.

        If the cache is at capacity, the least recently used entry
        is evicted before storing the new one.

        Args:
            entry: The CacheEntry to store.
        """
        # Evict the oldest entry if at capacity
        if len(self.entries) >= self.max_entries:
            # Pop the first (oldest) item from the ordered dictionary
            evicted_key, evicted_entry = self.entries.popitem(last=False)
            logger.debug("Evicted LRU entry with key: %.16s...", evicted_key)

        # Store the new entry using its cache key
        self.entries[entry.cache_key] = entry

        # Move to end to mark as most recently used
        self.entries.move_to_end(entry.cache_key)

        logger.debug(
            "Stored entry in memory cache. Total entries: %d", len(self.entries)
        )

    def get_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve a specific cache entry by its key.

        Moves the accessed entry to the end to update LRU ordering.

        Args:
            cache_key: The unique key identifying the cache entry.

        Returns:
            The CacheEntry if found, None otherwise.
        """
        # Look up the entry in the dictionary
        entry = self.entries.get(cache_key)

        if entry:
            # Move to end to mark as recently accessed (LRU update)
            self.entries.move_to_end(cache_key)

        return entry

    def get_all_entries(self) -> List[CacheEntry]:
        """Retrieve all cache entries currently in memory.

        Returns:
            List of all CacheEntry objects in the cache.
        """
        return list(self.entries.values())

    def update_entry(self, entry: CacheEntry) -> None:
        """Update an existing cache entry in memory.

        Args:
            entry: The updated CacheEntry to persist.
        """
        # Overwrite the existing entry with the updated version
        if entry.cache_key in self.entries:
            self.entries[entry.cache_key] = entry
            logger.debug("Updated entry with key: %.16s...", entry.cache_key)

    def remove_expired(self) -> int:
        """Remove all expired entries from the memory cache.

        Returns:
            Number of entries that were removed.
        """
        # Identify all expired entry keys
        expired_keys = [
            key for key, entry in self.entries.items() if entry.is_expired()
        ]

        # Remove each expired entry from the dictionary
        for key in expired_keys:
            del self.entries[key]

        if expired_keys:
            logger.info("Removed %d expired entries from memory cache", len(expired_keys))

        return len(expired_keys)

    def clear(self) -> None:
        """Remove all entries from the memory cache."""
        self.entries.clear()
        logger.info("Memory cache cleared")
