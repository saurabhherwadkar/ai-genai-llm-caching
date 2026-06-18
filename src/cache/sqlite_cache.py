# -------------------------------------------------------------------
# sqlite_cache.py
# SQLite-based persistent cache storage backend.
# Stores cache entries on disk for durability across restarts.
# -------------------------------------------------------------------

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from src.models.cache_entry import CacheEntry

# Module-level logger for SQLite cache operations
logger = logging.getLogger(__name__)


class SqliteCache:
    """SQLite-based persistent cache storage backend.

    Stores cache entries in a SQLite database file, providing
    durability across application restarts and efficient lookups.

    Attributes:
        database_path: File path to the SQLite database.
        connection: Active database connection instance.
    """

    def __init__(self, database_path: str = "data/llm_cache.db"):
        """Initialize the SQLite cache backend.

        Creates the database directory and table if they don't exist.

        Args:
            database_path: File path for the SQLite database file.
        """
        self.database_path = database_path  # Store the database file path

        # Create the directory for the database file if needed
        database_directory = os.path.dirname(database_path)
        if database_directory:
            os.makedirs(database_directory, exist_ok=True)

        # Establish a connection to the SQLite database
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name

        # Create the cache table if it doesn't already exist
        self._create_table()

        logger.info("SQLite cache initialized at: %s", database_path)

    def _create_table(self) -> None:
        """Create the cache entries table if it does not exist."""
        # Define the SQL schema for the cache table
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                request_prompt TEXT NOT NULL,
                response_content TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                ttl_seconds INTEGER DEFAULT 86400
            )
        """

        # Execute the table creation statement
        cursor = self.connection.cursor()
        cursor.execute(create_table_sql)
        self.connection.commit()

        logger.debug("Cache table created or verified")

    def store_entry(self, entry: CacheEntry) -> None:
        """Store a cache entry in the SQLite database.

        Serializes the embedding list to JSON for storage.

        Args:
            entry: The CacheEntry to persist to the database.
        """
        # Serialize the embedding vector to a JSON string
        embedding_json = json.dumps(entry.embedding)

        # Format the datetime as an ISO string for storage
        created_at_str = entry.created_at.isoformat()

        # Insert or replace the entry in the database
        insert_sql = """
            INSERT OR REPLACE INTO cache_entries
            (cache_key, request_prompt, response_content, model,
             embedding, created_at, hit_count, ttl_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                insert_sql,
                (
                    entry.cache_key,
                    entry.request_prompt,
                    entry.response_content,
                    entry.model,
                    embedding_json,
                    created_at_str,
                    entry.hit_count,
                    entry.ttl_seconds,
                ),
            )
            self.connection.commit()

            logger.debug("Entry stored in SQLite: %.16s...", entry.cache_key)

        except sqlite3.Error as db_error:
            logger.error("SQLite store error: %s", db_error)
            raise

    def get_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve a specific cache entry by its key.

        Args:
            cache_key: The unique key identifying the cache entry.

        Returns:
            The CacheEntry if found, None otherwise.
        """
        # Query the database for the specific cache key
        select_sql = "SELECT * FROM cache_entries WHERE cache_key = ?"

        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (cache_key,))
            row = cursor.fetchone()

            # Return None if no matching entry exists
            if row is None:
                return None

            # Convert the database row to a CacheEntry object
            return self._row_to_entry(row)

        except sqlite3.Error as db_error:
            logger.error("SQLite get error: %s", db_error)
            raise

    def get_all_entries(self) -> List[CacheEntry]:
        """Retrieve all cache entries from the database.

        Returns:
            List of all CacheEntry objects stored in the database.
        """
        # Query all entries from the cache table
        select_all_sql = "SELECT * FROM cache_entries"

        try:
            cursor = self.connection.cursor()
            cursor.execute(select_all_sql)
            rows = cursor.fetchall()

            # Convert each row to a CacheEntry object
            entries = [self._row_to_entry(row) for row in rows]

            logger.debug("Retrieved %d entries from SQLite", len(entries))
            return entries

        except sqlite3.Error as db_error:
            logger.error("SQLite get_all error: %s", db_error)
            raise

    def update_entry(self, entry: CacheEntry) -> None:
        """Update an existing cache entry in the database.

        Args:
            entry: The updated CacheEntry to persist.
        """
        # Update the hit count and other mutable fields
        update_sql = """
            UPDATE cache_entries
            SET hit_count = ?, response_content = ?
            WHERE cache_key = ?
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                update_sql,
                (entry.hit_count, entry.response_content, entry.cache_key),
            )
            self.connection.commit()

            logger.debug("Updated entry in SQLite: %.16s...", entry.cache_key)

        except sqlite3.Error as db_error:
            logger.error("SQLite update error: %s", db_error)
            raise

    def remove_expired(self) -> int:
        """Remove all expired entries from the database.

        Returns:
            Number of entries that were removed.
        """
        # Retrieve all entries and check expiration
        all_entries = self.get_all_entries()
        expired_keys = [
            entry.cache_key for entry in all_entries if entry.is_expired()
        ]

        if not expired_keys:
            return 0

        # Delete expired entries by their cache keys
        placeholders = ",".join("?" * len(expired_keys))
        delete_sql = f"DELETE FROM cache_entries WHERE cache_key IN ({placeholders})"

        try:
            cursor = self.connection.cursor()
            cursor.execute(delete_sql, expired_keys)
            self.connection.commit()

            logger.info("Removed %d expired entries from SQLite", len(expired_keys))
            return len(expired_keys)

        except sqlite3.Error as db_error:
            logger.error("SQLite remove_expired error: %s", db_error)
            raise

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        """Convert a database row to a CacheEntry object.

        Args:
            row: A sqlite3.Row object from a query result.

        Returns:
            A CacheEntry populated with data from the row.
        """
        # Deserialize the JSON embedding back to a list of floats
        embedding_list = json.loads(row["embedding"])

        # Parse the ISO datetime string back to a datetime object
        created_at = datetime.fromisoformat(row["created_at"])

        # Construct and return the CacheEntry
        return CacheEntry(
            cache_key=row["cache_key"],
            request_prompt=row["request_prompt"],
            response_content=row["response_content"],
            model=row["model"],
            embedding=embedding_list,
            created_at=created_at,
            hit_count=row["hit_count"],
            ttl_seconds=row["ttl_seconds"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
        logger.info("SQLite connection closed")
