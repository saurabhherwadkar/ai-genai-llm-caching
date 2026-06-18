# -------------------------------------------------------------------
# exceptions.py
# Custom exception classes for domain-specific error handling.
# -------------------------------------------------------------------


class LlmCacheError(Exception):
    """Base exception for all LLM Cache application errors.

    All custom exceptions in this application inherit from this class
    to allow catching any application-specific error with a single handler.
    """

    pass


class LlmClientError(LlmCacheError):
    """Raised when an LLM provider API call fails.

    This may occur due to network issues, authentication failures,
    rate limiting, or invalid request parameters.
    """

    pass


class CacheStorageError(LlmCacheError):
    """Raised when a cache storage operation fails.

    This may occur due to database connection issues, disk space,
    or data serialization problems.
    """

    pass


class ConfigurationError(LlmCacheError):
    """Raised when application configuration is invalid or missing.

    This may occur when required settings are absent or
    environment variables are not properly set.
    """

    pass


class SimilarityComputationError(LlmCacheError):
    """Raised when similarity computation fails.

    This may occur due to model loading issues, invalid embeddings,
    or numerical computation errors.
    """

    pass
