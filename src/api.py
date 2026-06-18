# -------------------------------------------------------------------
# api.py
# FastAPI REST API layer exposing the LLM caching service endpoints.
# -------------------------------------------------------------------

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config_loader import ConfigLoader
from src.logger_setup import setup_logging
from src.llm_cache_service import LlmCacheService
from src.models.llm_request import LlmRequest
from src.exceptions import LlmClientError

# Module-level logger for API operations
logger = logging.getLogger(__name__)


# --- Pydantic request/response models for the REST API ---


class PromptRequest(BaseModel):
    """Request body for the /query endpoint.

    Attributes:
        prompt: The user prompt to send to the LLM.
        model: Optional model override.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        system_message: Optional system instruction.
    """

    prompt: str = Field(..., description="The user prompt text")
    model: Optional[str] = Field(None, description="Model override")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(1024, description="Maximum response tokens")
    system_message: Optional[str] = Field(None, description="System instruction")


class PromptResponse(BaseModel):
    """Response body for the /query endpoint.

    Attributes:
        content: The generated or cached response text.
        model: The model that produced the response.
        cached: Whether the response came from cache.
        similarity_score: Similarity score if cache hit.
        match_type: Type of similarity match (syntactic/semantic).
        tokens_used: Total tokens consumed (0 if cached).
    """

    content: str = Field(..., description="Response text")
    model: str = Field(..., description="Model used")
    cached: bool = Field(..., description="Whether served from cache")
    similarity_score: float = Field(0.0, description="Similarity score")
    match_type: Optional[str] = Field(None, description="Match type")
    tokens_used: int = Field(0, description="Total tokens consumed")


class StatsResponse(BaseModel):
    """Response body for the /stats endpoint.

    Attributes:
        total_requests: Total number of requests processed.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses.
        hit_rate_percent: Cache hit rate percentage.
        total_entries: Number of entries in cache.
        active_entries: Non-expired entries.
    """

    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate_percent: float
    total_entries: int
    active_entries: int


# --- Application factory ---


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Loads configuration, sets up logging, initializes the cache service,
    and registers all API route handlers.

    Returns:
        Configured FastAPI application ready to serve requests.
    """
    # Load application configuration from file and environment
    config = ConfigLoader()

    # Set up application-wide logging
    setup_logging(config.get_section("logging"))

    # Initialize the LLM cache service with all required configuration
    service = LlmCacheService(
        cache_config=config.get_section("cache"),
        similarity_config=config.get_section("similarity"),
        llm_config=config.get_section("llm"),
    )

    # Create the FastAPI application with metadata
    app = FastAPI(
        title="LLM Cache Service",
        description="Intelligent LLM response caching with similarity detection",
        version="1.0.0",
    )

    # --- Route handlers ---

    @app.post("/query", response_model=PromptResponse)
    async def query_llm(request_body: PromptRequest) -> PromptResponse:
        """Process an LLM query with caching.

        Checks the cache for similar prompts before forwarding to the LLM.

        Args:
            request_body: The incoming prompt request.

        Returns:
            PromptResponse with generated or cached content.

        Raises:
            HTTPException: If the LLM request fails.
        """
        # Convert the API request to an internal LlmRequest model
        llm_request = LlmRequest(
            prompt=request_body.prompt,
            model=request_body.model or config.get("llm", "model", "gpt-4"),
            temperature=request_body.temperature,
            max_tokens=request_body.max_tokens,
            system_message=request_body.system_message,
        )

        try:
            # Process the request through the caching service
            response = service.process_request(llm_request)

            # Build the API response from the internal response
            return PromptResponse(
                content=response.content,
                model=response.model,
                cached=response.cached,
                similarity_score=response.similarity_score,
                match_type=response.metadata.get("match_type"),
                tokens_used=response.total_tokens,
            )

        except LlmClientError as client_error:
            # Return a 502 error for upstream LLM failures
            logger.error("LLM client error: %s", client_error)
            raise HTTPException(
                status_code=502,
                detail=f"LLM provider error: {str(client_error)}",
            )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """Retrieve cache and service statistics.

        Returns:
            StatsResponse with current metrics.
        """
        # Get statistics from the service layer
        stats = service.get_stats()

        return StatsResponse(
            total_requests=stats["total_requests"],
            cache_hits=stats["cache_hits"],
            cache_misses=stats["cache_misses"],
            hit_rate_percent=stats["hit_rate_percent"],
            total_entries=stats["total_entries"],
            active_entries=stats["active_entries"],
        )

    @app.post("/cache/clear-expired")
    async def clear_expired_cache() -> Dict[str, Any]:
        """Remove expired entries from the cache.

        Returns:
            Dictionary with the count of removed entries.
        """
        removed_count = service.clear_expired_cache()
        return {"removed_entries": removed_count}

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint for monitoring.

        Returns:
            Status indicator for the service.
        """
        return {"status": "healthy"}

    return app


# Create the application instance for uvicorn
app = create_app()
