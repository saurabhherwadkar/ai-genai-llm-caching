# -------------------------------------------------------------------
# openai_client.py
# OpenAI API client implementation for sending LLM requests.
# -------------------------------------------------------------------

import logging
from typing import Dict, Any

from openai import OpenAI

from src.llm.llm_client import LlmClient
from src.models.llm_request import LlmRequest
from src.models.llm_response import LlmResponse
from src.exceptions import LlmClientError

# Module-level logger for OpenAI client operations
logger = logging.getLogger(__name__)


class OpenAiClient(LlmClient):
    """OpenAI API client for sending requests to GPT models.

    Wraps the OpenAI Python SDK to provide a consistent interface
    for LLM interactions within the caching framework.

    Attributes:
        client: The OpenAI SDK client instance.
        default_model: Default model to use if not specified in request.
        timeout_seconds: Request timeout duration.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize the OpenAI client with configuration.

        Args:
            llm_config: Dictionary containing API key, model, and timeout settings.

        Raises:
            LlmClientError: If the API key is missing or invalid.
        """
        # Extract the API key from configuration
        api_key = llm_config.get("api_key", "")

        # Validate that an API key is provided
        if not api_key:
            error_message = "OpenAI API key is required but not configured"
            logger.error(error_message)
            raise LlmClientError(error_message)

        # Initialize the OpenAI SDK client with the API key
        self.client = OpenAI(api_key=api_key)

        # Store default configuration values
        self.default_model = llm_config.get("model", "gpt-4")
        self.timeout_seconds = llm_config.get("timeout_seconds", 30)

        logger.info("OpenAI client initialized with model: %s", self.default_model)

    def send_request(self, request: LlmRequest) -> LlmResponse:
        """Send a request to the OpenAI API and return the response.

        Args:
            request: The LlmRequest containing prompt and parameters.

        Returns:
            An LlmResponse containing the generated text and token usage.

        Raises:
            LlmClientError: If the API call fails or times out.
        """
        # Build the messages list for the chat completion API
        messages = []

        # Add system message if provided
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})

        # Add the user prompt as the main message
        messages.append({"role": "user", "content": request.prompt})

        # Determine which model to use (request-level overrides default)
        model = request.model or self.default_model

        logger.info("Sending request to OpenAI model: %s", model)

        try:
            # Call the OpenAI chat completion API
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=self.timeout_seconds,
            )

            # Extract the generated content from the response
            response_content = completion.choices[0].message.content

            # Extract token usage information
            usage = completion.usage

            # Build and return the LlmResponse object
            response = LlmResponse(
                content=response_content,
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                cached=False,
            )

            logger.info(
                "OpenAI response received. Tokens used: %d", response.total_tokens
            )

            return response

        except Exception as api_error:
            # Wrap any API errors in our custom exception type
            error_message = f"OpenAI API request failed: {str(api_error)}"
            logger.error(error_message)
            raise LlmClientError(error_message) from api_error
