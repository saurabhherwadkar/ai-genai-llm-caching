# -------------------------------------------------------------------
# llm_client.py
# Abstract base class defining the interface for LLM provider clients.
# -------------------------------------------------------------------

from abc import ABC, abstractmethod

from src.models.llm_request import LlmRequest
from src.models.llm_response import LlmResponse


class LlmClient(ABC):
    """Abstract base class for LLM provider clients.

    Defines the contract that all LLM client implementations must follow.
    This allows swapping providers via dependency injection without
    changing the consuming code.
    """

    @abstractmethod
    def send_request(self, request: LlmRequest) -> LlmResponse:
        """Send a request to the LLM provider and return the response.

        Args:
            request: The LlmRequest containing prompt and parameters.

        Returns:
            An LlmResponse containing the generated text and metadata.

        Raises:
            LlmClientError: If the request to the provider fails.
        """
        pass
