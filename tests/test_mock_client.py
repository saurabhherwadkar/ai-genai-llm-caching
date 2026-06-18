# -------------------------------------------------------------------
# test_mock_client.py
# Unit tests for the MockLlmClient class.
# -------------------------------------------------------------------

import pytest

from src.llm.mock_client import MockLlmClient
from src.models.llm_request import LlmRequest


class TestMockLlmClient:
    """Tests for the MockLlmClient class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = MockLlmClient()

    def test_send_request_returns_response(self):
        """send_request should return a valid LlmResponse."""
        request = LlmRequest(prompt="What is Python?")
        response = self.client.send_request(request)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    def test_response_not_marked_as_cached(self):
        """Mock responses should not be marked as cached."""
        request = LlmRequest(prompt="Test prompt")
        response = self.client.send_request(request)
        assert response.cached is False

    def test_response_model_is_mock(self):
        """Mock responses should report the mock model name."""
        request = LlmRequest(prompt="Test prompt")
        response = self.client.send_request(request)
        assert response.model == "mock-gpt-4"

    def test_deterministic_responses(self):
        """Same prompt should produce the same response content."""
        request = LlmRequest(prompt="What is machine learning?")
        response_a = self.client.send_request(request)
        response_b = self.client.send_request(request)
        assert response_a.content == response_b.content

    def test_different_prompts_produce_different_responses(self):
        """Different prompts should produce different response content."""
        request_a = LlmRequest(prompt="What is Python?")
        request_b = LlmRequest(prompt="What is JavaScript?")
        response_a = self.client.send_request(request_a)
        response_b = self.client.send_request(request_b)
        assert response_a.content != response_b.content

    def test_call_count_increments(self):
        """call_count should increment with each request."""
        assert self.client.call_count == 0

        self.client.send_request(LlmRequest(prompt="First"))
        assert self.client.call_count == 1

        self.client.send_request(LlmRequest(prompt="Second"))
        assert self.client.call_count == 2

    def test_token_counts_are_positive(self):
        """Token counts should be positive non-zero values."""
        request = LlmRequest(prompt="This is a test prompt with several words")
        response = self.client.send_request(request)

        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens
