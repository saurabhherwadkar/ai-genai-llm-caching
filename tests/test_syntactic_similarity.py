# -------------------------------------------------------------------
# test_syntactic_similarity.py
# Unit tests for the SyntacticSimilarity class.
# -------------------------------------------------------------------

import pytest

from src.similarity.syntactic_similarity import SyntacticSimilarity


class TestSyntacticSimilarity:
    """Tests for the SyntacticSimilarity class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a similarity instance with default threshold
        self.similarity = SyntacticSimilarity(threshold=0.85)

    def test_identical_strings_return_perfect_score(self):
        """Identical strings should return a score of 1.0."""
        text = "What is machine learning?"
        score = self.similarity.compute_score(text, text)
        assert score == 1.0

    def test_completely_different_strings_return_low_score(self):
        """Completely different strings should return a low score."""
        text_a = "What is machine learning?"
        text_b = "The quick brown fox jumps over the lazy dog"
        score = self.similarity.compute_score(text_a, text_b)
        assert score < 0.5

    def test_similar_strings_with_minor_differences(self):
        """Strings with minor differences should return a high score."""
        text_a = "What is machine learning and how does it work?"
        text_b = "What is machine learning and how does it work"
        score = self.similarity.compute_score(text_a, text_b)
        assert score > 0.95

    def test_case_insensitivity(self):
        """Comparison should be case-insensitive."""
        text_a = "What Is Machine Learning?"
        text_b = "what is machine learning?"
        score = self.similarity.compute_score(text_a, text_b)
        assert score == 1.0

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should be normalized."""
        text_a = "  What is machine learning?  "
        text_b = "What is machine learning?"
        score = self.similarity.compute_score(text_a, text_b)
        assert score == 1.0

    def test_is_similar_returns_true_above_threshold(self):
        """is_similar should return True when score exceeds threshold."""
        text_a = "What is machine learning?"
        text_b = "What is machine learning"
        result = self.similarity.is_similar(text_a, text_b)
        assert result is True

    def test_is_similar_returns_false_below_threshold(self):
        """is_similar should return False when score is below threshold."""
        text_a = "What is machine learning?"
        text_b = "How to cook pasta at home?"
        result = self.similarity.is_similar(text_a, text_b)
        assert result is False

    def test_empty_strings_return_perfect_score(self):
        """Two empty strings should return a score of 1.0."""
        score = self.similarity.compute_score("", "")
        assert score == 1.0

    def test_one_empty_string_returns_zero(self):
        """Comparing a non-empty string with empty should return 0.0."""
        score = self.similarity.compute_score("hello world", "")
        assert score == 0.0

    def test_custom_threshold(self):
        """Custom threshold should be respected in is_similar."""
        # Create an instance with a lower threshold
        lenient_similarity = SyntacticSimilarity(threshold=0.5)
        text_a = "What is machine learning?"
        text_b = "What is deep learning?"
        result = lenient_similarity.is_similar(text_a, text_b)
        assert result is True
