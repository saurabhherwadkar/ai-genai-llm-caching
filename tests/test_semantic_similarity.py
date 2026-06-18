# -------------------------------------------------------------------
# test_semantic_similarity.py
# Unit tests for the SemanticSimilarity class.
# -------------------------------------------------------------------

import pytest

from src.similarity.semantic_similarity import SemanticSimilarity


class TestSemanticSimilarity:
    """Tests for the SemanticSimilarity class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures. Uses a lightweight model for speed."""
        # Initialize with the default model (will download on first run)
        self.similarity = SemanticSimilarity(
            threshold=0.80, model_name="all-MiniLM-L6-v2"
        )

    def test_identical_texts_return_high_score(self):
        """Identical texts should return a score close to 1.0."""
        text = "What is machine learning?"
        score = self.similarity.compute_score(text, text)
        assert score > 0.99

    def test_semantically_similar_texts_return_high_score(self):
        """Texts with same meaning but different wording should score high."""
        text_a = "What is machine learning and how does it work?"
        text_b = "Explain how machine learning works and what it is"
        score = self.similarity.compute_score(text_a, text_b)
        assert score > 0.70

    def test_unrelated_texts_return_low_score(self):
        """Completely unrelated texts should return a low score."""
        text_a = "What is quantum computing?"
        text_b = "How to bake chocolate cake at home?"
        score = self.similarity.compute_score(text_a, text_b)
        assert score < 0.4

    def test_generate_embedding_returns_list(self):
        """generate_embedding should return a list of floats."""
        text = "Test embedding generation"
        embedding = self.similarity.generate_embedding(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

    def test_embedding_dimension_consistency(self):
        """All embeddings should have the same dimension."""
        text_a = "First sentence"
        text_b = "Second completely different sentence with more words"
        embedding_a = self.similarity.generate_embedding(text_a)
        embedding_b = self.similarity.generate_embedding(text_b)
        assert len(embedding_a) == len(embedding_b)

    def test_cosine_similarity_with_same_vector(self):
        """Cosine similarity of a vector with itself should be 1.0."""
        embedding = self.similarity.generate_embedding("Test text")
        score = self.similarity.compute_cosine_similarity(embedding, embedding)
        assert abs(score - 1.0) < 0.001

    def test_cosine_similarity_with_zero_vector(self):
        """Cosine similarity with a zero vector should return 0.0."""
        embedding = self.similarity.generate_embedding("Test text")
        zero_vector = [0.0] * len(embedding)
        score = self.similarity.compute_cosine_similarity(embedding, zero_vector)
        assert score == 0.0

    def test_is_similar_above_threshold(self):
        """is_similar should return True for semantically close texts."""
        text_a = "How do neural networks learn?"
        text_b = "What is the learning process of neural networks?"
        result = self.similarity.is_similar(text_a, text_b)
        assert result is True

    def test_is_similar_below_threshold(self):
        """is_similar should return False for unrelated texts."""
        text_a = "How do neural networks learn?"
        text_b = "What is the capital of France?"
        result = self.similarity.is_similar(text_a, text_b)
        assert result is False

    def test_compute_score_from_embeddings(self):
        """Pre-computed embeddings should give same score as compute_score."""
        text_a = "Machine learning explanation"
        text_b = "Deep learning overview"
        # Compute using text directly
        direct_score = self.similarity.compute_score(text_a, text_b)
        # Compute using pre-generated embeddings
        emb_a = self.similarity.generate_embedding(text_a)
        emb_b = self.similarity.generate_embedding(text_b)
        embedding_score = self.similarity.compute_score_from_embeddings(emb_a, emb_b)
        # Scores should be approximately equal
        assert abs(direct_score - embedding_score) < 0.01
