# -------------------------------------------------------------------
# semantic_similarity.py
# Computes semantic (meaning-based) similarity between prompts
# using sentence-transformers embeddings and cosine similarity.
# -------------------------------------------------------------------

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level logger for semantic similarity operations
logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """Computes semantic similarity between text strings using embeddings.

    Uses a pre-trained sentence-transformer model to encode text into
    dense vector embeddings, then measures cosine similarity to determine
    meaning-level closeness between prompts.

    Attributes:
        threshold: Minimum similarity score to consider a semantic match.
        model_name: Name of the sentence-transformer model to use.
        model: The loaded sentence-transformer model instance.
    """

    def __init__(
        self, threshold: float = 0.80, model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the semantic similarity calculator.

        Args:
            threshold: Minimum score (0.0 to 1.0) for semantic match.
            model_name: Name of the sentence-transformer model to load.
        """
        self.threshold = threshold  # Store the semantic threshold
        self.model_name = model_name  # Store model identifier

        # Load the sentence-transformer model for embedding generation
        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for the given text.

        Encodes the text into a fixed-size dense vector that captures
        its semantic meaning.

        Args:
            text: The text string to encode into an embedding.

        Returns:
            List of floats representing the embedding vector.
        """
        # Encode the text using the sentence-transformer model
        embedding_array = self.model.encode(text, convert_to_numpy=True)

        # Convert numpy array to Python list for serialization compatibility
        embedding_list = embedding_array.tolist()

        logger.debug(
            "Generated embedding of dimension %d for text of length %d",
            len(embedding_list),
            len(text),
        )

        return embedding_list

    def compute_cosine_similarity(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Cosine similarity measures the angle between two vectors,
        returning 1.0 for identical directions and 0.0 for orthogonal vectors.

        Args:
            embedding_a: First embedding vector as a list of floats.
            embedding_b: Second embedding vector as a list of floats.

        Returns:
            Cosine similarity score between 0.0 and 1.0.
        """
        # Convert lists to numpy arrays for efficient computation
        vector_a = np.array(embedding_a)
        vector_b = np.array(embedding_b)

        # Compute the dot product of the two vectors
        dot_product = np.dot(vector_a, vector_b)

        # Compute the L2 norms (magnitudes) of both vectors
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        # Guard against division by zero for zero-length vectors
        if norm_a == 0.0 or norm_b == 0.0:
            logger.warning("Zero-norm vector detected in cosine similarity")
            return 0.0

        # Compute cosine similarity as dot product divided by product of norms
        cosine_score = dot_product / (norm_a * norm_b)

        return float(cosine_score)

    def compute_score(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity score between two text strings.

        Generates embeddings for both texts and computes their cosine similarity.

        Args:
            text_a: The first text string to compare.
            text_b: The second text string to compare.

        Returns:
            Semantic similarity score between 0.0 and 1.0.
        """
        # Generate embeddings for both input texts
        embedding_a = self.generate_embedding(text_a)
        embedding_b = self.generate_embedding(text_b)

        # Compute and return the cosine similarity between embeddings
        score = self.compute_cosine_similarity(embedding_a, embedding_b)

        logger.debug("Semantic similarity score: %.4f", score)
        return score

    def is_similar(self, text_a: str, text_b: str) -> bool:
        """Determine if two texts are semantically similar.

        Args:
            text_a: The first text string to compare.
            text_b: The second text string to compare.

        Returns:
            True if the similarity score meets or exceeds the threshold.
        """
        # Compute the semantic score and compare against threshold
        score = self.compute_score(text_a, text_b)
        return score >= self.threshold

    def compute_score_from_embeddings(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        """Compute similarity from pre-computed embeddings (avoids re-encoding).

        Args:
            embedding_a: Pre-computed embedding for the first text.
            embedding_b: Pre-computed embedding for the second text.

        Returns:
            Cosine similarity score between 0.0 and 1.0.
        """
        return self.compute_cosine_similarity(embedding_a, embedding_b)
