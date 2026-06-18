# -------------------------------------------------------------------
# syntactic_similarity.py
# Computes syntactic (text-based) similarity between two prompts
# using difflib SequenceMatcher for character-level comparison.
# -------------------------------------------------------------------

import logging
from difflib import SequenceMatcher

# Module-level logger for syntactic similarity operations
logger = logging.getLogger(__name__)


class SyntacticSimilarity:
    """Computes syntactic similarity between text strings.

    Uses Python's SequenceMatcher to determine how similar two strings are
    based on their character sequences. This catches near-duplicate prompts
    with minor typos, rephrasing, or whitespace differences.

    Attributes:
        threshold: Minimum similarity score to consider a match.
    """

    def __init__(self, threshold: float = 0.85):
        """Initialize the syntactic similarity calculator.

        Args:
            threshold: Minimum score (0.0 to 1.0) to classify as similar.
        """
        self.threshold = threshold  # Store the similarity threshold

    def compute_score(self, text_a: str, text_b: str) -> float:
        """Compute the syntactic similarity score between two texts.

        Normalizes both texts (lowercase, stripped whitespace) before
        comparing them using the SequenceMatcher ratio algorithm.

        Args:
            text_a: The first text string to compare.
            text_b: The second text string to compare.

        Returns:
            A float between 0.0 (completely different) and 1.0 (identical).
        """
        # Normalize both inputs to lowercase and strip surrounding whitespace
        normalized_a = text_a.lower().strip()
        normalized_b = text_b.lower().strip()

        # Use SequenceMatcher to compute character-level similarity ratio
        similarity_ratio = SequenceMatcher(
            None, normalized_a, normalized_b
        ).ratio()

        logger.debug(
            "Syntactic similarity: %.4f between texts of length %d and %d",
            similarity_ratio,
            len(normalized_a),
            len(normalized_b),
        )

        return similarity_ratio

    def is_similar(self, text_a: str, text_b: str) -> bool:
        """Determine if two texts are syntactically similar.

        Args:
            text_a: The first text string to compare.
            text_b: The second text string to compare.

        Returns:
            True if the similarity score meets or exceeds the threshold.
        """
        # Compute the score and compare against the configured threshold
        score = self.compute_score(text_a, text_b)
        return score >= self.threshold
