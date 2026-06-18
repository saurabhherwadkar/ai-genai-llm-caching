# -------------------------------------------------------------------
# similarity_engine.py
# Orchestrates both syntactic and semantic similarity checks to
# determine if a new prompt matches any cached prompt.
# -------------------------------------------------------------------

import logging
from typing import Dict, Any, List, Optional, Tuple

from src.models.cache_entry import CacheEntry
from src.similarity.syntactic_similarity import SyntacticSimilarity
from src.similarity.semantic_similarity import SemanticSimilarity

# Module-level logger for similarity engine operations
logger = logging.getLogger(__name__)


class SimilarityEngine:
    """Orchestrates syntactic and semantic similarity checks.

    First performs a fast syntactic check. If that doesn't find a match,
    falls back to the more expensive semantic similarity comparison.
    This two-tier approach optimizes for speed while maintaining accuracy.

    Attributes:
        syntactic: The syntactic similarity calculator instance.
        semantic: The semantic similarity calculator instance.
    """

    def __init__(self, similarity_config: Dict[str, Any]):
        """Initialize the similarity engine with configuration.

        Args:
            similarity_config: Dictionary containing similarity thresholds
                             and embedding model configuration.
        """
        # Extract threshold values from configuration
        syntactic_threshold = similarity_config.get("syntactic_threshold", 0.85)
        semantic_threshold = similarity_config.get("semantic_threshold", 0.80)
        embedding_model = similarity_config.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )

        # Initialize the syntactic similarity calculator
        self.syntactic = SyntacticSimilarity(threshold=syntactic_threshold)
        logger.info(
            "Syntactic similarity initialized with threshold: %.2f",
            syntactic_threshold,
        )

        # Initialize the semantic similarity calculator with the model
        self.semantic = SemanticSimilarity(
            threshold=semantic_threshold, model_name=embedding_model
        )
        logger.info(
            "Semantic similarity initialized with threshold: %.2f",
            semantic_threshold,
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text.

        Delegates to the semantic similarity model for embedding generation.

        Args:
            text: The text to encode into an embedding vector.

        Returns:
            List of floats representing the dense embedding vector.
        """
        return self.semantic.generate_embedding(text)

    def find_best_match(
        self, query_prompt: str, cache_entries: List[CacheEntry]
    ) -> Optional[Tuple[CacheEntry, float, str]]:
        """Find the best matching cache entry for a given prompt.

        Performs a two-tier similarity search:
        1. Syntactic check (fast, character-level comparison)
        2. Semantic check (slower, meaning-level via embeddings)

        Returns the best match if any entry exceeds the configured thresholds.

        Args:
            query_prompt: The new prompt to match against cached entries.
            cache_entries: List of existing cache entries to search through.

        Returns:
            Tuple of (matching CacheEntry, similarity score, match type)
            where match type is "syntactic" or "semantic", or None if no match.
        """
        # Return None immediately if there are no entries to compare
        if not cache_entries:
            logger.debug("No cache entries to compare against")
            return None

        # --- Phase 1: Syntactic Similarity Check (Fast) ---
        best_syntactic_entry = None
        best_syntactic_score = 0.0

        for entry in cache_entries:
            # Skip expired entries to avoid returning stale data
            if entry.is_expired():
                continue

            # Compute syntactic similarity against this entry's prompt
            score = self.syntactic.compute_score(
                query_prompt, entry.request_prompt
            )

            # Track the best syntactic match found so far
            if score > best_syntactic_score:
                best_syntactic_score = score
                best_syntactic_entry = entry

        # If syntactic match exceeds threshold, return it immediately
        if (
            best_syntactic_entry
            and best_syntactic_score >= self.syntactic.threshold
        ):
            logger.info(
                "Syntactic cache hit with score %.4f", best_syntactic_score
            )
            return (best_syntactic_entry, best_syntactic_score, "syntactic")

        # --- Phase 2: Semantic Similarity Check (Slower, More Accurate) ---
        # Generate embedding for the query prompt
        query_embedding = self.semantic.generate_embedding(query_prompt)

        best_semantic_entry = None
        best_semantic_score = 0.0

        for entry in cache_entries:
            # Skip expired entries
            if entry.is_expired():
                continue

            # Skip entries without pre-computed embeddings
            if not entry.embedding:
                continue

            # Compute cosine similarity between query and cached embeddings
            score = self.semantic.compute_score_from_embeddings(
                query_embedding, entry.embedding
            )

            # Track the best semantic match found so far
            if score > best_semantic_score:
                best_semantic_score = score
                best_semantic_entry = entry

        # If semantic match exceeds threshold, return it
        if (
            best_semantic_entry
            and best_semantic_score >= self.semantic.threshold
        ):
            logger.info(
                "Semantic cache hit with score %.4f", best_semantic_score
            )
            return (best_semantic_entry, best_semantic_score, "semantic")

        # No match found in either tier
        logger.debug(
            "No cache match found. Best syntactic: %.4f, Best semantic: %.4f",
            best_syntactic_score,
            best_semantic_score,
        )
        return None
