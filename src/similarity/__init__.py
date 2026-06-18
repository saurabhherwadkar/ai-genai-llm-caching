# -------------------------------------------------------------------
# similarity package
# Contains syntactic and semantic similarity computation logic.
# -------------------------------------------------------------------

from src.similarity.syntactic_similarity import SyntacticSimilarity
from src.similarity.semantic_similarity import SemanticSimilarity
from src.similarity.similarity_engine import SimilarityEngine

__all__ = ["SyntacticSimilarity", "SemanticSimilarity", "SimilarityEngine"]
