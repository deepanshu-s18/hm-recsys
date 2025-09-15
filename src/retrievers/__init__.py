"""Retrieval module: Popularity, ALS, Two-Tower, and Fusion retrievers.

All retrievers conform to the BaseRetriever interface:
    fit(dataset) -> self
    retrieve(user_ids, top_k) -> polars.DataFrame
"""

from src.retrievers.als import ALSRetriever
from src.retrievers.fusion import CandidateFusion
from src.retrievers.popularity import PopularityRetriever
from src.retrievers.two_tower import TwoTowerRetriever

__all__ = [
    "PopularityRetriever",
    "ALSRetriever",
    "TwoTowerRetriever",
    "CandidateFusion",
]
