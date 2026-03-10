from typing import List, Tuple, Dict

from higra_agent.retriever.bm25_retriever.bm25_retriever import BM25Retriever
from higra_agent.retriever.dense_retriever.dense_retriever import DenseRetriever



class HybridRetriever:
    """
    Combines BM25 and Dense retrievers using reciprocal rank fusion.
    """
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.docs = self.bm25_retriever.docs  # assumes both retrievers use same doc list

    def retrieve(self, query: str, q_emb) -> List[Tuple[str, float]]:
        bm25_results = self.bm25_retriever.retrieve(query)
        dense_results = self.dense_retriever.retrieve(query, q_emb)

        rr_scores: Dict[str, float] = {}
        for rank_list in (bm25_results, dense_results):
            for rank, doc in enumerate(rank_list, start=1):
                rr_scores[doc] = rr_scores.get(doc, 0.0) + 1.0 / rank

        ranked = sorted(rr_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = [x[0] for x in ranked]
        return ranked

    def rank(self, query: str, q_emb) -> List[int]:
        """
        Returns ranking of all document indices using reciprocal rank fusion (no scores).
        """
        bm25_indices = self.bm25_retriever.rank(query)
        dense_indices = self.dense_retriever.rank(query, q_emb)

        bm25_indices = [x[0] for x in bm25_indices]
        dense_indices = [x[0] for x in dense_indices]
        
        rr_scores: Dict[int, float] = {}

        for rank_list in (bm25_indices, dense_indices):
            for rank, idx in enumerate(rank_list, start=1):
                rr_scores[idx] = rr_scores.get(idx, 0.0) + 1.0 / rank

        ranked_indices = sorted(rr_scores, key=lambda i: rr_scores[i], reverse=True)
        return ranked_indices