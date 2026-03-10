import re
from unidecode import unidecode
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from higra_agent.model_cache import ModelCache





class BM25Retriever:
    """
    BM25-based retrieval over a list of documents.
    """
    def __init__(self, docs: List[str], top_k=20, k1=1.5, b=0.75, epsilon=0.25):
        self.docs = docs
        # Pre-tokenize documents
        self.tokenized_docs = [self._tokenize(doc) for doc in docs]
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=k1,
            b=b,
            epsilon=epsilon
        )
        self.top_k=top_k

    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', ' ', text)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a string into a list of tokens combining model tokenizer and simple split.
        """
        text = text.lower()
        # optionally normalize accents
        text = unidecode(text)
        # model_tokens = ModelCache._tokenizer.tokenize(text)
        manual_tokens = self._remove_punctuation(text).split()
        # tokens = model_tokens + manual_tokens
        return manual_tokens

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Returns top_k documents and their BM25 scores for the given query.
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            [(self.docs[i], float(scores[i])) for i in range(len(self.docs))],
            key=lambda x: x[1], reverse=True
        )
        ranked = ranked[:self.top_k]
        ranked = [x[0] for x in ranked]
        return ranked

    def rank(self, query: str) -> List[Tuple[int, float]]:
        """
        Returns ranking of all document indices and their BM25 scores for the given query.
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            [(i, float(scores[i])) for i in range(len(self.docs))],
            key=lambda x: x[1], reverse=True
        )
        return ranked