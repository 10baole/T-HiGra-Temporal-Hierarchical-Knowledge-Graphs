import os
import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer


def fixed_size_chunking_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks


class DenseRetriever:
    """
    Dense retrieval using FAISS for fast similarity search over document chunks.
    Supports dynamic indexing and persistent load/save without re-encoding.
    """
    def __init__(
        self,
        docs: List[str],
        top_k: int,
        embedder: Optional[SentenceTransformer] = None,
        embedding_path: Optional[str] = None,
        faiss_metric: str = 'L2',
        chunk_size: int = 256,
        overlap: int = 20,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        self.docs = docs
        self.top_k = top_k
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.nlist = nlist
        self.nprobe = nprobe
        # interpret metric
        self.use_ip = True if faiss_metric.upper() == 'IP' else False
        # prepare storage paths
        self.embedding_path = embedding_path
        self.index: Optional[faiss.Index] = None
        self.chunk_to_doc: List[int]
        self.chunk_embeddings: np.ndarray
            
        # if persistent files exist, load them
        if embedding_path and os.path.isdir(embedding_path):
            idx_file = os.path.join(embedding_path, 'index.faiss')
            map_file = os.path.join(embedding_path, 'mapping.npy')
            emb_file = os.path.join(embedding_path, 'embeddings.npy')
            if os.path.exists(idx_file) and os.path.exists(map_file):
                # load mapping
                self.chunk_to_doc = np.load(map_file).tolist()
                # load embeddings if available
                if os.path.exists(emb_file):
                    self.chunk_embeddings = np.load(emb_file)
                # load index
                self.index = faiss.read_index(idx_file)
                self.index.nprobe = self.nprobe
                if self.top_k is None: self.top_k = len(self.chunk_to_doc)
                return
        # else, build from scratch
        # prepare chunks
        self.chunks, self.chunk_to_doc = self._prepare_chunks(docs)
        if self.top_k is None: self.top_k = len(self.chunk_to_doc)
            
        # compute embeddings
        if not self.embedder:
            raise ValueError("Embedder is required for DenseRetriever initialization.")
        self.chunk_embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        # normalize for IP
        if self.use_ip:
            faiss.normalize_L2(self.chunk_embeddings)
        # build FAISS index
        dim = self.chunk_embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dim) if self.use_ip else faiss.IndexFlatL2(dim)
        ivf = faiss.IndexIVFFlat(
            quantizer,
            dim,
            min(self.nlist, self.chunk_embeddings.shape[0]),
            faiss.METRIC_INNER_PRODUCT if self.use_ip else faiss.METRIC_L2
        )
        ivf.train(self.chunk_embeddings)
        ivf.add(self.chunk_embeddings)
        ivf.nprobe = self.nprobe
        self.index = ivf
        # save if path provided
        if embedding_path:
            os.makedirs(embedding_path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(embedding_path, 'index.faiss'))
            np.save(os.path.join(embedding_path, 'mapping.npy'), np.array(self.chunk_to_doc))
            np.save(os.path.join(embedding_path, 'embeddings.npy'), self.chunk_embeddings)

        if self.top_k is None: 
            self.top_k = len(self.chunk_to_doc)
            print("Dense top_k", self.top_k)

    @staticmethod
    def _prepare_chunks(
        docs: List[str], chunk_size: int = 256, overlap: int = 20
    ) -> Tuple[List[str], List[int]]:
        all_chunks: List[str] = []
        chunk_to_doc: List[int] = []
        for doc_idx, doc in enumerate(docs):
            chunks = fixed_size_chunking_with_overlap(doc, chunk_size, overlap)
            all_chunks.extend(chunks)
            chunk_to_doc.extend([doc_idx] * len(chunks))
        return all_chunks, chunk_to_doc

    def retrieve(
        self,
        query: str,
        q_emb: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        if not self.embedder:
            raise ValueError("Embedder is required for query embedding.")
        if q_emb is None:
            q_emb = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        if self.use_ip:
            faiss.normalize_L2(q_emb)

        
        D, I = self.index.search(q_emb.reshape(1, -1), self.top_k)
        scores, idxs = D[0], I[0]
        # aggregate by doc
        doc_scores: Dict[int, float] = {}
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            # Bounds check: skip indices that are out of range
            if idx >= len(self.chunk_to_doc):
                continue
            doc_idx = self.chunk_to_doc[idx]
            doc_scores[doc_idx] = max(score, doc_scores.get(doc_idx, float('-inf')))
        ranked = sorted(
            [(self.docs[d], float(s)) for d, s in doc_scores.items()],
            key=lambda x: x[1], reverse=False # Smallest First for Similarity Score
        )
        ranked = ranked[:self.top_k]
        ranked = [x[0] for x in ranked]
        return ranked

    def rank(
        self,
        query: str,
        q_emb: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """
        Return ranking of all documents based on the max score of their chunks.
        Only returns doc indices and scores (not full text).
        """
        if not self.embedder:
            raise ValueError("Embedder is required for query embedding.")
        if q_emb is None:
            q_emb = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        if self.use_ip:
            faiss.normalize_L2(q_emb)
        # Search all chunks
        D, I = self.index.search(q_emb.reshape(1, -1), len(self.chunk_to_doc))
        scores, idxs = D[0], I[0]
        # Aggregate max score per document
        doc_scores: Dict[int, float] = {}
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            # Bounds check: skip indices that are out of range
            if idx >= len(self.chunk_to_doc):
                continue
            doc_idx = self.chunk_to_doc[idx]
            doc_scores[doc_idx] = max(score, doc_scores.get(doc_idx, float('-inf')))
        # Return sorted list of doc indices and scores
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=False) # Smallest First for Similarity Score
        return ranked
        
    @classmethod
    def load(
        cls,
        docs: List[str],
        embedder: Optional[SentenceTransformer] = None,
        embedding_path: Optional[str] = None,
        faiss_metric: str = 'IP',
        chunk_size: int = 256,
        overlap: int = 20,
        nprobe: int = 10
    ) -> 'DenseRetriever':
        # instantiate and let __init__ handle loading
        return cls(
            docs,
            embedder=embedder,
            embedding_path=embedding_path,
            faiss_metric=faiss_metric,
            chunk_size=chunk_size,
            overlap=overlap,
            nlist=100,
            nprobe=nprobe
        )