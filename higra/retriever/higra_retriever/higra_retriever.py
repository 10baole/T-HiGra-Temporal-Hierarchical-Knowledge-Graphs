import re
import os
import json
import string
import operator
import numpy as np
import networkx as nx

from functools import reduce
from unidecode import unidecode
from flashtext import KeywordProcessor
from typing import List, Dict, Any, Optional, Set, Tuple

from higra_agent.model_cache import ModelCache
from higra_agent.ner.ner import NameEntityRecognizer
from higra_agent.retriever.bm25_retriever.bm25_retriever import BM25Retriever
from higra_agent.retriever.dense_retriever.dense_retriever import DenseRetriever
from higra_agent.retriever.hybrid_retriever.hybrid_retriever import HybridRetriever
from higra_agent.retriever.higra_retriever.higra_schema import HierarchicalKnowledgeGraph



class RetrieverConfig:
    """Configuration for HiGraRetriever."""

    def __init__(self, config_dict: dict= None):
        """
        Initialize configuration from a dictionary.

        Args:
            config_dict (dict, optional): A dictionary containing configuration values.
        """
        cfg = config_dict or {}

        self.ontology_top_k = int(cfg.get("ontology_top_k", 20))
        self.passage_top_k = int(cfg.get("passage_top_k", 20))
        self.entity_passage_top_k = int(cfg.get("entity_passage_top_k", 10))
        self.entity_sentence_top_k = int(cfg.get("entity_sentence_top_k", 5))
        self.prune_top_k = int(cfg.get("prune_top_k", 10))
        self.ppr_alpha = float(cfg.get("ppr_alpha", 0.85))
        self.rrf_semantic_constant = float(cfg.get("rrf_semantic_constant", 10))
        self.rrf_ppr_constant = int(cfg.get("rrf_ppr_constant", 10))
        self.top_sentence_in_entity_layer = int(cfg.get("top_sentence_in_entity_layer", 10))
        self.similarity_passage_top_k = int(cfg.get("similarity_passage_top_k", 5))

class HiGraRetriever:
    def __init__(
        self, 
        higra: HierarchicalKnowledgeGraph, 
        embedder, 
        api_key,
        model_name,
        config_dict,
        embedding_save_path: Optional[str] = None,
        
    ):
        self.higra = higra
        self.embedder = embedder
        self.embedding_save_path = embedding_save_path
        self.config = RetrieverConfig(config_dict)
        self.ner = NameEntityRecognizer(
            api_key=api_key,
            model_name=model_name
        )
        
        self._initialize_mappings()
        self._initialize_retrievers()

    def _initialize_mappings(self) -> None:
        """Initialize lookup dictionaries and adjacency mappings."""
        self.higra._build_mappings()
        self.passage_lookup = {p.passage_id: p for p in self.higra.passage_layer.passages}
        self.sentence_lookup = {s.sentence_id: s for s in self.higra.sentence_layer.sentences}
        self.node_lookup = {n.id: n for n in self.higra.entity_layer.nodes}
        self.edge_lookup = {e.edge_id: e for e in self.higra.entity_layer.edges}
        self.adj_map = self.higra.node_id_to_adjacency_node_id_map
        self.edge_map = self.higra.node_id_to_edge_id_map

    def _initialize_retrievers(self) -> None:
        """Initialize all retrievers for ontology, passages, sentences, and triplets."""
        self._init_graph()
        self._init_ontology_retriever()
        self._init_passage_retrievers()
        self._init_sentence_retrievers()
        self._init_triplet_retrievers()

    def _init_graph(self) -> None:
        """Build NetworkX graph for entity layer."""
        self.graph = nx.Graph()
        for node in self.higra.entity_layer.nodes:
            self.graph.add_node(node.id, name=node.name, type=node.type)
        for edge in self.higra.entity_layer.edges:
            self.graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                edge_id=edge.edge_id,
                relationship=edge.relationship_name
            )

    def _init_ontology_retriever(self) -> None:
        """Initialize BM25 retriever for ontology."""
        docs = [f"Name: $${n.name}$$; Type: [{','.join(n.type)}]; Alias: [{','.join(n.aliases or [])}]"
                for n in self.higra.entity_layer.nodes]
        self.ontology_bm25 = BM25Retriever(docs, top_k=self.config.ontology_top_k)
        
        keywords = list(self.higra.node_name_to_node_id_map.keys())
        self.keyword_processor = KeywordProcessor()
        self.keyword_processor.add_keywords_from_list(keywords)

    def _init_passage_retrievers(self) -> None:
        """Initialize passage retrievers (BM25, Dense, Hybrid)."""
        texts = [p.text for p in self.higra.passage_layer.passages]
        self.passage_id_list = [p.passage_id for p in self.higra.passage_layer.passages]
        passage_emb_path = os.path.join(self.embedding_save_path, "passage_dense") if self.embedding_save_path else None
        
        self.passage_bm25 = BM25Retriever(texts, top_k=None)
        self.passage_dense = DenseRetriever(
            texts,
            embedder=self.embedder,
            embedding_path=passage_emb_path,
            faiss_metric='L2',
            top_k=None
        )
        self.passage_hybrid = HybridRetriever(self.passage_bm25, self.passage_dense)

    def _init_sentence_retrievers(self) -> None:
        """Initialize sentence retrievers (BM25, Dense, Hybrid)."""
        texts = [s.text for s in self.higra.sentence_layer.sentences]
        self.sentence_id_list = [s.sentence_id for s in self.higra.sentence_layer.sentences]
        sentence_emb_path = os.path.join(self.embedding_save_path, "sentence_dense") if self.embedding_save_path else None
        
        self.sentence_bm25 = BM25Retriever(texts, top_k=None)
        self.sentence_dense = DenseRetriever(
            texts,
            embedder=self.embedder,
            embedding_path=sentence_emb_path,
            faiss_metric='L2',
            top_k=None
        )
        self.sentence_hybrid = HybridRetriever(self.sentence_bm25, self.sentence_dense)

    def _init_triplet_retrievers(self) -> None:
        """Initialize triplet retrievers (BM25, Dense, Hybrid)."""
        triplets = [
            f"{self.higra.node_id_to_node_name_map[e.source_node_id]}|{e.relationship_name}|"
            f"{self.higra.node_id_to_node_name_map[e.target_node_id]}"
            for e in self.higra.entity_layer.edges
        ]
        self.triplet_id_list = [(e.source_node_id, e.target_node_id) for e in self.higra.entity_layer.edges]
        triplet_emb_path = os.path.join(self.embedding_save_path, "triplet_dense") if self.embedding_save_path else None
        
        self.triplet_bm25 = BM25Retriever(triplets, top_k=None)
        self.triplet_dense = DenseRetriever(
            triplets,
            embedder=self.embedder,
            embedding_path=triplet_emb_path,
            faiss_metric='L2',
            top_k=None
        )
        self.triplet_hybrid = HybridRetriever(self.triplet_bm25, self.triplet_dense)

    def create_ontology(self, question: str) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
        """Retrieve ontology, passages, sentences, and triplets for a given question."""
        q_embedding = ModelCache.get_embedding_model().encode(
            question,
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cuda"
        )

        # Retrieve top related entities
        top_entities = self.ontology_bm25.retrieve(question)
        clean_question = re.sub(f"[{re.escape(string.punctuation)}]", " ", question).lower()
        keyword_entities = set(self.keyword_processor.extract_keywords(question))
        keyword_entities = [kw for kw in keyword_entities if re.search(rf'\b{re.escape(kw)}\b', clean_question)]

        # Map keyword matched entities to nodes
        matching_node_ids = [
            node_id for name in keyword_entities
            for node_id in self.higra.node_name_to_node_id_map.get(name.lower(), [])
        ]
        matching_nodes = [self.node_lookup[node_id] for node_id in matching_node_ids]
        keyword_matching_entities = [
            f"Name: $${n.name}$$; Type: [{','.join(n.type)}]; Alias: [{','.join(n.aliases or [])}]"
            for n in matching_nodes
        ]

        # Combine and deduplicate entities
        top_entities = keyword_matching_entities + [
            entity for entity in top_entities if entity not in keyword_matching_entities
        ]

        # Retrieve ranked passages, sentences, and triplets
        # Note: rank() should return List[int] of indices, but handle case where it returns IDs
        passage_indices = self.passage_hybrid.rank(question, q_embedding)
        sentence_indices = self.sentence_hybrid.rank(question, q_embedding)
        triplet_indices = self.triplet_hybrid.rank(question, q_embedding)
        
        # Handle both indices (int) and IDs (str) - if it's already an ID string, use it directly
        def get_id_from_index_or_id(item, id_list):
            if isinstance(item, str):
                # Already an ID, return as-is
                return item
            elif isinstance(item, int):
                # It's an index, look up the ID
                return id_list[item]
            else:
                # Fallback: try to use as index
                try:
                    return id_list[int(item)]
                except:
                    return item
        
        passage_ranks = [get_id_from_index_or_id(i, self.passage_id_list) for i in passage_indices]
        sentence_ranks = [get_id_from_index_or_id(i, self.sentence_id_list) for i in sentence_indices]
        triplet_ranks = [get_id_from_index_or_id(i, self.triplet_id_list) for i in triplet_indices]

        return top_entities, passage_ranks, sentence_ranks, triplet_ranks

    def get_context(
        self,
        entity_names,
        triplet_ranks,
        sentence_ranks,
        passage_ranks
    ):
        # Get seed id
        seed_ids = self._get_seed_node_ids(entity_names)  
    
        # Get Sub Graph
        sub_graph = self._get_subgraph(self.graph, seed_ids, radius=1)
        
        # Get nodes and edges
        edges_set = set(sub_graph.edges())
        nodes_set = set(sub_graph.nodes())

        # PPR Score
        personalization = {nid: 1.0 if nid in seed_ids else 0 for nid in sub_graph.nodes}
        ppr_scores = nx.pagerank(sub_graph, alpha=self.config.ppr_alpha, personalization=personalization)

        # Edge Ranking
        ranked_edges = self.rank_edges(triplet_ranks, edges_set, sub_graph, ppr_scores)

        # Prepare
        edge_id_list = [self.graph.edges[x]['edge_id'] for x in ranked_edges]
        edge_obj_list = [self.edge_lookup[edge_id] for edge_id in edge_id_list]
        sub_nodes = {e[0] for e in ranked_edges} | {e[1] for e in ranked_edges}
        sub_nodes |= seed_ids
        sub_node_names = [self.node_lookup[x].name for x in sub_nodes]

        ## Passage Ranking
        ranked_passages = self.rank_passages(sub_graph, seed_ids, passage_ranks)
    
        # Passage by Entities
        # passage_by_entities = self._get_entity_passages(sub_node_names, passage_ranks) if passage_ranks else []
    
        # Passage by Semantic
        passage_by_semantic = [
            self.passage_lookup[pid].text for pid in passage_ranks[:self.config.similarity_passage_top_k]
        ][::-1] if passage_ranks else []
    
        # Combine
        relevant_passages = list(dict.fromkeys(ranked_passages+passage_by_semantic))[::-1]

        # Graph Result
        graph_retrieval_result = self._build_entity_layer_results(sub_nodes, edge_obj_list, seed_ids)
        
        return {
                'relevant_graphs': graph_retrieval_result,
                'relevant_passages': relevant_passages,
        }

    def rank_edges(self, triplet_ranks, edges_set, sub_graph, ppr_scores):
        """
        Rank edges using Semantic Rank, PPR Rank, and Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        triplet_ranks : list of tuple
            Candidate ranked edges (triplets).
        edges_set : set of tuple
            All edges to consider.
        sub_graph : networkx.Graph
            Subgraph containing the edges.
        ppr_scores : dict
            Node -> score mapping for Personalized PageRank (PPR).

        Returns
        -------
        ranked_edges : list
            Top-k ranked edges according to RRF.
        """

        # --- Semantic Rank ---
        semantic_rank_list = [x for x in triplet_ranks if x in edges_set]
        semantic_rank = {
            edge: i + 1
            for i, edge in enumerate(semantic_rank_list)
            if edge not in semantic_rank_list[:i]  # keep first occurrence only
        }

        # --- PPR Rank ---
        edge_scores = {
            (e[0], e[1]): np.mean(
                [ppr_scores.get(e[0], 1e8), ppr_scores.get(e[1], 1e8)]
            )
            for e in sub_graph.edges
        }
        edge_ppr_rank = {
            edge: rank
            for rank, (edge, _) in enumerate(
                sorted(edge_scores.items(), key=lambda x: x[1], reverse=True),
                start=1
            )
        }

        # --- Edge RRF Rank ---
        scored = []
        for edge in edges_set:
            key1, key2 = (edge[0], edge[1]), (edge[1], edge[0])
            rank_ppr = edge_ppr_rank.get(key1, edge_ppr_rank.get(key2, 1e6))
            rank_sem = semantic_rank.get(key1, semantic_rank.get(key2, 1e6))
            rrf_score = (
                1 / (self.config.rrf_ppr_constant + rank_ppr)
                + 1 / (self.config.rrf_semantic_constant + rank_sem)
            )
            scored.append((edge, rrf_score))

        # --- Sort and Select ---
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_edges = [e for e, _ in scored[:self.config.prune_top_k]]

        return ranked_edges

    def rank_passages(self, sub_graph, seed_ids, passage_ranks):
        """
        Rank passages given a sub_graph, seeds, and semantic passage ranks.
        
        Args:
            sub_graph: networkx.Graph
            seed_ids: list of seed node ids (subset of sub_graph)
            passage_ranks: list of passage ids ranked semantically
        
        Returns:
            ranked_passage: list of passage ids (sorted by combined RRF score)
        """
    
        # --- Build reverse mapping ---
        filterd_passages = {p for n in sub_graph.nodes() 
                            for p in self.higra.node_id_to_passage_id_map[n]}
        passage_to_nodes = {
            pid: [n for n in sub_graph.nodes() if pid in self.higra.node_id_to_passage_id_map[n]]
            for pid in filterd_passages
        }
    
        # --- Build augmented graph (nodes + passage nodes) ---
        G_aug = nx.Graph()
        G_aug.add_nodes_from(sub_graph.nodes())
        G_aug.add_edges_from(sub_graph.edges())
        G_aug.add_nodes_from(filterd_passages)  # add passage nodes
        for pid, nodes in passage_to_nodes.items():
            for n in nodes:
                G_aug.add_edge(pid, n)
    
        # --- Personalization (only seed_ids) ---
        personalization = {n: 1.0 for n in seed_ids}
        personalization.update({n: 0.0 for n in G_aug.nodes() if n not in personalization})
    
        # --- Run PPR ---
        ppr_scores = nx.pagerank(
            G_aug,
            alpha=self.config.ppr_alpha,
            personalization=personalization,
        )
    
        # --- Extract passage PPR rank ---
        passage_scores = {p: ppr_scores.get(p, 0.0) for p in filterd_passages}
        passage_ppr_rank = {
            passage: rank
            for rank, (passage, _) in enumerate(
                sorted(passage_scores.items(), key=lambda x: x[1], reverse=True),
                start=1
            )
        }
    
        # --- Semantic Rank ---
        sematic_rank_list = [p for p in passage_ranks if p in filterd_passages]
        semantic_rank = {
            p: i + 1
            for i, p in enumerate(sematic_rank_list)
            if p not in sematic_rank_list[:i]  # keep first occurrence only
        }
    
        # --- RRF Rank ---
        scored = []
        for p in filterd_passages:
            rank_ppr = passage_ppr_rank.get(p, 1e6)
            rank_sem = semantic_rank.get(p, 1e6)
            rrf_score = (
                1 / (self.config.rrf_ppr_constant + rank_ppr)
                + 1 / (self.config.rrf_semantic_constant + rank_sem)
            )
            scored.append((p, rrf_score))
    
        # --- Sort and Select ---
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_passage = [p for p, _ in scored[:self.config.entity_passage_top_k]]
        ranked_passage = [self.passage_lookup[pid].text for pid in ranked_passage][::-1]
        
        return ranked_passage

    def retrieve_relevant_passages(self, question: str, passage_ranks: List[str]) -> Dict[str, Any]:
        """Retrieve top-k passages based on ranks."""
        passages = [
            self.passage_lookup[passage_id].text
            for passage_id in passage_ranks[:self.config.passage_top_k]
        ]
        return {
            "relevant_passages": passages[::-1]
        }

    def _get_subgraph(self, G, seeds, radius=1):
        """
        Return a subgraph of all nodes within `radius` hops
        from any of the given seed nodes.
        
        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        seeds : list or set
            List of seed node IDs.
        radius : int
            Number of hops to include (default=1).
        
        Returns
        -------
        subG : networkx.Graph
            Induced subgraph containing all nodes within radius hops.
        """
        if not isinstance(seeds, (list, set, tuple)):
            seeds = [seeds]
        
        nodes_to_include = set()
        
        for s in seeds:
            if s in G:  # make sure seed exists
                # all nodes within `radius` hops from s
                nodes_to_include.update(
                    nx.single_source_shortest_path_length(G, s, cutoff=radius).keys()
                )
        
        return G.subgraph(nodes_to_include).copy()

    
    def _get_seed_node_ids(self, entity_names: List[str]) -> Set[str]:
        """Resolve entity names to node IDs."""
        key_list = [
            key for key in self.higra.node_name_to_node_id_map
            for name in entity_names
            if unidecode(key).lower() == unidecode(name).lower()
        ]
        return {nid for key in key_list for nid in self.higra.node_name_to_node_id_map.get(key, [])}
        
    def _build_entity_layer_results(self, sub_nodes: Set[str], edge_objs: List[Any], seed_ids: Set[str]) -> List[Dict[str, Any]]:
        """Build results for entity layer, prioritizing seed nodes."""
        results = []
        for nid in sorted(sub_nodes, key=lambda nid: nid in seed_ids):
            node = self.node_lookup.get(nid)
            if not node:
                continue
            related_edges = [
                _map_edge_to_dict(e, self.node_lookup)
                for e in edge_objs
                if e.source_node_id == nid or e.target_node_id == nid
            ]
            results.append({'node': node.model_dump(), 'edges': related_edges})
        return transform_kg_to_llm_format(results)

    def _get_entity_passages(self, entity_names: List[str], passage_ranks: List[str]) -> List[str]:
        """Retrieve passages related to given entity names."""
        node_ids = self._get_seed_node_ids(entity_names)
        passage_ids = {pid for nid in node_ids for pid in self.higra.node_id_to_passage_id_map.get(nid, [])}
        ranked_passage_ids = sorted(
            (pid for pid in passage_ids if pid in passage_ranks),
            key=lambda pid: passage_ranks.index(pid)
        )[:self.config.entity_passage_top_k]
        
        return [f'{self.passage_lookup[pid].text}' for pid in ranked_passage_ids][::-1]

    def _get_entity_sentences(self, entity_names: List[str], sentence_ranks: List[str]) -> List[str]:
        """Retrieve sentences related to given entity names."""
        node_ids = self._get_seed_node_ids(entity_names)
        sentence_ids = {sid for nid in node_ids for sid in self.higra.node_id_to_sentence_id_map.get(nid, [])}
        ranked_sentence_ids = sorted(
            (sid for sid in sentence_ids if sid in sentence_ranks),
            key=lambda sid: sentence_ranks.index(sid)
        )[:self.config.entity_sentence_top_k]
        
        return [f'{self.sentence_lookup[sid].text}' for sid in ranked_sentence_ids][::-1]

    async def retrieve(self, question: str):
        top_related_entities, passage_ranks, sentence_ranks, triplet_ranks = self.create_ontology(question)
        named_entities, usage = await self.ner.run(question, top_related_entities)
        if named_entities['entity_name'] == []:
            context = self.retrieve_relevant_passages(question, passage_ranks)
        else:
            context = self.get_context(
                entity_names=named_entities['entity_name'],
                triplet_ranks=triplet_ranks,                
                passage_ranks=passage_ranks,
                sentence_ranks=None
            )

        context = json.dumps(context, indent=4, ensure_ascii=False)
        return context, usage

    async def retrieve_multiple(self, question_list: str):

        ontology_information_list = [self.create_ontology(question) for question in question_list]
            
        top_related_entities_list = [x[0] for x in ontology_information_list]
        
        named_entities_list, usage_list = await self.ner.run_multiple(question_list, top_related_entities_list)

        context_list = []
        for question, named_entities, ontology_information in zip(question_list, named_entities_list, ontology_information_list):
            if named_entities['entity_name'] == []:
                context = self.retrieve_relevant_passages(question, ontology_information[1])
            else:
                context = self.get_context(
                    entity_names=named_entities['entity_name'],
                    passage_ranks=ontology_information[1],                
                    triplet_ranks=ontology_information[3],
                    sentence_ranks=None
                )
            context_list.append(context)

        return context_list, usage_list

    async def retrieve_multiple_hybrid(self, question_list: str):
        q_embed_list = [
            self.embedder.encode(
                x, convert_to_numpy=True, truncation=True, show_progress_bar=False
            )
            for x in question_list
        ]

        hybrid_context_list = [
            self.passage_hybrid.retrieve(x, q_emb=q_emb) for x, q_emb in zip(question_list, q_embed_list)
        ]
        hybrid_context_list = [x[:20] for x in hybrid_context_list]
        return hybrid_context_list, []


def _map_edge_to_dict(edge: Any, nodes_by_id: Dict[str, Any]) -> Dict[str, str]:
    """Map an Edge model to a dictionary with labels."""
    def get_label(nid: str) -> str:
        node = nodes_by_id.get(nid)
        return f"{node.name} ({', '.join(node.type)})" if node else nid

    return {
        'source_name': get_label(edge.source_node_id),
        'relationship_name': edge.relationship_name,
        'target_name': get_label(edge.target_node_id)
    }


def transform_kg_to_llm_format(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform knowledge graph data to LLM-friendly format with separate node and edge lists.
    Deduplicates edges by normalizing direction.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    seen_edges: Set[Tuple[str, str, str]] = set()

    for entry in data:
        # Process node
        node = entry['node']
        node_dict = {
            "entity": node["name"],
            "types": node["type"]
        }
        if node.get("description"):
            node_dict[f"description"] = node["description"][:200] + "..."
        if node.get("aliases"):
            node_dict[f"aliases"] = node["aliases"]
            nodes.append(node_dict)

        # Process edges
        for edge in entry.get("edges", []):
            # Normalize edge direction to avoid duplicates
            source, relation, target = (
                edge["source_name"],
                edge["relationship_name"],
                edge["target_name"]
            )
            # Create a tuple with sorted source/target to ensure (A,B) and (B,A) are treated as the same
            edge_key = tuple(sorted([source, target]) + [relation])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "source": source,
                    "relationship": relation,
                    "target": target
                })
    # return edges
    return {
        "nodes_alias_information": nodes,
        "edges_information": edges
    }