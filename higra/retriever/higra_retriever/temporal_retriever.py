import json
import math
import networkx as nx
import numpy as np
import re
import asyncio

from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from collections import defaultdict

from higra_agent.retriever.higra_retriever.higra_retriever import HiGraRetriever, RetrieverConfig
from higra_agent.retriever.higra_retriever.higra_schema import HierarchicalKnowledgeGraph, Edge, TemporalInterval
from higra_agent.retriever.higra_retriever.temporal_utils import TemporalNormalizer, TemporalValidator
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.retriever.higra_retriever.query_classifier import QueryIntent, QueryIntentClassifier
from higra_agent.model_cache import ModelCache

# Define TemporalRetrieverConfig here (or import if defined elsewhere)
class TemporalRetrieverConfig(RetrieverConfig):
    """Configuration for TemporalHiGraRetriever."""
    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict)
        cfg = config_dict or {}
        
        # Stage 1: Edge Weighting Balance
        self.ppr_edge_weight_alpha = float(cfg.get("ppr_edge_weight_alpha", 0.5))
        self.time_decay_sigma = float(cfg.get("time_decay_sigma", 5.0))
        
        # Stage 2: Passage Ranking
        self.entity_passage_boost_beta = float(cfg.get("entity_passage_boost_beta", 2.0))
        
        # Fusion: Adaptive RRF
        self.rrf_time_constant = int(cfg.get("rrf_time_constant", 10))
        
        # Weights for dynamic calculation (can be overridden by config)
        default_intent_weights = {
            "explicit_temporal": {"ppr": 2.0, "semantic": 0.5, "time": 2.0},
            "implicit_temporal": {"ppr": 1.5, "semantic": 1.0, "time": 1.0},
            "ordering": {"ppr": 1.5, "semantic": 0.5, "time": 2.5},
            "atemporal": {"ppr": 0.5, "semantic": 2.0, "time": 0.1}
        }
        # Allow partial updates to intent weights
        self.intent_weights = default_intent_weights.copy()
        if "intent_weights" in cfg:
            for k, v in cfg["intent_weights"].items():
                if k in self.intent_weights:
                    self.intent_weights[k].update(v)

        self.default_fusion_weights = cfg.get("fusion_weights", {
            "ppr": 1.0, 
            "semantic": 1.0, 
            "time": 1.0
        })
        
        # Check retrieval section first, then root
        retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval"), dict) else {}
        self.snapshot_strictness = retrieval_cfg.get("snapshot_strictness", cfg.get("snapshot_strictness", True))

class TemporalHiGraRetriever(HiGraRetriever):
    """
    Temporal-aware Knowledge Graph Retriever (V2).
    Implements Time-Aware Dual-Stage PPR and Adaptive Balanced RRF.
    """
    def __init__(
        self,
        higra: HierarchicalKnowledgeGraph,
        embedder,
        api_key: str,
        model_name: str,
        config_dict: Optional[Dict],
        embedding_save_path: Optional[str] = None
    ):
        super().__init__(
            higra=higra,
            embedder=embedder,
            api_key=api_key,
            model_name=model_name,
            config_dict=config_dict,
            embedding_save_path=embedding_save_path,
        )
        
        # Initialize Temporal Config
        self.config = TemporalRetrieverConfig(config_dict)
        
        # Temporal core components
        self.temporal_normalizer = TemporalNormalizer()
        self.temporal_validator = TemporalValidator()

        # LLM based temporal assistance
        self.llm_client = LLMAsyncClient(api_key=api_key, model_name=model_name)
        self.intent_classifier = QueryIntentClassifier(self.llm_client)

        self.available_context_time = self._extract_available_context_time()
        
        # Precompute IDF for Stage 2 (Entity-Passage Propagation)
        self.node_idf = self._precompute_node_idf()

    def _precompute_node_idf(self) -> Dict[str, float]:
        """Precompute Inverse Document Frequency for nodes across passages."""
        total_passages = len(self.higra.passage_layer.passages)
        idf_map = {}
        if total_passages == 0:
            return idf_map
            
        for node_id, passage_ids in self.higra.node_id_to_passage_id_map.items():
            # df = document frequency (number of passages containing the node)
            df = len(set(passage_ids)) 
            if df > 0:
                idf_map[node_id] = math.log(total_passages / df)
            else:
                idf_map[node_id] = 0.0
        return idf_map

    def _extract_available_context_time(self) -> List[str]:
        timestamps = set()
        for node in self.higra.entity_layer.nodes:
            if node.valid_time:
                timestamps.add(node.valid_time.end if node.valid_time.end else node.valid_time.start)
        for passage in self.higra.passage_layer.passages:
            if passage.primary_time:
                timestamps.add(passage.primary_time.end if passage.primary_time.end else passage.primary_time.start)
        return sorted(list(timestamps))

    def _parse_timestamp(self, timestamp: str) -> datetime:
        try:
            if len(timestamp) == 4: return datetime(int(timestamp), 1, 1)
            return datetime.fromisoformat(timestamp)
        except:
            return datetime(1900, 1, 1)

    def _parse_entities(self, entities_list: List[str]):
        parsed_results = []

        for row in entities_list:
            entry = {}
            
            name_match = re.search(r'Name:\s*\$\$(.*?)\$\$', row)
            entry['name'] = name_match.group(1).strip() if name_match else None
            
            type_match = re.search(r'Type:\s*\[(.*?)\]', row)
            if type_match:
                entry['type'] = [t.strip() for t in type_match.group(1).split(',')]
            else:
                entry['type'] = []

            alias_match = re.search(r'Alias:\s*\[(.*?)\]', row)
            if alias_match:
                entry['alias'] = [a.strip() for a in alias_match.group(1).split(',')]
            else:
                entry['alias'] = []
                
            parsed_results.append(entry)
        
        return parsed_results
    
    def _is_valid_at_query_time(self, valid_time: TemporalInterval, query_timestamp: TemporalInterval) -> bool:
        if not valid_time or not valid_time.start: return True
        query_start = self._parse_timestamp(query_timestamp.start)
        query_end = self._parse_timestamp(query_timestamp.end) if query_timestamp.end else query_start
        valid_start = self._parse_timestamp(valid_time.start)
        valid_end = self._parse_timestamp(valid_time.end) if valid_time.end else datetime(9999, 12, 31)
        return valid_start <= query_end and query_start <= valid_end
        
    def filter_nodes_by_valid_time(self, node_ids: List[str], query_timestamp: TemporalInterval) -> List[str]:
        """Filter nodes based on their availability in specific snapshots or valid time intervals."""
        if not query_timestamp:
            return node_ids

        try:   
            filtered = []
            for nid in node_ids:
                node = self.node_lookup.get(nid)
                if not node:
                    continue
                
                # If node has valid_time, check overlap
                if node.valid_time and node.valid_time.start:
                    if self._is_valid_at_query_time(node.valid_time, query_timestamp):
                        filtered.append(nid)
                else:
                    filtered.append(nid)
           
            return filtered if filtered else node_ids
    
        except Exception as e:
            return node_ids

    def filter_graph_snapshot(self, subgraph: nx.Graph, query_timestamp: TemporalInterval) -> nx.Graph:
        """Remove nodes belonging to future snapshots from the subgraph."""
        if not query_timestamp or not self.config.snapshot_strictness:
            return subgraph

        # Get all nodes and filter them using the robust logic
        all_nodes = list(subgraph.nodes())
        valid_nodes = set(self.filter_nodes_by_valid_time(all_nodes, query_timestamp))
        
        # Identify nodes to remove
        nodes_to_remove = [n for n in all_nodes if n not in valid_nodes]

        if nodes_to_remove:
            snapshot = subgraph.copy()
            snapshot.remove_nodes_from(nodes_to_remove)
            return snapshot
        
        return subgraph

    def filter_passages_by_valid_time(self, passage_ids: List[str], query_timestamp: TemporalInterval) -> List[str]:
        """Filter passages based on their temporal validity."""
        if not query_timestamp:
            return passage_ids
        
        filtered = []
        for pid in passage_ids:
            passage = self.passage_lookup.get(pid)
            if not passage:
                continue
            
            # If passage has primary_time, check overlap
            if passage.primary_time and passage.primary_time.start:
                if self._is_valid_at_query_time(passage.primary_time, query_timestamp):
                    filtered.append(pid)
            else:
                filtered.append(pid)
        
        return filtered if filtered else passage_ids

    # -------------------------------------------------------------------------
    # STAGE 1: Entity-Entity Propagation (Enhanced Edge Weighting)
    # -------------------------------------------------------------------------

    def calculate_temporal_score(
        self,
        valid_time: 'TemporalInterval', 
        query_time_float: float, 
        sigma: float
    ) -> float:
        """Gaussian decay for temporal relevance."""
        if not valid_time or not valid_time.start:
            return 0.5 # Neutral

        try:
            start_year = float(valid_time.start[:4])
            end_year = float(valid_time.end[:4]) if valid_time.end else start_year
            node_year = (start_year + end_year) / 2
        except:
            node_year = 2025.0
            
        dist = abs(query_time_float - node_year)
        return math.exp(-(dist**2) / (2 * sigma**2))

    def apply_temporal_weights(
        self, 
        subgraph: nx.Graph, 
        query_timestamp: Optional[TemporalInterval],
        query_embedding: Optional[np.ndarray] = None
    ):
        """
        V2 Implementation: Combined Temporal and Semantic Edge Weighting.
        W = alpha * S_time + (1 - alpha) * S_sem
        """
        alpha = self.config.ppr_edge_weight_alpha
        sigma = self.config.time_decay_sigma

        # Prepare query year for temporal calc
        query_year_float = 2025.0
        if query_timestamp:
            ts_str = query_timestamp.end if query_timestamp.end else query_timestamp.start
            try:
                query_year_float = float(ts_str[:4])
            except:
                pass

        # Pre-calculate node embeddings for the subgraph if semantic weighting is enabled
        # Optimization: Only embed nodes in the subgraph (small #)
        node_embeddings = {}
        if query_embedding is not None and alpha < 1.0:
            # We use the target node's name for semantic similarity of the edge
            # (Assuming edge reflects transition to that concept)
            for node_id in subgraph.nodes():
                if node_id not in node_embeddings:
                    node = self.node_lookup.get(node_id)
                    if node:
                        # Quick embedding (cached if possible)
                        # Note: self.embedder.encode usually handles caching internally if configured, 
                        # or we rely on ModelCache
                        node_embeddings[node_id] = self.embedder.encode(
                            node.name, convert_to_numpy=True, show_progress_bar=False
                        )

        for u, v, data in subgraph.edges(data=True):
            edge_id = data.get("edge_id")
            edge_obj = self.edge_lookup.get(edge_id)
            
            # 1. Temporal Score (S_time)
            s_time = 0.5
            if query_timestamp and edge_obj and edge_obj.valid_time:
                s_time = self.calculate_temporal_score(edge_obj.valid_time, query_year_float, sigma)
            
            # 2. Semantic Score (S_sem)
            s_sem = 0.5
            if query_embedding is not None and alpha < 1.0:
                # Use target node (v) embedding
                # Since graph is undirected in NetworkX but directed in logic, 
                # this is an approximation. PPR propagates in all directions.
                # Ideally check edge direction or average both nodes.
                emb_u = node_embeddings.get(u)
                emb_v = node_embeddings.get(v)
                
                score_u = np.dot(query_embedding, emb_u) if emb_u is not None else 0.0
                score_v = np.dot(query_embedding, emb_v) if emb_v is not None else 0.0
                
                # Normalize approx (assuming embeddings are normalized, dot product is cosine sim)
                # If not normalized, should divide by norms. SentenceTransformers usually normalize.
                s_sem = (score_u + score_v) / 2.0
                # Clip to [0,1] just in case
                s_sem = max(0.0, min(1.0, s_sem))

            # 3. Combine
            weight = alpha * s_time + (1 - alpha) * s_sem
            subgraph[u][v]["weight"] = weight

    # -------------------------------------------------------------------------
    # STAGE 2: Entity-Passage Propagation (TF-IDF + Soft Temporal Boost)
    # -------------------------------------------------------------------------

    def rank_passages(self, sub_graph, seed_ids, passage_ranks, query_timestamp: Optional[TemporalInterval] = None):
        """
        V2 Implementation: 
        - Use TF-IDF for Entity-Passage edge weights.
        - Use Soft Temporal Boosting instead of Hard Filtering.
        """
        # 1. Identify Candidate Passages
        # Instead of filtering purely by time, we consider passages linked to subgraph nodes
        # AND passages from semantic search (passage_ranks)
        
        # Get all passages linked to nodes in subgraph
        graph_passage_ids = {
            pid 
            for n in sub_graph.nodes() 
            for pid in self.higra.node_id_to_passage_id_map.get(n, [])
        }
        
        # Combine with semantic passage ranks (to ensure we don't miss semantically relevant ones)
        candidate_passages = graph_passage_ids.union(set(passage_ranks))
        
        # 2. Build Augmented Graph
        G_aug = nx.Graph()
        G_aug.add_nodes_from(sub_graph.nodes())
        G_aug.add_edges_from(sub_graph.edges(data=True)) # Keep weights from Stage 1
        G_aug.add_nodes_from(candidate_passages)
        
        # 3. Add Weighted Edges (Entity <-> Passage)
        beta = self.config.entity_passage_boost_beta
        
        for pid in candidate_passages:
            passage = self.passage_lookup.get(pid)
            if not passage: continue
            
            # Calculate Temporal Boost
            time_boost = 1.0
            if query_timestamp and passage.primary_time:
                 if self._is_valid_at_query_time(passage.primary_time, query_timestamp):
                     time_boost = 1.0 + beta
            
            # Add edges for nodes in this passage that are also in the subgraph
            for nid in passage.node_map:
                if nid in sub_graph:
                    # TF-IDF Weight
                    # TF = 1.0 (Binary presence for now), IDF = precomputed
                    idf = self.node_idf.get(nid, 1.0)
                    weight = idf * time_boost
                    
                    G_aug.add_edge(nid, pid, weight=weight)
        
        # 4. Run PPR on Augmented Graph
        personalization = {n: 1.0 for n in seed_ids}
        personalization.update({n: 0.0 for n in G_aug.nodes() if n not in personalization})
        
        try:
            ppr_scores = nx.pagerank(
                G_aug,
                alpha=self.config.ppr_alpha,
                personalization=personalization,
                weight="weight"
            )
        except:
             ppr_scores = {n: 0.0 for n in G_aug.nodes()}

        # 5. Extract Passage Ranks
        passage_scores = {p: ppr_scores.get(p, 0.0) for p in candidate_passages}
        passage_ppr_rank = {
            p: rank for rank, (p, _) in enumerate(
                sorted(passage_scores.items(), key=lambda x: x[1], reverse=True), start=1
            )
        }
        
        # 6. Fuse with Semantic Ranks (Standard RRF)
        # We use standard RRF here for Passage fusion, saving the 3-way for Edge fusion
        semantic_rank_map = {p: i+1 for i, p in enumerate(passage_ranks)}
        
        scored = []
        for p in candidate_passages:
            r_ppr = passage_ppr_rank.get(p, 1e6)
            r_sem = semantic_rank_map.get(p, 1e6)
            
            score = 1.0 / (10 + r_ppr) + 1.0 / (10 + r_sem)
            scored.append((p, score))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [p for p, _ in scored[:self.config.entity_passage_top_k]]
        
        return [self.passage_lookup[pid].text for pid in ranked_ids][::-1]

    # -------------------------------------------------------------------------
    # STAGE 3: Adaptive 3-Way Fusion
    # -------------------------------------------------------------------------

    def rank_edges_adaptive(
        self, 
        triplet_ranks: List[Tuple[str, str]], 
        edges_set: Set[Tuple[str, str]], 
        sub_graph: nx.Graph, 
        ppr_scores: Dict[str, float], 
        intent: QueryIntent,
        query_timestamp: Optional[TemporalInterval] = None
    ) -> List[Tuple[str, str]]:
        """
        V2 Implementation: 3-Way RRF (PPR, Semantic, TimeRank) with Dynamic Weights.
        """
        # 1. Generate Rankings
        
        # A. Semantic Rank
        semantic_rank_list = []
        for u, v in triplet_ranks:
            if (u, v) in edges_set: semantic_rank_list.append((u, v))
            elif (v, u) in edges_set: semantic_rank_list.append((v, u))
        semantic_rank = {edge: i + 1 for i, edge in enumerate(semantic_rank_list)}
        
        # B. PPR Rank
        edge_ppr_scores = {
            (u, v): np.mean([ppr_scores.get(u, 0.0), ppr_scores.get(v, 0.0)])
            for u, v in sub_graph.edges()
        }
        ppr_rank = {
            edge: rank for rank, (edge, _) in enumerate(
                sorted(edge_ppr_scores.items(), key=lambda x: x[1], reverse=True), start=1
            )
        }
        
        # C. TimeRank (Explicit temporal proximity of the edge)
        edge_time_scores = {}
        query_year = 2025.0
        if query_timestamp:
             try:
                query_year = float(query_timestamp.start[:4])
             except: pass
             
        for u, v, data in sub_graph.edges(data=True):
             edge_id = data.get("edge_id")
             edge_obj = self.edge_lookup.get(edge_id)
             score = 0.5
             if edge_obj and edge_obj.valid_time:
                 score = self.calculate_temporal_score(edge_obj.valid_time, query_year, sigma=self.config.time_decay_sigma)
             edge_time_scores[(u, v)] = score
             
        time_rank = {
            edge: rank for rank, (edge, _) in enumerate(
                sorted(edge_time_scores.items(), key=lambda x: x[1], reverse=True), start=1
            )
        }

        # 2. Calculate Dynamic Weights
        weights = self._calculate_dynamic_weights(intent, ppr_rank, time_rank)
        w_ppr, w_sem, w_time = weights['ppr'], weights['semantic'], weights['time']
        
        k = 60 # Base RRF constant
        
        # 3. Fusion
        scored = []
        for u, v in edges_set:
            # Handle undirected matching
            key = (u, v) if (u, v) in ppr_rank else (v, u)
            
            r_ppr = ppr_rank.get(key, 1e6)
            r_sem = semantic_rank.get(key, 1e6)
            r_time = time_rank.get(key, 1e6)
            
            rrf_score = (
                w_ppr * (1.0 / (k + r_ppr)) +
                w_sem * (1.0 / (k + r_sem)) +
                w_time * (1.0 / (k + r_time))
            )
            scored.append(((u, v), rrf_score))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:self.config.prune_top_k]]

    def _calculate_dynamic_weights(
        self, 
        intent: QueryIntent, 
        ppr_rank: Dict, 
        time_rank: Dict
    ) -> Dict[str, float]:
        """Adjust weights based on intent and signal overlap."""
        
        # Use weights from config (initialized with defaults)
        intent_key = intent.value
        if intent_key in self.config.intent_weights:
            w = self.config.intent_weights[intent_key].copy()
        else:
            w = self.config.intent_weights["atemporal"].copy()
            
        # Calculate Overlap (Jaccard of top 10)
        top_ppr = set(list(ppr_rank.keys())[:10])
        top_time = set(list(time_rank.keys())[:10])
        
        if not top_ppr or not top_time:
            overlap = 0.0
        else:
            intersection = len(top_ppr.intersection(top_time))
            union = len(top_ppr.union(top_time))
            overlap = intersection / union if union > 0 else 0.0
            
        # If high overlap (> 0.5), PPR is already capturing time well. 
        # Reduce explicit time rank weight to avoid double counting.
        if overlap > 0.5:
            w["time"] *= 0.5
            
        return w

    async def extract_query_timestamp(self, question: str) -> TemporalInterval:
        temporal_exprs = self.temporal_normalizer.extract_temporal_expressions(question)

        if temporal_exprs:
            expr = temporal_exprs[0]
            if 'start' in expr:
                return TemporalInterval(start=expr['start'], end=expr.get('end'))
        else:
            return self.temporal_normalizer.expand_to_interval("2025")

    def retrieve_relevant_passages(self, question: str, passage_ranks: List[str]) -> Dict[str, Any]:
        """Retrieve top-k passages based on ranks."""
        top_p = passage_ranks[:self.config.passage_top_k]
        passages = [
            self.passage_lookup[passage_id].text
            for passage_id in top_p if passage_id in self.passage_lookup
        ]
            
        return {
            "relevant_passages": passages[::-1]
        }

    # -------------------------------------------------------------------------
    # Main Pipeline Override
    # -------------------------------------------------------------------------

    async def retrieve_temporal(self, question: str) -> Dict[str, Any]:
        # 1. Extract query timestamp & intent
        query_timestamp = await self.extract_query_timestamp(question)
        intent = await self.intent_classifier.classify(question)

        # Get query embedding for semantic weighting
        query_embedding = self.embedder.encode(
            question, convert_to_numpy=True, show_progress_bar=False
        )
  
        # 2. Standard ontology creation
        top_entities, passage_ranks, sentence_ranks, triplet_ranks = self.create_ontology(question)
        
        # 3. Extract named entities
        named_entities, usage = await self.ner.run(question, top_entities)
        entities = self._parse_entities(top_entities)
        exact_match_entities = named_entities.get("entity_name", []) + [e["name"] for e in entities if e["name"] in question]
        match_entities = list(dict.fromkeys(exact_match_entities))
        # If no entities, use passage-based retrieval (with temporal boost logic if we had it there)
        if not match_entities:
            # Fallback to passage retrieval (using semantic ranks only for now)
            # Could enhance to use rank_passages with dummy graph
            context = self.retrieve_relevant_passages(question, passage_ranks)
            context["ner_entities"] = []
        else:
            # 4. Get seed nodes from entities
            seed_node_ids = self._get_seed_node_ids(match_entities)
            
            # Note: In V2 we DO NOT Strictly filter nodes here anymore (Soft filtering pref).
            # But for large graphs, we might still want to prune completely irrelevant eras 
            # if strictness is ON.
            if self.config.snapshot_strictness and query_timestamp:
                seed_node_ids = set(self.filter_nodes_by_valid_time(list(seed_node_ids), query_timestamp))
                if not seed_node_ids:
                     # Fallback if strict filtering killed everything
                     context = self.retrieve_relevant_passages(question, passage_ranks)
                     return {
                        "context": json.dumps(context, indent=4, ensure_ascii=False),
                        "usage": usage,
                        "temporal_metadata": {"query_timestamp": query_timestamp.to_dict() if query_timestamp else None}
                    }

            # 5. Get Context (Calls V2 methods)
            context = self.get_context(
                entity_names=match_entities, # Passed for compatibility, actual seeds used inside
                triplet_ranks=triplet_ranks,
                sentence_ranks=sentence_ranks,
                passage_ranks=passage_ranks,
                intent=intent, 
                query_timestamp=query_timestamp,
                query_embedding=query_embedding
            )
            
        context["ner_entities"] = named_entities.get('entity_name', [])
        context["query_timestamp"] = query_timestamp.to_dict() if query_timestamp else None
        
        return {
            "context": json.dumps(context, indent=4, ensure_ascii=False),
            "usage": usage,
            "temporal_metadata": {
                "query_timestamp": query_timestamp.to_dict() if query_timestamp else None,
            }
        }

    def get_context(
        self,
        entity_names: List[str],
        triplet_ranks: List[Tuple[str, str]],
        sentence_ranks: List[str],
        passage_ranks: List[str],
        intent: QueryIntent = QueryIntent.ATEMPORAL,
        query_timestamp: Optional[TemporalInterval] = None,
        query_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        V2 context retrieval orchestration.
        """
        seed_ids = self._get_seed_node_ids(entity_names)
        
        if self.config.snapshot_strictness and query_timestamp:
            seed_ids = set(self.filter_nodes_by_valid_time(list(seed_ids), query_timestamp))

        # 1. Get Subgraph
        sub_graph = self._get_subgraph(self.graph, seed_ids, radius=1)
        
        # 2. Filter Snapshot (Optional strict filtering)
        if self.config.snapshot_strictness:
            sub_graph = self.filter_graph_snapshot(sub_graph, query_timestamp)

        # 3. Apply Stage 1 Weights (Temporal + Semantic)
        self.apply_temporal_weights(sub_graph, query_timestamp, query_embedding)
        
        # 4. Run PPR
        personalization = {nid: 1.0 if nid in seed_ids else 0 for nid in sub_graph.nodes}
        try:
            ppr_scores = nx.pagerank(
                sub_graph, 
                alpha=self.config.ppr_alpha, 
                personalization=personalization, 
                weight="weight"
            )
        except:
            ppr_scores = {n: 1.0 for n in sub_graph.nodes()}

        # 5. Adaptive 3-Way Fusion Ranking
        ranked_edges = self.rank_edges_adaptive(
            triplet_ranks, 
            set(sub_graph.edges()), 
            sub_graph, 
            ppr_scores, 
            intent,
            query_timestamp
        )
        
        # 6. Retrieve Edge Objects
        edge_objs = []
        for u, v in ranked_edges:
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data and "edge_id" in edge_data:
                edge_objs.append(self.edge_lookup[edge_data["edge_id"]])

        # 7. Collect Nodes
        sub_nodes = {e[0] for e in ranked_edges} | {e[1] for e in ranked_edges} | seed_ids

        # 8. Rank Passages (Stage 2)
        ranked_passages = self.rank_passages(
            sub_graph, 
            seed_ids, 
            passage_ranks, 
            query_timestamp
        )
        
        # Fallback/Additional semantic passages
        passage_by_semantic = [
            self.passage_lookup[pid].text 
            for pid in passage_ranks[:self.config.similarity_passage_top_k] 
            if pid in self.passage_lookup
        ][::-1]

        return {
            "relevant_graphs": self._build_entity_layer_results(sub_nodes, edge_objs, seed_ids),
            "relevant_passages": list(dict.fromkeys(ranked_passages + passage_by_semantic))[::-1],
            "relevant_sentences": sentence_ranks,
            "seed_nodes": list(seed_ids)
        }
