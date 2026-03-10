import json
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
import math

from higra_agent.retriever.higra_retriever.higra_retriever import HiGraRetriever
from higra_agent.retriever.higra_retriever.higra_schema import HierarchicalKnowledgeGraph, Edge, TemporalInterval
from higra_agent.retriever.higra_retriever.temporal_utils import TemporalNormalizer, TemporalValidator
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.retriever.higra_retriever.query_classifier import QueryIntent, QueryIntentClassifier


class TemporalHiGraRetriever(HiGraRetriever):
    """
    Temporal-aware Knowledge Graph Retriever.
    Inherits from HiGraRetriever and adds temporal snapshot filtering and reasoning.
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
        
        # Temporal core components
        self.temporal_normalizer = TemporalNormalizer()
        self.temporal_validator = TemporalValidator()

        # LLM based temporal assistance
        self.llm_client = LLMAsyncClient(api_key=api_key, model_name=model_name)
        self.intent_classifier = QueryIntentClassifier(self.llm_client)

        self.available_context_time = self._extract_available_context_time()

        # Configuration
        cfg = config_dict or {}
        self.snapshot_strictness = cfg.get("snapshot_strictness", True)
        self.time_decay_sigma = float(cfg.get("time_decay_sigma", 5.0))

    def calculate_temporal_weight(
        self,
        node_time: 'TemporalInterval', 
        query_time: Union[float, 'TemporalInterval'], 
        sigma: float = 5.0
    ) -> float:
        """
        Calculate weight based on temporal proximity using Gaussian decay.
        
        Args:
            node_time: The valid_time of a node or edge.
            query_time: The timestamp from the user's query (float year or TemporalInterval).
            sigma: Standard deviation (years). Controls how fast weight decays.
                e.g., sigma=5 means ~60% weight at 5 years diff.
        
        Returns:
            float: Weight between 0.0 and 1.0
        """
        if not node_time or not node_time.start:
            return 0.5 # Default weight for atemporal nodes (neutral)

        # Convert query_time to float year
        if isinstance(query_time, TemporalInterval):
            query_year_str = query_time.end if query_time.end else query_time.start
            try:
                query_year = float(query_year_str[:4]) if query_year_str else 2025.0
            except:
                query_year = 2025.0
        else:
            query_year = float(query_time)
        
        # Convert node_time to float year (use midpoint or start)
        try:
            node_start_year = float(node_time.start[:4]) if node_time.start else 2025.0
            if node_time.end:
                node_end_year = float(node_time.end[:4])
                node_year = (node_start_year + node_end_year) / 2  # Use midpoint
            else:
                node_year = node_start_year
        except:
            node_year = 2025.0
        
        # Calculate distance in years
        dist = abs(query_year - node_year)
        
        # Gaussian Decay: exp(-dist^2 / (2*sigma^2))
        weight = math.exp(-(dist**2) / (2 * sigma**2))
        
        return weight

    def _extract_available_context_time(self) -> List[str]:
        """Extract all unique context timestamps (snapshot years) from the graph."""
        timestamps = set()
        for node in self.higra.entity_layer.nodes:
            if node.valid_time:
                timestamps.add(node.valid_time.end if node.valid_time.end else node.valid_time.start)
        
        # From passages
        for passage in self.higra.passage_layer.passages:
            if passage.primary_time:
                timestamps.add(passage.primary_time.end if passage.primary_time.end else passage.primary_time.start)
                
        return sorted(list(timestamps))

    def _parse_timestamp(self, timestamp: str) -> datetime:
        try:
            if len(timestamp) == 4:  # year-only 
                return datetime(int(timestamp), 1, 1)
            return datetime.fromisoformat(timestamp) # ISO format
        except:
            return datetime(1900, 1, 1)
        
    def _is_valid_at_query_time(self, valid_time: TemporalInterval, query_timestamp: TemporalInterval) -> bool:
        """Check if valid_time interval overlaps with query_timestamp interval."""
        if not valid_time or not valid_time.start:
            return True  # No temporal constraint, always valid
        
        query_start = self._parse_timestamp(query_timestamp.start)
        query_end   = self._parse_timestamp(query_timestamp.end) if query_timestamp.end else query_start
        
        valid_start = self._parse_timestamp(valid_time.start)
        # If no end time, assume it's still valid (open-ended)
        if valid_time.end:
            valid_end = self._parse_timestamp(valid_time.end)
        else:
            # Open-ended validity - valid from start onwards
            valid_end = datetime(9999, 12, 31)
        
        # Check if intervals overlap: valid_start <= query_end AND query_start <= valid_end
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
        if not query_timestamp or not self.snapshot_strictness:
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

    def apply_temporal_weights(self, subgraph: nx.Graph, query_timestamp: Optional[TemporalInterval]):
        """Calculate and assign edge weights based on temporal distance to query."""
        if not query_timestamp:
            for u, v in subgraph.edges(): subgraph[u][v]["weight"] = 1.0
            return

        try:
            # Extract year from TemporalInterval
            timestamp_str = query_timestamp.end if query_timestamp.end else query_timestamp.start
            query_year_float = float(timestamp_str[:4]) if timestamp_str else 2025.0
        except:
            # Fallback if timestamp parsing fails
            for u, v in subgraph.edges(): subgraph[u][v]["weight"] = 1.0
            return

        for u, v, data in subgraph.edges(data=True):
            edge_id = data.get("edge_id")
            edge_obj: Optional[Edge] = self.edge_lookup.get(edge_id) #type: ignore
            
            t_weight = 0.5 # Neutral weight default
            
            if edge_obj and edge_obj.valid_time:
                t_weight = self.calculate_temporal_weight(
                    edge_obj.valid_time, query_year_float, sigma=self.time_decay_sigma
                )
            
            # Assign weight to graph edge
            subgraph[u][v]["weight"] = t_weight



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
            # If passage has context_timestamp, check if query time is at or after
            # elif passage.context_timestamp:
            #     query_start = self._parse_timestamp(query_timestamp.start)
            #     context_time = self._parse_timestamp(passage.context_timestamp)
            #     if query_start >= context_time:
            #         filtered.append(pid)
            # If no temporal info, include it
            else:
                filtered.append(pid)
        
        return filtered if filtered else passage_ids

    def get_context(
        self,
        entity_names: List[str],
        triplet_ranks: List[Tuple[str, str]],
        sentence_ranks: List[str],
        passage_ranks: List[str],
        intent: QueryIntent = QueryIntent.ATEMPORAL,
        query_timestamp: Optional[TemporalInterval] = None,
    ) -> Dict[str, Any]:
        """Override parent method to include temporal filtering and weighting."""
        seed_ids = self._get_seed_node_ids(entity_names)
        
        if query_timestamp:
            seed_ids = set(self.filter_nodes_by_valid_time(list(seed_ids), query_timestamp))

        sub_graph = self._get_subgraph(self.graph, seed_ids, radius=1)
        
        # Temporal Pipeline
        sub_graph = self.filter_graph_snapshot(sub_graph, query_timestamp) #type: ignore
        self.apply_temporal_weights(sub_graph, query_timestamp) #type: ignore
        
        # Weighted PPR
        personalization = {nid: 1.0 if nid in seed_ids else 0 for nid in sub_graph.nodes}
        try:
            ppr_scores = nx.pagerank(sub_graph, alpha=self.config.ppr_alpha, personalization=personalization, weight="weight")
        except:
            ppr_scores = {n: 1.0 for n in sub_graph.nodes()}

        # Adaptive Ranking
        ranked_edges = self.rank_edges_adaptive(triplet_ranks, set(sub_graph.edges()), sub_graph, ppr_scores, intent)
        
        edge_objs = []
        for x in ranked_edges:
            # Safe edge lookup with bi-directional check fallback (though ranked_edges usually preserves structure)
            u, v = x
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data and "edge_id" in edge_data:
                edge_objs.append(self.edge_lookup[edge_data["edge_id"]])

        sub_nodes = {e[0] for e in ranked_edges} | {e[1] for e in ranked_edges} | seed_ids

        # Filter passages temporally
        if query_timestamp:
            passage_ranks = self.filter_passages_by_valid_time(passage_ranks, query_timestamp) #type: ignore

        ranked_passages = self.rank_passages(sub_graph, seed_ids, passage_ranks)
        passage_by_semantic = [self.passage_lookup[pid].text for pid in passage_ranks[:self.config.similarity_passage_top_k]][::-1]

        return {
            "relevant_graphs": self._build_entity_layer_results(sub_nodes, edge_objs, seed_ids),
            "relevant_passages": list(dict.fromkeys(ranked_passages + passage_by_semantic))[::-1],
            "seed_nodes": list(seed_ids)
        }

    def rank_edges_adaptive(self, triplet_ranks: List[Tuple[str, str]], edges_set: Set[Tuple[str, str]], sub_graph: nx.Graph, ppr_scores: Dict[str, float], intent: QueryIntent) -> List[Tuple[str, str]]:
        """
        Adjust RRF constants to favor temporal weight (PPR) or semantic match based on intent.
        Handles undirected edge matching properly.
        """
        k_ppr, k_sem = self.config.rrf_ppr_constant, self.config.rrf_semantic_constant
        
        if intent in [QueryIntent.EXPLICIT_TEMPORAL, QueryIntent.ORDERING]:
            k_ppr, k_sem = 1, 60 # Prioritize PPR (temporal weights)
        elif intent == QueryIntent.ATEMPORAL:
            k_ppr, k_sem = 60, 1 # Prioritize semantic match

        # --- Semantic Ranks ---
        # Only rank edges that actually exist in the filtered subgraph
        # Important: Check both (u, v) and (v, u) directions against edges_set
        semantic_rank_list = []
        for u, v in triplet_ranks:
            if (u, v) in edges_set:
                semantic_rank_list.append((u, v))
            elif (v, u) in edges_set:
                semantic_rank_list.append((v, u))
        
        semantic_rank = {edge: i + 1 for i, edge in enumerate(semantic_rank_list)}

        # --- PPR Ranks ---
        # Calculate mean PPR score for each edge in the subgraph
        edge_scores = {}
        for u, v in sub_graph.edges():
            score = np.mean([ppr_scores.get(u, 0.0), ppr_scores.get(v, 0.0)])
            edge_scores[(u, v)] = score
            
        ppr_rank = {
            edge: rank 
            for rank, (edge, _) in enumerate(
                sorted(edge_scores.items(), key=lambda x: x[1], reverse=True), start=1
            )
        }

        # --- RRF Fusion ---
        scored = []
        for u, v in edges_set:
            # Check both directions for ranks since graph is undirected
            # but rank dictionaries keys depend on how they were populated

            rank_ppr = ppr_rank.get((u, v), ppr_rank.get((v, u), 1e6)) # PPR rank
            rank_sem = semantic_rank.get((u, v), semantic_rank.get((v, u), 1e6)) # Semantic rank
        
            score = 1 / (k_ppr + rank_ppr) + 1 / (k_sem + rank_sem)
            scored.append(((u, v), score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [e for e, _ in scored[:self.config.prune_top_k]]

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

    async def extract_query_timestamp(self, question: str) -> TemporalInterval:
        temporal_exprs = self.temporal_normalizer.extract_temporal_expressions(question)

        if temporal_exprs:
            expr = temporal_exprs[0]
            if 'start' in expr:
                return TemporalInterval(start=expr['start'], end=expr.get('end'))
        else:
            return self.temporal_normalizer.expand_to_interval("2025")
    
    async def retrieve_temporal(self, question: str) -> Dict[str, Any]:
        # 1. Extract query timestamp
        query_timestamp = await self.extract_query_timestamp(question)
        intent = await self.intent_classifier.classify(question)
  
        # 3. Standard ontology creation
        top_entities, passage_ranks, sentence_ranks, triplet_ranks = self.create_ontology(question)
        
        # 4. Apply temporal filtering to passages based on valid_time
        if query_timestamp:
            passage_ranks = self.filter_passages_by_valid_time(passage_ranks, query_timestamp)
        
        # 5. Extract named entities
        named_entities, usage = await self.ner.run(question, top_entities)
        
        # If no entities, use passage-based retrieval
        if not named_entities['entity_name']:
            context = self.retrieve_relevant_passages(question, passage_ranks)
            context["ner_entities"] = []
        else:
            # 6. Get seed nodes from entities
            seed_node_ids = self._get_seed_node_ids(named_entities['entity_name'])
            
            # 7. Apply temporal filtering to nodes based on valid_time (but keep fallback)
            original_seed_count = len(seed_node_ids)
            if query_timestamp and original_seed_count > 0:
                filtered_seed_ids = set(self.filter_nodes_by_valid_time(list(seed_node_ids), query_timestamp))
                
                # Only use filtered results if we didn't lose ALL entities
                if filtered_seed_ids:
                    seed_node_ids = filtered_seed_ids
                    # Update entity names to match filtered nodes
                    filtered_entity_names = [self.node_lookup[nid].name for nid in seed_node_ids if nid in self.node_lookup]
                    if filtered_entity_names:
                        named_entities['entity_name'] = filtered_entity_names
                # If filtering removed all entities but we had some originally, 
                # fall back to passage-based retrieval instead of refusing
                else:
                    context = self.retrieve_relevant_passages(question, passage_ranks)
                    context["ner_entities"] = []
                    context["query_timestamp"] = query_timestamp.to_dict() if query_timestamp else None
                    # context["available_context_time"] = self.available_context_time
                    context["note"] = "Temporal filtering removed all entities, using passage-based retrieval"
                    
                    return {
                        "context": json.dumps(context, indent=4, ensure_ascii=False),
                        "usage": usage,
                        "temporal_metadata": {
                            "query_timestamp": query_timestamp.to_dict() if query_timestamp else None,
                            #"available_context_time": self.available_context_time
                        }
                    }
            
            # 8. Get context with temporally-filtered entities and passages
            context = self.get_context(
                entity_names=named_entities['entity_name'],
                triplet_ranks=triplet_ranks,
                sentence_ranks=sentence_ranks,
                passage_ranks=passage_ranks,
                intent=intent, 
                query_timestamp=query_timestamp
            )
            
        # Add NER entities to context
        context["ner_entities"] = named_entities.get('entity_name', [])
        context["query_timestamp"] = query_timestamp.to_dict() if query_timestamp else None
        # context["available_context_time"] = self.available_context_time
        
        return {
            "context": json.dumps(context, indent=4, ensure_ascii=False),
            "usage": usage,
            "temporal_metadata": {
                "query_timestamp": query_timestamp.to_dict() if query_timestamp else None,
                #"available_context_time": self.available_context_time
            }
        }