import re
import uuid
from pydantic import Field, BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class TemporalInterval(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: Optional[str] = None 
    end:   Optional[str] = None  # None = now
    confidence: float = 1.0 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }
    
    def __hash__(self):
        return hash((self.start, self.end, self.confidence))
    
    def __eq__(self, other):
        if not isinstance(other, TemporalInterval):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.confidence == other.confidence)
    
    def __lt__(self, other):
        if not isinstance(other, TemporalInterval):
            return NotImplemented
        # Sort by start time, then end time
        return (self.start or "", self.end or "") < (other.start or "", other.end or "")

class Node(BaseModel):
    id: str
    name: str
    type: List[str]
    description: Optional[str]
    aliases: Optional[List[str]]
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    valid_time: Optional[TemporalInterval] = None
    is_temporal_entity: bool = False 

class Edge(BaseModel):
    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    relationship_name: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    valid_time: Optional[TemporalInterval] = None
    temporal_relation: Optional[str] = None  

class EntityLayer(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

class Sentence(BaseModel):
    sentence_id: str
    passage_id: str
    text: str
    node_map: List[str] = Field(default_factory=list)
    temporal_expressions: List[Dict[str, Any]] = Field(default_factory=list)  # time expressions
    
class Passage(BaseModel):
    passage_id: str
    text: str
    node_map: List[str] = Field(default_factory=list)
    sentence_map: List[str] = Field(default_factory=list)
    primary_time: Optional[TemporalInterval] = None
    temporal_expressions: List[Dict[str, Any]] = Field(default_factory=list)
    
class SentenceLayer(BaseModel):
    sentences: List[Sentence] = Field(default_factory=list)
    
class PassageLayer(BaseModel):
    passages: List[Passage] = Field(default_factory=list)


class HierarchicalKnowledgeGraph(BaseModel):
    passage_layer: PassageLayer = Field(default_factory=PassageLayer)
    sentence_layer: SentenceLayer = Field(default_factory=SentenceLayer)
    entity_layer: EntityLayer = Field(default_factory=EntityLayer)

    # Internal indexes for O(1) access
    node_id_to_node_name_map: Dict[str, List[str]] = Field(default_factory=dict)
    node_name_to_node_id_map: Dict[str, List[str]] = Field(default_factory=dict)
    
    node_id_to_edge_id_map: Dict[str, List[str]] = Field(default_factory=dict)
    node_id_to_adjacency_node_id_map: Dict[str, List[str]] = Field(default_factory=dict)
    node_id_to_passage_id_map: Dict[str, List[str]] = Field(default_factory=dict)
    node_id_to_sentence_id_map: Dict[str, List[str]] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], sentenizer=None) -> "HierarchicalKnowledgeGraph":
        if sentenizer is None:
            raise ValueError("sentenizer must be provided to split passage into sentences")

        entity_layer = EntityLayer.model_validate(data["entity_layer"])
        passage_text = data["paragraph_text"]

        # Step 1: Remap node IDs
        id_mapping = {node.id: str(uuid.uuid4()) for node in entity_layer.nodes}
        updated_nodes = [
            Node(
                id=id_mapping[node.id],
                name=node.name,
                type=node.type,
                description=node.description,
                aliases=node.aliases,
                properties=node.properties,
                valid_time=node.valid_time,
                is_temporal_entity=node.is_temporal_entity,
            )
            for node in entity_layer.nodes
        ]
        node_name_lookup = {
            node.id: {node.name.lower(), *(alias.lower() for alias in (node.aliases or []))}
            for node in updated_nodes
        }

        # Step 2: Update edges with new UUIDs
        updated_edges = []
        for edge in entity_layer.edges:
            if edge.source_node_id in id_mapping and edge.target_node_id in id_mapping:
                updated_edges.append(Edge(
                    source_node_id=id_mapping[edge.source_node_id],
                    target_node_id=id_mapping[edge.target_node_id],
                    relationship_name=edge.relationship_name,
                    properties=edge.properties,
                    valid_time=edge.valid_time,
                    temporal_relation=edge.temporal_relation,
                ))

        new_entity_layer = EntityLayer(nodes=updated_nodes, edges=updated_edges)

        # Step 3: Create passage and sentence layers
        passage_id = str(uuid.uuid4())
        doc = sentenizer(passage_text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text
            matched_node_ids = [
                node.id
                for node in updated_nodes
                if any(name in sent_text.lower() for name in node_name_lookup[node.id])
            ]
            sentences.append(Sentence(
                sentence_id=str(uuid.uuid4()),
                passage_id=passage_id,
                text=sent_text,
                node_map=matched_node_ids,
                temporal_expressions=[]
            ))

        passage = Passage(
            passage_id=passage_id,
            text=passage_text,
            sentence_map=[s.sentence_id for s in sentences],
            node_map=[node.id for node in updated_nodes],
            primary_time=None,
            temporal_expressions=[],
        )

        # Step 4: Build HKG
        hkg = cls(
            passage_layer=PassageLayer(passages=[passage]),
            sentence_layer=SentenceLayer(sentences=sentences),
            entity_layer=new_entity_layer,
        )
        hkg._build_mappings()
        return hkg
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        # Build all mappings after initialization
        self._build_mappings()
        
    def _build_mappings(self) -> None:
        """
        Populate internal lookup tables:
          - node_id_to_node_name_map: maps node_id to its list of names and aliases
          - node_name_to_node_id_map: maps lowercase name/alias to list of node_ids
          - node_id_to_edge_id_map: maps node_id to all connected edge_ids
          - node_id_to_adjacency_node_id_map: maps node_id to neighboring node_ids
          - node_id_to_passage_id_map: maps node_id to passages where it appears
          - node_id_to_sentence_id_map: maps node_id to sentences where it appears
        """
        # Reset all maps
        self.node_id_to_node_name_map.clear()
        self.node_name_to_node_id_map.clear()
        self.node_id_to_edge_id_map.clear()
        self.node_id_to_adjacency_node_id_map.clear()
        self.node_id_to_passage_id_map.clear()
        self.node_id_to_sentence_id_map.clear()

        # Build node name mappings
        for node in self.entity_layer.nodes:
            # Collect primary name and aliases
            names = [node.name] + (node.aliases or [])
            self.node_id_to_node_name_map[node.id] = names
            for name in names:
                key = name.lower()
                self.node_name_to_node_id_map.setdefault(key, []).append(node.id)

        # Build edge and adjacency mappings
        for edge in self.entity_layer.edges:
            # Edge associations
            self.node_id_to_edge_id_map.setdefault(edge.source_node_id, []).append(edge.edge_id)
            self.node_id_to_edge_id_map.setdefault(edge.target_node_id, []).append(edge.edge_id)
            # Adjacency (undirected)
            self.node_id_to_adjacency_node_id_map.setdefault(edge.source_node_id, []).append(edge.target_node_id)
            self.node_id_to_adjacency_node_id_map.setdefault(edge.target_node_id, []).append(edge.source_node_id)

        # Map nodes to passages
        for passage in self.passage_layer.passages:
            for nid in passage.node_map:
                self.node_id_to_passage_id_map.setdefault(nid, []).append(passage.passage_id)

        # Map nodes to sentences
        for sentence in self.sentence_layer.sentences:
            for nid in sentence.node_map:
                self.node_id_to_sentence_id_map.setdefault(nid, []).append(sentence.sentence_id)


    def reset_node_id(self, mode: str = 'uuid') -> None:
        """
        Resets all node IDs with new UUIDs or sequential numbers.
        """
        if mode == 'uuid':
            new_node_ids = [str(uuid.uuid4()) for _ in self.entity_layer.nodes]
        elif mode == 'number':
            new_node_ids = [str(i + 1) for i in range(len(self.entity_layer.nodes))]
        else:
            raise ValueError("mode must be 'uuid' or 'number'")

        # Map old node IDs to new ones
        id_mapping = {old.id: new for old, new in zip(self.entity_layer.nodes, new_node_ids)}

        # Update node IDs
        for node in self.entity_layer.nodes:
            node.id = id_mapping[node.id]

        # Update edges
        for idx, edge in enumerate(self.entity_layer.edges):
            edge.source_node_id = id_mapping[edge.source_node_id]
            edge.target_node_id = id_mapping[edge.target_node_id]

        # Update passage node maps
        for passage in self.passage_layer.passages:
            passage.node_map = [id_mapping[nid] for nid in passage.node_map]

        # Update sentence node maps
        for sentence in self.sentence_layer.sentences:
            sentence.node_map = [id_mapping[nid] for nid in sentence.node_map]

        # Rebuild internal mappings
        self._build_mappings()

    
    def merge_nodes(self, node_ids: List[str], reindex: bool = False) -> None:
        # Filter to existing nodes
        existing = {node.id: node for node in self.entity_layer.nodes}
        targets = [existing[nid] for nid in node_ids if nid in existing]
        if not targets:
            return
        # Base attributes
        primary = targets[0]
        merged_id = str(uuid.uuid4()) if reindex else primary.id
        # Combine names, types, aliases, descriptions, properties
        combined_types = list({t for node in targets for t in node.type})
        combined_aliases = list({alias for node in targets for alias in (node.aliases or [])}) or None
        descriptions = [node.description for node in targets if node.description]
        combined_description = "\n".join(descriptions) if descriptions else None
        combined_props: Dict[str, Any] = {}
        for node in targets:
            for k, v in (node.properties or {}).items():
                if k not in combined_props:
                    combined_props[k] = v
        # Create new node
        new_node = Node(
            id=merged_id,
            name=primary.name,
            type=combined_types,
            description=combined_description,
            aliases=combined_aliases,
            properties=combined_props,
            valid_time=primary.valid_time,
            is_temporal_entity=primary.is_temporal_entity,
        )
        # Remove old nodes and add new
        self.entity_layer.nodes = [n for n in self.entity_layer.nodes if n.id not in node_ids]
        self.entity_layer.nodes.append(new_node)

        # Rewire edges
        new_edges: List[Edge] = []
        seen = set()
        for edge in self.entity_layer.edges:
            src = edge.source_node_id
            tgt = edge.target_node_id
            if src in node_ids:
                src = merged_id
            if tgt in node_ids:
                tgt = merged_id
            if src == tgt:
                continue
            key = (src, tgt, edge.relationship_name)
            if key in seen:
                continue
            seen.add(key)
            edge.source_node_id = src
            edge.target_node_id = tgt
            new_edges.append(edge)
        self.entity_layer.edges = new_edges

        # Update passage & sentence maps
        for passage in self.passage_layer.passages:
            pm = passage.node_map
            if any(n in node_ids for n in pm):
                passage.node_map = list({merged_id if n in node_ids else n for n in pm})
        for sentence in self.sentence_layer.sentences:
            sm = sentence.node_map
            if any(n in node_ids for n in sm):
                sentence.node_map = list({merged_id if n in node_ids else n for n in sm})

        # Rebuild all internal mappings
        if reindex:
            self._build_mappings()
            
    def print_graph_report(self) -> None:
        """
        Prints a concise, human-readable summary of the HKG:
        • Total nodes & edges
        • Total passages & sentences
        • Node degree stats (min/avg/max) + top 5 by degree
        • Mapping stats (passage/sentence appearances per node)
        • Top 5 nodes by passage appearance
        """
        print("\n" + "=" * 60)
        print(" Hierarchical Knowledge Graph Summary ".center(60, "="))
        print("=" * 60)

        # Basic counts
        n_nodes = len(self.entity_layer.nodes)
        n_edges = len(self.entity_layer.edges)
        n_passages = len(self.passage_layer.passages)
        n_sentences = len(self.sentence_layer.sentences)
        print(f"• Total nodes:      {n_nodes}")
        print(f"• Total edges:      {n_edges}")
        print(f"• Total passages:   {n_passages}")
        print(f"• Total sentences:  {n_sentences}")

        # Degree stats
        degrees = [
            len(self.node_id_to_adjacency_node_id_map.get(node.id, []))
            for node in self.entity_layer.nodes
        ]
        if n_nodes:
            print(f"• Node degree:      min {min(degrees)}   avg {sum(degrees)/n_nodes:.2f}   max {max(degrees)}")
        else:
            print("• Node degree:      N/A")

        # Top 5 nodes by degree
        ranked = sorted(
            ((node.id, deg) for node, deg in zip(self.entity_layer.nodes, degrees)),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        if ranked:
            print("\n Top 5 nodes by degree:")
            for nid, deg in ranked:
                name = self.node_id_to_node_name_map.get(nid, [nid])[0]
                print(f"   • {name} ({nid}): {deg}")

        # Mapping stats
        pass_counts = [
            len(self.node_id_to_passage_id_map.get(node.id, []))
            for node in self.entity_layer.nodes
        ]
        sent_counts = [
            len(self.node_id_to_sentence_id_map.get(node.id, []))
            for node in self.entity_layer.nodes
        ]
        if n_nodes:
            p_min, p_avg, p_max = min(pass_counts), sum(pass_counts)/n_nodes, max(pass_counts)
            s_min, s_avg, s_max = min(sent_counts), sum(sent_counts)/n_nodes, max(sent_counts)
            print("\n Mapping appearances per node:")
            print(f"   • Passages:   min {p_min}   avg {p_avg:.2f}   max {p_max}")
            print(f"   • Sentences:  min {s_min}   avg {s_avg:.2f}   max {s_max}")
        else:
            print("\n Mapping appearances per node: N/A")

        # Top 5 nodes by passage appearances
        top_passage_nodes = sorted(
            ((node.id, count) for node, count in zip(self.entity_layer.nodes, pass_counts)),
            key=lambda x: x[1],
            reverse=True)[:5]
        if top_passage_nodes:
            print("\n Top 5 nodes by passage appearance:")
            for nid, count in top_passage_nodes:
                name = self.node_id_to_node_name_map.get(nid, [nid])[0]
                print(f"   • {name} ({nid}): {count}")

        print("=" * 60 + "\n")


    @classmethod
    def combine(cls, hkg_list: List["HierarchicalKnowledgeGraph"]) -> "HierarchicalKnowledgeGraph":
        """
        Combines multiple HierarchicalKnowledgeGraph objects into a single one.
        Ensures unique IDs across graphs, updates edges accordingly, and rebuilds mappings.
        """
        # Accumulators
        combined_nodes: List[Node] = []
        combined_edges: List[Edge] = []
        combined_passages: List[Passage] = []
        combined_sentences: List[Sentence] = []

        # Global remap tables
        node_map: Dict[str, str] = {}
        passage_map: Dict[str, str] = {}
        sentence_map: Dict[str, str] = {}

        # 1) Remap and collect nodes, passages, sentences
        for hkg in hkg_list:
            # Nodes
            for n in hkg.entity_layer.nodes:
                new_id = str(uuid.uuid4())
                node_map[n.id] = new_id
                combined_nodes.append(
                    Node(
                        id=new_id,
                        name=n.name,
                        type=list(dict.fromkeys(n.type)),                # preserve order & dedupe
                        description=n.description,
                        aliases=list(dict.fromkeys(n.aliases or [])) or None,
                        properties=(n.properties or {}).copy(),
                        valid_time=n.valid_time,
                        is_temporal_entity=n.is_temporal_entity
                    )
                )
            # Passages
            for p in hkg.passage_layer.passages:
                new_pid = str(uuid.uuid4())
                passage_map[p.passage_id] = new_pid
                combined_passages.append(
                    Passage(
                        passage_id=new_pid,
                        text=p.text,
                        node_map=[node_map[x] for x in p.node_map if x in node_map],
                        sentence_map=[],  # fill after sentence remap
                        primary_time=p.primary_time,
                        temporal_expressions=p.temporal_expressions
                    )
                )
            # Sentences
            for s in hkg.sentence_layer.sentences:
                new_sid = str(uuid.uuid4())
                sentence_map[s.sentence_id] = new_sid
                combined_sentences.append(Sentence(
                        sentence_id=new_sid,
                        passage_id=passage_map.get(s.passage_id),
                        text=s.text,
                        node_map=[node_map[x] for x in s.node_map if x in node_map],
                        temporal_expressions=s.temporal_expressions
                    )
                )

        # 2) Wire up passage → sentences
        sentences_by_passage: Dict[str, List[str]] = {}
        for s in combined_sentences:
            sentences_by_passage.setdefault(s.passage_id, []).append(s.sentence_id)
        for p in combined_passages:
            p.sentence_map = sentences_by_passage.get(p.passage_id, [])

        # 3) Remap & collect edges, avoiding self-loops
        seen = set()
        for hkg in hkg_list:
            for e in hkg.entity_layer.edges:
                src = node_map.get(e.source_node_id)
                tgt = node_map.get(e.target_node_id)
                if not src or not tgt or src == tgt:
                    continue
                key = (src, tgt, e.relationship_name)
                if key in seen:
                    continue
                seen.add(key)
                combined_edges.append(
                    Edge(
                        source_node_id=src,
                        target_node_id=tgt,
                        relationship_name=e.relationship_name,
                        properties=(e.properties or {}).copy(),
                        valid_time=e.valid_time,
                        temporal_relation=e.temporal_relation,
                    )
                )

        # 4) Build and index the merged HKG
        merged = cls(
            passage_layer=PassageLayer(passages=combined_passages),
            sentence_layer=SentenceLayer(sentences=combined_sentences),
            entity_layer=EntityLayer(nodes=combined_nodes, edges=combined_edges),
        )
        merged._build_mappings()
        return merged