import os
import time
import json
import ahocorasick

from tqdm import tqdm
from collections import Counter
from wordfreq import top_n_list
from sklearn.cluster import DBSCAN
from collections import defaultdict
from typing import List, Dict, Any, Literal

from higra_agent.model_cache import ModelCache
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.higra_merge_prompt import HiGraMergePrompt
from higra_agent.prompt.higra_completion_prompt import HiGraCompletionPrompt
from higra_agent.retriever.higra_retriever.higra_schema import Node, Edge, HierarchicalKnowledgeGraph


class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            
class HiGraAligner:
    def __init__(self, embedder, api_key: str, model_name: str, batch_size: int):
        self.embedder = embedder
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size

    @staticmethod
    def combine(hkgs: List[HierarchicalKnowledgeGraph]) -> HierarchicalKnowledgeGraph:
        combined = HierarchicalKnowledgeGraph.combine(hkgs)
        combined._build_mappings()
        return combined

    @staticmethod
    def align_exact(hkg: HierarchicalKnowledgeGraph, reindex: bool = True) -> HierarchicalKnowledgeGraph:
        start_time = time.time()
        groups = defaultdict(list)
        for n in hkg.entity_layer.nodes:
            groups[n.name].append((n.id, set(n.type)))
        to_merge = []
        for items in groups.values():
            if len(items) < 2:
                continue
            uf = UnionFind()
            for nid, types in items:
                for oid, otypes in items:
                    if nid != oid and types & otypes:
                        uf.union(nid, oid)
            comps = defaultdict(list)
            for nid, _ in items:
                comps[uf.find(nid)].append(nid)
            for comp in comps.values():
                if len(comp) > 1:
                    to_merge.append(comp)
        print(f"[HiGraAligner][Align Exact]: {time.time() - start_time:.2f} seconds to find candidates merge nodes")
        print(f"[HiGraAligner][Align Exact]: {len(to_merge)} clusters found")
        for cluster in tqdm(to_merge):
            hkg.merge_nodes(cluster)
        print(f"[HiGraAligner][Align Exact]: {time.time() - start_time:.2f} seconds to merge nodes")
        if reindex:
            hkg._build_mappings()
        return hkg

    @staticmethod
    def _all_strings_no_alphabet(strings: List[str]) -> bool:
        return any(not any(c.isalpha() for c in s) for s in strings)

    @staticmethod
    def _are_all_nodes_interconnected(
        nodes: List[Node],
        edges: List[Edge]
    ) -> bool:
        ids = {n.id for n in nodes}
        conn = {nid: set() for nid in ids}
        for e in edges:
            if e.source_node_id in ids and e.target_node_id in ids:
                conn[e.source_node_id].add(e.target_node_id)
                conn[e.target_node_id].add(e.source_node_id)
        for nid in ids:
            # each node must connect to all others
            if len(conn[nid]) != len(ids) - 1:
                return False
        return True
    
    async def align_semantic(
        self,
        hkg: HierarchicalKnowledgeGraph,
        mode: Literal["merge", "complete"],
        response_format: str = None
    ) -> HierarchicalKnowledgeGraph:
        # cluster by embedding
        desc = [f"[{','.join(n.type)}]: {n.name}" for n in hkg.entity_layer.nodes]
        emb = self.embedder.encode(desc)
        labels = DBSCAN(metric="cosine", eps=0.5, min_samples=2).fit_predict(emb)
        clusters = defaultdict(list)
        for n, lbl in zip(hkg.entity_layer.nodes, labels):
            if lbl >= 0:
                clusters[lbl].append(n)
        messages = []
        for cluster in tqdm(clusters.values()):
            
            if len(cluster) < 2 or len(cluster) > 50:
                continue

            names = [n.name for n in cluster]
            # 1) skip non-alphabetic clusters
            if self._all_strings_no_alphabet(names):
                # print(f"[HiGraAlginer][Align Semantic]: skipping non-alphabetic cluster {names}")
                continue

            # 2) skip fully connected clusters in completion mode
            if mode == "complete" and \
               self._are_all_nodes_interconnected(cluster, hkg.entity_layer.edges):
                # print(f"[HiGraAlginer][Align Semantic]: skipping ally connected cluster {names}")
                continue

            node_information = [n.model_dump() for n in cluster]
            
            for node in node_information:
                node['description'] = ""
            
            
            summary = json.dumps(node_information, ensure_ascii=False)
            
            if mode == "merge":
                messages.append([
                    HiGraMergePrompt.system_prompt,
                    HiGraMergePrompt.user_prompt.format(candidate_summary=summary)
                ])
            else:
                messages.append([
                    HiGraCompletionPrompt.system_prompt,
                    HiGraCompletionPrompt.user_prompt.format(candidate_summary=summary)
                ])
        
        if not messages:
            return hkg

        # print(f"[HiGraAlginer][Align Semantic]: {len(messages)} clusters found")
        
        # LLM call stays the same
        client = LLMAsyncClient(api_key=self.api_key, model_name=self.model_name)
        raw = await client.call_multiple_batched(
            data=messages,
            response_format=response_format,
            max_tokens=1500,
            batch_size=self.batch_size
        )

        # process responses
        result = [x[0] for x in raw]
        for text in result:
            parsed = client.parse_json(text)
            if not parsed:
                continue
            if mode == "merge":
                instrs = parsed.get("merge_instruction", [])
                actions = self._process_merge_instructions(instrs)
                for act in tqdm(actions):
                    target_node_ids = [act["base_node_id"]] + act["merge_node_ids"]
                    # print(f"[HiGraAlginer][Align Semantic]: Merge {[hkg.node_id_to_node_name_map[nid] for nid in target_node_ids]}")    
                    hkg.merge_nodes(target_node_ids, reindex=False)
            else:
                for rel in parsed.get("relation_predictions", []):
                    s, t = rel["entity_2_id"], rel["entity_1_id"]
                    if s == t:
                        # print(f"[HiGraAlginer][Align Semantic]: skipping self-loop edge {s} -> {t}")
                        continue
                    if s not in hkg.node_id_to_node_name_map.keys():
                        # print(f"[HiGraAlginer][Align Semantic]: skipping missing source node {s}")
                        continue
                    if t not in hkg.node_id_to_node_name_map.keys():
                        # print(f"[HiGraAlginer][Align Semantic]: skipping missing target node {t}")
                        continue
                
                    # print(f"[HiGraAlginer][Align Semantic]: adding edge {hkg.node_id_to_node_name_map[s]} -> {hkg.node_id_to_node_name_map[t]}")
                    hkg.entity_layer.edges.append(Edge(
                        source_node_id=s,
                        target_node_id=t,
                        relationship_name=rel.get("relationship_name", "related_to")
                    ))
        hkg._build_mappings()
        return hkg

    @staticmethod
    def _process_merge_instructions(instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for instr in instructions:
            base = instr["base_node_id"]
            for m in instr["merge_node_ids"]:
                union(base, m)
        clusters = defaultdict(set)
        for instr in instructions:
            ids = {instr["base_node_id"]} | set(instr["merge_node_ids"])
            for nid in ids:
                clusters[find(nid)].add(nid)
        merged = []
        for root, members in clusters.items():
            if len(members) < 2:
                continue
            base = min(members)
            others = sorted(members - {base})
            merged.append({"base_node_id": base, "merge_node_ids": others})
        return merged

    @staticmethod
    def interlayer(hkg: HierarchicalKnowledgeGraph, max_occurrence: int = 100) -> HierarchicalKnowledgeGraph:
        """
        Scan every passage and sentence for node names/aliases,
        and link any newly discovered node IDs into their node_map.
        Skips overly common matches. Finally rebuilds all mappings.
        """

        # Build Aho–Corasick trie of lowercase names → node IDs
        A = ahocorasick.Automaton()
        common_words = top_n_list('en', 2000) + ['le']
        for node in hkg.entity_layer.nodes:
            for label in ([node.name] + (node.aliases or [])):
                key = label.strip().lower()
                if (
                    not key or
                    key in common_words or
                    key.isnumeric() or
                    len(key) <= 1
                ):
                    # print(f"[HiGraAligner][Interlayer Alignment]: skipping label '{key}'")
                    continue
                A.add_word(key, node.id)
        A.make_automaton()

        # First pass: count occurrences of node IDs across all text
        freq_counter = Counter()
        for container in hkg.passage_layer.passages + hkg.sentence_layer.sentences:
            for _, nid in A.iter(getattr(container, "text").lower()):
                freq_counter[nid] += 1

        # Filter out overused node IDs
        disallowed_nids = {nid for nid, count in freq_counter.items() if count > max_occurrence}

        # Annotator with frequency filter
        new_links = 0
        def _annotate(container, text_attr: str, map_attr: str):
            nonlocal new_links
            text = getattr(container, text_attr).lower()
            seen = set(getattr(container, map_attr))
            for _, nid in A.iter(text):
                if nid in disallowed_nids:
                    continue
                if nid not in seen:
                    getattr(container, map_attr).append(nid)
                    seen.add(nid)
                    new_links += 1

        for passage in hkg.passage_layer.passages:
            _annotate(passage, "text", "node_map")
        for sentence in hkg.sentence_layer.sentences:
            _annotate(sentence, "text", "node_map")

        hkg._build_mappings()

        print(f"[HiGraAligner] Interlayer Alignment Complete: {new_links} new connections added.")
        return hkg

    async def build_open_higra(
        self,
        higra_list: List[HierarchicalKnowledgeGraph],
        save_path:str,
    ) -> HierarchicalKnowledgeGraph:

        def pre_load_data(path):
            with open(cur_path, 'r') as f:
                higra_data = json.load(f)
            open_higra = HierarchicalKnowledgeGraph.model_validate(higra_data)
            return open_higra

        
        print("### Phase: Combine")
        cur_path = save_path.replace(".json", "-original.json")
        if os.path.exists(cur_path):
            open_higra = pre_load_data(cur_path)
        else:
            open_higra = self.combine(higra_list)
            open_higra.reset_node_id(mode='number')
            print("Original")
            open_higra.print_graph_report()
        
            with open(cur_path, "w", encoding='utf-8') as file:
                json.dump(open_higra.model_dump(), file, indent=4, ensure_ascii=False)

        print("### Phase: Align Exact")
        cur_path = save_path.replace(".json", "-align_exact.json")
        if os.path.exists(cur_path):
            open_higra = pre_load_data(cur_path)
        else:
            open_higra = self.align_exact(open_higra, reindex=False)
            open_higra.reset_node_id(mode='number')
            print("Align exact")
            open_higra.print_graph_report()
    
            with open(cur_path, "w", encoding='utf-8') as file:
                json.dump(open_higra.model_dump(), file, indent=4, ensure_ascii=False)

        print("### Phase: Align Semantic")
        cur_path = save_path.replace(".json", "-align_semantic.json")
        if os.path.exists(cur_path):
            open_higra = pre_load_data(cur_path)
        else:
            open_higra = await self.align_semantic(open_higra, mode="merge")
            print("Merge semantic")
            open_higra.reset_node_id(mode='number')
            open_higra.print_graph_report()
    
            with open(cur_path, "w", encoding='utf-8') as file:
                json.dump(open_higra.model_dump(), file, indent=4, ensure_ascii=False)

        print("### Phase: Align Complete")
        cur_path = save_path.replace(".json", "-align_complete.json")
        if os.path.exists(cur_path):
            open_higra = pre_load_data(cur_path)
        else:
            open_higra = await self.align_semantic(open_higra, mode="complete")
            print("Merge complete")
            open_higra.reset_node_id(mode='number')
            open_higra.print_graph_report()

            with open(cur_path, "w", encoding='utf-8') as file:
                json.dump(open_higra.model_dump(), file, indent=4, ensure_ascii=False)

        print("### Phase: Final")
        open_higra = self.interlayer(open_higra)
        open_higra.reset_node_id(mode='number')
        open_higra.print_graph_report()

        with open(save_path, "w", encoding='utf-8') as file:
            json.dump(open_higra.model_dump(), file, indent=4, ensure_ascii=False)
        return open_higra