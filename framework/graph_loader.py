"""
graph_loader.py
---------------
Loads and validates GRAPH JSON documents representing TOCAP/OCAP
manufacturing diagnostic flowcharts.

Each document is a directed graph with four node types:
  - terminator : start or resolution/escalation endpoint
  - decision   : Yes/No branching condition
  - process    : action or diagnostic step
  - document   : reference to a sub-procedure
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    node_id: str
    node_type: str          # terminator | decision | process | document
    text: str

    def is_decision(self) -> bool:
        return self.node_type == "decision"

    def is_terminator(self) -> bool:
        return self.node_type == "terminator"

    def is_process(self) -> bool:
        return self.node_type == "process"

    def is_document(self) -> bool:
        return self.node_type == "document"


@dataclass
class GraphEdge:
    source: str
    target: str
    label: Optional[str]    # "Yes" | "No" | None

    def is_yes(self) -> bool:
        return self.label == "Yes"

    def is_no(self) -> bool:
        return self.label == "No"

    def is_unconditional(self) -> bool:
        return self.label is None


@dataclass
class Graph:
    graph_id: str
    description: str
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]

    # Derived lookup tables (populated post-init)
    outgoing: dict[str, list[GraphEdge]] = field(default_factory=dict)
    incoming: dict[str, list[GraphEdge]] = field(default_factory=dict)

    def __post_init__(self):
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        self.outgoing = {}
        self.incoming = {}
        for edge in self.edges:
            self.outgoing.setdefault(edge.source, []).append(edge)
            self.incoming.setdefault(edge.target, []).append(edge)

    def get_node(self, node_id: str) -> GraphNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in graph '{self.graph_id}'")
        return self.nodes[node_id]

    def get_start_node(self) -> GraphNode:
        """Return the start terminator node (the one with no incoming edges)."""
        for node_id, node in self.nodes.items():
            if node.is_terminator() and node_id not in self.incoming:
                return node
        raise ValueError(f"No start node found in graph '{self.graph_id}'")

    def get_terminal_nodes(self) -> list[GraphNode]:
        """Return all terminator nodes that are not the start node."""
        start = self.get_start_node()
        return [
            node for node in self.nodes.values()
            if node.is_terminator() and node.node_id != start.node_id
        ]

    def get_decision_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.is_decision()]

    def get_process_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.is_process()]

    def get_document_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.is_document()]

    def successors(self, node_id: str) -> list[GraphEdge]:
        """Return outgoing edges from a given node."""
        return self.outgoing.get(node_id, [])

    def predecessors(self, node_id: str) -> list[GraphEdge]:
        """Return incoming edges to a given node."""
        return self.incoming.get(node_id, [])

    @property
    def complexity(self) -> str:
        """Classify graph complexity based on number of decision nodes."""
        n = len(self.get_decision_nodes())
        if n == 0:
            return "linear"
        elif n <= 3:
            return "short_tree"
        elif n <= 7:
            return "medium_chain"
        else:
            return "deep_ladder"

    @property
    def has_subdocument_references(self) -> bool:
        return len(self.get_document_nodes()) > 0

    def summary(self) -> dict:
        return {
            "id": self.graph_id,
            "description": self.description,
            "total_nodes": len(self.nodes),
            "decision_nodes": len(self.get_decision_nodes()),
            "process_nodes": len(self.get_process_nodes()),
            "document_nodes": len(self.get_document_nodes()),
            "terminator_nodes": len([n for n in self.nodes.values() if n.is_terminator()]),
            "total_edges": len(self.edges),
            "complexity": self.complexity,
            "has_subdocument_references": self.has_subdocument_references,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class GraphLoader:
    """
    Loads GRAPH JSON documents from a directory or individual files.

    Usage:
        loader = GraphLoader("data/graphs/")
        graphs = loader.load_all()
        graph = loader.load("GRAPH21")
    """

    VALID_NODE_TYPES = {"terminator", "decision", "process", "document"}

    def __init__(self, graphs_dir: str):
        self.graphs_dir = Path(graphs_dir)
        if not self.graphs_dir.exists():
            raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")
        self._cache: dict[str, Graph] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, graph_id: str) -> Graph:
        """Load a single graph by ID (e.g. 'GRAPH21')."""
        if graph_id in self._cache:
            return self._cache[graph_id]

        fpath = self.graphs_dir / f"{graph_id}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Graph file not found: {fpath}")

        graph = self._load_file(fpath)
        self._cache[graph_id] = graph
        return graph

    def load_all(self) -> dict[str, Graph]:
        """Load all GRAPH JSON files in the directory."""
        graphs = {}
        for fpath in sorted(self.graphs_dir.glob("GRAPH*.json")):
            graph = self._load_file(fpath)
            graphs[graph.graph_id] = graph
            self._cache[graph.graph_id] = graph
        return graphs

    def load_by_complexity(self, complexity: str) -> dict[str, Graph]:
        """
        Load graphs filtered by complexity tier.
        complexity: 'linear' | 'short_tree' | 'medium_chain' | 'deep_ladder'
        """
        all_graphs = self.load_all()
        return {
            gid: g for gid, g in all_graphs.items()
            if g.complexity == complexity
        }

    def load_with_subdocuments(self) -> dict[str, Graph]:
        """Load only graphs that contain sub-procedure document references."""
        all_graphs = self.load_all()
        return {gid: g for gid, g in all_graphs.items() if g.has_subdocument_references}

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _load_file(self, fpath: Path) -> Graph:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph_id = data.get("id", fpath.stem)
        description = data.get("description", "")
        graph_data = data.get("graph", data)

        nodes = self._parse_nodes(graph_data.get("nodes", {}), graph_id)
        edges = self._parse_edges(graph_data.get("edges", []), graph_id)
        self._validate(graph_id, nodes, edges)

        return Graph(
            graph_id=graph_id,
            description=description,
            nodes=nodes,
            edges=edges,
        )

    def _parse_nodes(self, raw_nodes: dict, graph_id: str) -> dict[str, GraphNode]:
        nodes = {}
        for node_id, node_data in raw_nodes.items():
            node_type = node_data.get("type", "").lower()
            if node_type not in self.VALID_NODE_TYPES:
                raise ValueError(
                    f"[{graph_id}] Unknown node type '{node_type}' on node '{node_id}'. "
                    f"Valid types: {self.VALID_NODE_TYPES}"
                )
            nodes[node_id] = GraphNode(
                node_id=node_id,
                node_type=node_type,
                text=node_data.get("text", ""),
            )
        return nodes

    def _parse_edges(self, raw_edges: list, graph_id: str) -> list[GraphEdge]:
        edges = []
        for edge_data in raw_edges:
            edges.append(GraphEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                label=edge_data.get("label"),
            ))
        return edges

    def _validate(self, graph_id: str, nodes: dict[str, GraphNode], edges: list[GraphEdge]):
        """
        Validate graph structural integrity:
        - All edge endpoints reference existing nodes
        - At least one start terminator exists
        - No dead-end process nodes
        """
        node_ids = set(nodes.keys())

        # Check edge endpoints
        for edge in edges:
            if edge.source not in node_ids:
                raise ValueError(f"[{graph_id}] Edge source '{edge.source}' not in nodes")
            if edge.target not in node_ids:
                raise ValueError(f"[{graph_id}] Edge target '{edge.target}' not in nodes")

        # Check for start node
        outgoing = {}
        incoming = {}
        for e in edges:
            outgoing.setdefault(e.source, []).append(e)
            incoming.setdefault(e.target, []).append(e)

        start_nodes = [
            nid for nid, n in nodes.items()
            if n.is_terminator() and nid not in incoming
        ]
        if not start_nodes:
            raise ValueError(f"[{graph_id}] No start terminator node found")

        # Warn on dead-end process nodes (not fatal — log only)
        dead_ends = [
            nid for nid, n in nodes.items()
            if n.node_type not in {"terminator", "document"} and nid not in outgoing
        ]
        if dead_ends:
            print(f"[WARNING] [{graph_id}] Dead-end process nodes detected: {dead_ends}")
