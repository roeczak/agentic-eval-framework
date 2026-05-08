"""
graph_utils.py
--------------
Utility functions for traversing and analysing GRAPH documents.
Used by task implementations and the graph-aware scorer.

Key capabilities:
  - Enumerate all valid paths from start to terminal nodes
  - Find the correct path given a sequence of Yes/No decisions
  - Extract sub-graphs for specific decision branches
  - Compute node-level traversal overlap between two paths
"""

from __future__ import annotations
from typing import Optional
from .graph_loader import Graph, GraphNode, GraphEdge


# ---------------------------------------------------------------------------
# Path representation
# ---------------------------------------------------------------------------

class GraphPath:
    """
    A sequence of (node_id, edge_label) pairs representing one traversal
    through the graph from start to a terminal node.

    edge_label is None for unconditional edges, "Yes"/"No" for decision branches.
    """

    def __init__(self, steps: list[tuple[str, Optional[str]]]):
        # steps: list of (node_id, incoming_edge_label)
        self.steps = steps

    @property
    def node_ids(self) -> list[str]:
        return [s[0] for s in self.steps]

    @property
    def decision_labels(self) -> list[Optional[str]]:
        """Return only the Yes/No labels at decision branch points."""
        return [s[1] for s in self.steps if s[1] in ("Yes", "No")]

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        parts = []
        for node_id, label in self.steps:
            if label:
                parts.append(f"--[{label}]--> {node_id}")
            else:
                parts.append(node_id)
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "nodes": self.node_ids,
            "decision_labels": self.decision_labels,
            "length": len(self),
        }


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

class GraphUtils:
    """
    Static utility methods for graph traversal and analysis.
    All methods accept a Graph object from graph_loader.py.
    """

    MAX_PATH_DEPTH = 100    # Guard against infinite loops in malformed graphs

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def enumerate_paths(graph: Graph) -> list[GraphPath]:
        """
        Enumerate all valid paths from the start node to any terminal node.
        Uses DFS with cycle detection.
        Returns a list of GraphPath objects.
        """
        start = graph.get_start_node()
        paths = []
        GraphUtils._dfs(
            graph=graph,
            current_id=start.node_id,
            visited=set(),
            current_path=[(start.node_id, None)],
            paths=paths,
            depth=0,
        )
        return paths

    @staticmethod
    def _dfs(
        graph: Graph,
        current_id: str,
        visited: set[str],
        current_path: list[tuple[str, Optional[str]]],
        paths: list[GraphPath],
        depth: int,
    ):
        if depth > GraphUtils.MAX_PATH_DEPTH:
            return

        current_node = graph.get_node(current_id)

        # Terminal: record path
        if current_node.is_terminator() and depth > 0:
            paths.append(GraphPath(list(current_path)))
            return

        visited = visited | {current_id}   # immutable copy for branching

        for edge in graph.successors(current_id):
            if edge.target in visited:
                # Loop-back edge — follow once to allow re-check pattern
                # but don't recurse infinitely
                continue
            current_path.append((edge.target, edge.label))
            GraphUtils._dfs(
                graph=graph,
                current_id=edge.target,
                visited=visited,
                current_path=current_path,
                paths=paths,
                depth=depth + 1,
            )
            current_path.pop()

    # ------------------------------------------------------------------
    # Path lookup
    # ------------------------------------------------------------------

    @staticmethod
    def get_path_for_decisions(
        graph: Graph,
        decisions: dict[str, str],
    ) -> Optional[GraphPath]:
        """
        Find the graph path consistent with a given set of Yes/No decisions.

        decisions: dict mapping decision node_id to "Yes" or "No"
                   e.g. {"box_1": "Yes", "box_4": "No"}

        Returns the first matching GraphPath, or None if no path matches.
        """
        paths = GraphUtils.enumerate_paths(graph)
        for path in paths:
            if GraphUtils._path_matches_decisions(graph, path, decisions):
                return path
        return None

    @staticmethod
    def _path_matches_decisions(
        graph: Graph,
        path: GraphPath,
        decisions: dict[str, str],
    ) -> bool:
        """Check whether a path is consistent with a set of Yes/No decisions."""
        node_ids = path.node_ids
        for node_id, label in path.steps:
            if label in ("Yes", "No"):
                # Find the decision node that generated this label
                # (it's the node just before this step in the path)
                idx = path.node_ids.index(node_id)
                if idx == 0:
                    continue
                prev_node_id = path.node_ids[idx - 1]
                if prev_node_id in decisions:
                    expected = decisions[prev_node_id]
                    if label != expected:
                        return False
        return True

    # ------------------------------------------------------------------
    # Step-by-step traversal
    # ------------------------------------------------------------------

    @staticmethod
    def next_step(
        graph: Graph,
        current_node_id: str,
        decision: Optional[str] = None,
    ) -> Optional[str]:
        """
        Given the current node, return the next node ID.

        For decision nodes, decision must be "Yes" or "No".
        For process/document nodes, returns the single successor (if any).
        Returns None if no valid successor exists.
        """
        edges = graph.successors(current_node_id)
        if not edges:
            return None

        current_node = graph.get_node(current_node_id)

        if current_node.is_decision():
            if decision is None:
                raise ValueError(
                    f"Decision required at node '{current_node_id}' "
                    f"({current_node.text[:60]})"
                )
            for edge in edges:
                if edge.label == decision:
                    return edge.target
            return None  # No matching branch

        else:
            # Take first unconditional edge
            for edge in edges:
                if edge.label is None:
                    return edge.target
            # Fall back to first edge
            return edges[0].target

    # ------------------------------------------------------------------
    # Path comparison (for Task 4 scoring)
    # ------------------------------------------------------------------

    @staticmethod
    def path_overlap(
        reference: GraphPath,
        predicted: GraphPath,
    ) -> dict:
        """
        Compute node-level overlap between a reference and predicted path.

        Returns:
            precision : fraction of predicted nodes that are in reference
            recall    : fraction of reference nodes that are in predicted
            f1        : harmonic mean of precision and recall
            correct_decisions : fraction of Yes/No decisions matching reference
        """
        ref_nodes = set(reference.node_ids)
        pred_nodes = set(predicted.node_ids)

        if not pred_nodes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "correct_decisions": 0.0}

        precision = len(ref_nodes & pred_nodes) / len(pred_nodes)
        recall = len(ref_nodes & pred_nodes) / len(ref_nodes) if ref_nodes else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Decision accuracy: compare Yes/No sequences
        ref_decisions = reference.decision_labels
        pred_decisions = predicted.decision_labels
        if ref_decisions:
            matches = sum(
                r == p
                for r, p in zip(ref_decisions, pred_decisions)
            )
            correct_decisions = matches / len(ref_decisions)
        else:
            correct_decisions = 1.0  # No decisions to compare

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "correct_decisions": round(correct_decisions, 4),
        }

    # ------------------------------------------------------------------
    # Graph analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_decision_context(
        graph: Graph,
        decision_node_id: str,
    ) -> dict:
        """
        Return structured context for a decision node:
        - the decision text
        - the Yes branch: next node text
        - the No branch: next node text
        - predecessor process nodes (diagnostic steps leading to this decision)
        """
        node = graph.get_node(decision_node_id)
        if not node.is_decision():
            raise ValueError(f"Node '{decision_node_id}' is not a decision node")

        yes_branch = None
        no_branch = None
        for edge in graph.successors(decision_node_id):
            target = graph.get_node(edge.target)
            if edge.label == "Yes":
                yes_branch = {"node_id": edge.target, "text": target.text}
            elif edge.label == "No":
                no_branch = {"node_id": edge.target, "text": target.text}

        # Find process nodes that feed into this decision (diagnostics)
        diagnostic_steps = []
        for edge in graph.predecessors(decision_node_id):
            pred = graph.get_node(edge.source)
            if pred.is_process():
                diagnostic_steps.append({"node_id": edge.source, "text": pred.text})

        return {
            "decision_id": decision_node_id,
            "question": node.text,
            "yes_branch": yes_branch,
            "no_branch": no_branch,
            "diagnostic_steps": diagnostic_steps,
        }

    @staticmethod
    def get_escalation_node(graph: Graph) -> Optional[GraphNode]:
        """
        Return the escalation terminal node if present.
        Looks for terminators whose text contains escalation keywords.
        """
        keywords = ["escalate", "notify", "maintenance", "qa", "quality", "engineer"]
        for node in graph.get_terminal_nodes():
            text_lower = node.text.lower()
            if any(kw in text_lower for kw in keywords):
                return node
        return None

    @staticmethod
    def get_subdocument_references(graph: Graph) -> list[dict]:
        """Return all document reference nodes with their context."""
        refs = []
        for node in graph.get_document_nodes():
            # Find which decision/process node points to this reference
            parents = []
            for edge in graph.predecessors(node.node_id):
                parent = graph.get_node(edge.source)
                parents.append({"node_id": edge.source, "text": parent.text})
            refs.append({
                "node_id": node.node_id,
                "reference_text": node.text,
                "referenced_from": parents,
            })
        return refs

    @staticmethod
    def summarise_graph(graph: Graph) -> str:
        """
        Return a concise natural language summary of the graph structure.
        Used to build agent context in prompts.
        """
        n_decisions = len(graph.get_decision_nodes())
        n_process = len(graph.get_process_nodes())
        n_docs = len(graph.get_document_nodes())
        terminals = graph.get_terminal_nodes()

        lines = [
            f"Procedure: {graph.description}",
            f"Structure: {n_decisions} decision points, {n_process} action steps, "
            f"{n_docs} sub-procedure references.",
            f"Resolution endpoints: {len(terminals)}",
        ]

        esc = GraphUtils.get_escalation_node(graph)
        if esc:
            lines.append(f"Escalation: {esc.text}")

        return "\n".join(lines)
