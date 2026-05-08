"""
task1_error_identification.py
------------------------------
Task 1: Error Identification and Classification (Perception stage)

The agent receives:
  - A symptom description (natural language, as an operator would report it)
  - The full diagnostic procedure graph (in raw JSON or NL mode)

The agent must:
  1. Identify the error type/category from the symptom description
  2. Map the symptom to the correct entry decision node in the graph
  3. Determine the initial Yes/No answer for that decision node

Scoring:
  - error_classification_score : 1.0 if error type matches ground truth, 0.0 otherwise
  - branch_mapping_score       : 1.0 if correct entry decision node identified
  - initial_decision_score     : 1.0 if correct Yes/No at entry node
  - composite_score            : weighted average of the three above
  - hallucination_flag         : 1 if agent references non-existent nodes/steps, 0 otherwise

Scenario generation strategy:
  - For each graph, generate scenarios from each terminal node backwards:
    each terminal defines a concrete "what happened" situation
  - Symptom descriptions are derived from the path leading to that terminal
  - This ensures ground truth is grounded in actual graph structure
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils, GraphPath
from tasks.base_task import BaseTask, Scenario, TaskResult, GraphSerializer


# ---------------------------------------------------------------------------
# Task 1 implementation
# ---------------------------------------------------------------------------

class Task1ErrorIdentification(BaseTask):
    """
    Task 1: Error Identification and Classification.

    Evaluates the agent's ability to:
      - Classify an error from a symptom description
      - Map it to the correct branch of the diagnostic flowchart
      - Provide the correct initial Yes/No decision
    """

    TASK_ID = "task1"

    # Weights for composite score
    WEIGHTS = {
        "error_classification": 0.35,
        "branch_mapping": 0.40,
        "initial_decision": 0.25,
    }

    @property
    def task_id(self) -> str:
        return self.TASK_ID

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def load_scenarios(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        """
        Load scenarios from disk if available, otherwise generate them.
        """
        existing = self.load_scenarios_from_disk(graph_ids)
        if existing:
            return existing
        return self.generate_scenarios(graph_ids)

    def generate_scenarios(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        """
        Generate Task 1 scenarios from GRAPH documents.

        Strategy:
          For each graph, enumerate all paths from start to terminal nodes.
          For each path, derive a symptom description from the terminal node
          text and the sequence of decision nodes along the path.
          Ground truth = error category + entry decision node + initial answer.
        """
        all_graphs = self.loader.load_all()
        if graph_ids:
            graphs = {gid: all_graphs[gid] for gid in graph_ids if gid in all_graphs}
        else:
            graphs = all_graphs

        scenarios = []
        for gid, graph in graphs.items():
            # Skip linear graphs (no decision nodes — nothing to classify)
            if graph.complexity == "linear":
                continue
            graph_scenarios = self._generate_for_graph(graph)
            scenarios.extend(graph_scenarios)

        return scenarios

    def _generate_for_graph(self, graph: Graph) -> list[Scenario]:
        """Generate scenarios for a single graph."""
        paths = GraphUtils.enumerate_paths(graph)
        scenarios = []

        for path_idx, path in enumerate(paths):
            # Get terminal node for this path
            terminal_id = path.node_ids[-1]
            terminal = graph.get_node(terminal_id)

            # Get entry decision node (first decision in path)
            entry_decision_id = None
            entry_decision_label = None
            for i, (node_id, label) in enumerate(path.steps):
                node = graph.get_node(node_id)
                if node.is_decision():
                    entry_decision_id = node_id
                    # The label on the NEXT step is the answer to this decision
                    if i + 1 < len(path.steps):
                        entry_decision_label = path.steps[i + 1][1]
                    break

            if entry_decision_id is None:
                continue

            # Derive symptom from terminal node and graph description
            symptom = self._derive_symptom(graph, path, terminal)

            # Error category from graph description
            error_category = self._extract_error_category(graph)

            scenario = Scenario(
                scenario_id=f"{graph.graph_id}_t1_{path_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "symptom_description": symptom,
                    "operator_role": "Operator",
                },
                ground_truth={
                    "error_category": error_category,
                    "entry_decision_node": entry_decision_id,
                    "entry_decision_text": graph.get_node(entry_decision_id).text,
                    "initial_decision": entry_decision_label,
                    "resolution_path": path.node_ids,
                    "terminal_node": terminal_id,
                    "terminal_text": terminal.text,
                },
                metadata={
                    "complexity": graph.complexity,
                    "graph_description": graph.description,
                    "path_length": len(path),
                    "num_decisions": len(path.decision_labels),
                },
            )
            scenarios.append(scenario)

        return scenarios

    def _derive_symptom(
        self,
        graph: Graph,
        path: GraphPath,
        terminal: "GraphNode",
    ) -> str:
        """
        Derive a realistic symptom description from the terminal node
        and the graph description.

        The symptom is what an operator would report — not the procedure
        diagnosis, but the observable problem.
        """
        # Use the terminal text as the basis for the symptom
        # Strip resolution instructions to leave only the observable condition
        terminal_text = terminal.text

        # Extract observable keywords from graph description
        description = graph.description.lower()

        # Build symptom as "I am observing X" framed from operator perspective
        symptom = (
            f"During production, the following issue was observed: "
            f"{terminal_text}. "
            f"Context: {graph.description}"
        )
        return symptom

    def _extract_error_category(self, graph: Graph) -> str:
        """
        Extract a normalised error category label from the graph description.
        Used as ground truth for the error classification sub-score.
        """
        desc = graph.description.lower()
        # Simple keyword-based categorisation
        if "marking" in desc:
            return "marking_defect"
        elif "insert" in desc or "press" in desc or "pressing" in desc:
            return "press_fit_defect"
        elif "cap" in desc:
            return "cap_handling_defect"
        elif "vision" in desc or "camera" in desc:
            return "vision_system_fault"
        elif "conveyor" in desc or "supply" in desc:
            return "supply_system_fault"
        elif "sensor" in desc:
            return "sensor_fault"
        elif "damage" in desc or "damaged" in desc:
            return "part_damage"
        elif "material" in desc:
            return "material_defect"
        elif "dimension" in desc or "measurement" in desc:
            return "dimensional_deviation"
        elif "robot" in desc:
            return "robot_fault"
        else:
            return "general_fault"

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        """
        Build the Task 1 prompt.

        Structure:
          1. System context
          2. Graph procedure (serialised per context_mode)
          3. Symptom description
          4. Output format instructions
        """
        system = self._load_system_prompt()
        graph_context = self._get_graph_context(graph)
        symptom = scenario.input["symptom_description"]
        role = scenario.input.get("operator_role", "Operator")

        prompt = f"""{system}

---

You are evaluating a fault reported by a {role} in a manufacturing environment.

DIAGNOSTIC PROCEDURE:
{graph_context}

---

REPORTED SYMPTOM:
{symptom}

---

TASK:
Based on the reported symptom and the diagnostic procedure above:

1. Identify the ERROR CATEGORY that best describes this fault.
2. Identify the ENTRY DECISION NODE — the first decision point in the procedure
   most relevant to diagnosing this symptom (provide the node ID).
3. Provide the INITIAL DECISION — your Yes or No answer to that entry decision node,
   based on the reported symptom.

Respond ONLY with a valid JSON object matching this exact template:

{{
  "error_category": "<short label describing the fault type>",
  "entry_decision_node": "<node ID, e.g. box_1>",
  "entry_decision_text": "<the question text of that decision node>",
  "initial_decision": "<Yes or No>",
  "reasoning": "<one sentence explaining your classification>"
}}"""
        return prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, result: TaskResult, scenario: Scenario) -> dict:
        """
        Compute Task 1 scores for a single result.

        Returns:
          error_classification_score : 0.0 or 1.0
          branch_mapping_score       : 0.0 or 1.0
          initial_decision_score     : 0.0 or 1.0
          composite_score            : weighted average
          hallucination_flag         : 0 or 1
          parse_success              : 0 or 1
        """
        gt = scenario.ground_truth
        parsed = result.parsed_output

        # If parsing failed, all scores are 0
        if parsed is None:
            return {
                "error_classification_score": 0.0,
                "branch_mapping_score": 0.0,
                "initial_decision_score": 0.0,
                "composite_score": 0.0,
                "hallucination_flag": 0,
                "parse_success": 0,
            }

        # 1. Error classification score
        pred_category = parsed.get("error_category", "").lower().strip()
        gt_category = gt["error_category"].lower().strip()
        error_score = 1.0 if pred_category == gt_category else 0.0

        # 2. Branch mapping score — did the agent identify the correct entry node?
        pred_node = parsed.get("entry_decision_node", "").strip()
        gt_node = gt["entry_decision_node"].strip()
        branch_score = 1.0 if pred_node == gt_node else 0.0

        # 3. Initial decision score
        pred_decision = parsed.get("initial_decision", "").strip().capitalize()
        gt_decision = str(gt["initial_decision"]).strip().capitalize()
        decision_score = 1.0 if pred_decision == gt_decision else 0.0

        # 4. Composite score
        composite = (
            self.WEIGHTS["error_classification"] * error_score
            + self.WEIGHTS["branch_mapping"] * branch_score
            + self.WEIGHTS["initial_decision"] * decision_score
        )

        # 5. Hallucination flag — agent referenced a non-existent node ID
        hallucination = self._check_hallucination(parsed, result, scenario)

        return {
            "error_classification_score": error_score,
            "branch_mapping_score": branch_score,
            "initial_decision_score": decision_score,
            "composite_score": round(composite, 4),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _check_hallucination(
        self,
        parsed: dict,
        result: TaskResult,
        scenario: Scenario,
    ) -> int:
        """
        Return 1 if the agent's output contains hallucinated node references
        (node IDs that don't exist in the graph), 0 otherwise.
        """
        graph = self.loader.load(scenario.graph_id)
        valid_node_ids = set(graph.nodes.keys())

        # Check entry_decision_node
        pred_node = parsed.get("entry_decision_node", "")
        if pred_node and pred_node not in valid_node_ids:
            return 1

        # Check if reasoning references non-existent nodes
        reasoning = parsed.get("reasoning", "")
        import re
        referenced_nodes = re.findall(r'\bbox_\d+\b', reasoning)
        for ref in referenced_nodes:
            if ref not in valid_node_ids:
                return 1

        return 0

    # ------------------------------------------------------------------
    # Aggregate scoring (across multiple results)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_scores(results: list[TaskResult]) -> dict:
        """
        Aggregate scores across all results.
        Returns mean of each metric and overall task performance.
        """
        if not results:
            return {}

        scored = [r for r in results if r.scores]
        if not scored:
            return {}

        metrics = [
            "error_classification_score",
            "branch_mapping_score",
            "initial_decision_score",
            "composite_score",
            "hallucination_flag",
            "parse_success",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated[f"mean_{metric}"] = round(sum(values) / len(values), 4)

        aggregated["n_scenarios"] = len(scored)
        return aggregated
