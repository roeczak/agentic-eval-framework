"""
task5_trace_logging.py
-----------------------
Task 5: Outcome Reporting and Trace Logging (Action -> Reflection stage)

The agent receives:
  - The fault context
  - The completed diagnostic path (sequence of nodes visited, decisions made)
  - The resolution reached

The agent must generate a structured, auditable trace log that documents:
  1. The error identified and its classification
  2. Each decision point encountered and the Yes/No answer given
  3. Each action step performed
  4. Any sub-procedure references consulted
  5. The final resolution and escalation decision
  6. A brief justification for each decision

This task directly supports industrial audit and compliance requirements —
the trace log must be complete, accurate, and traceable back to the procedure.

Scoring (LLM-as-judge):
  - decision_coverage      : fraction of GT decision points documented
  - action_coverage        : fraction of GT process nodes documented
  - resolution_accuracy    : 1.0 if correct terminal/resolution documented
  - justification_quality  : LLM-as-judge score for reasoning quality (0-1)
  - trace_completeness     : overall coverage vs reference trace (primary metric)
  - hallucination_flag     : 1 if trace references non-existent nodes/steps
  - composite_score        : weighted combination

Scenario generation:
  One scenario per unique path — the completed path from Task 4 is the input.
  Ground truth trace is auto-generated from the path structure.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils, GraphPath
from tasks.base_task import BaseTask, Scenario, TaskResult


class Task5TraceLogging(BaseTask):
    """
    Task 5: Outcome Reporting and Trace Logging.

    Evaluates the agent's ability to generate a complete, accurate,
    and auditable trace log of a completed diagnostic procedure.
    """

    TASK_ID = "task5"

    WEIGHTS = {
        "trace_completeness": 0.40,
        "resolution_accuracy": 0.30,
        "justification_quality": 0.30,
    }

    @property
    def task_id(self) -> str:
        return self.TASK_ID

    # ------------------------------------------------------------------
    # Reference trace generation
    # ------------------------------------------------------------------

    def _build_reference_trace(
        self,
        graph: Graph,
        path: GraphPath,
    ) -> dict:
        """
        Build a structured reference trace from a graph path.
        This is the ground truth the agent's output is scored against.
        """
        decisions = []
        actions = []
        subdocs = []
        resolution = None

        start_id = graph.get_start_node().node_id

        for i, (node_id, incoming_label) in enumerate(path.steps):
            node = graph.get_node(node_id)

            if node_id == start_id:
                continue

            if node.is_decision():
                # The label on the NEXT step is the answer
                answer = None
                if i + 1 < len(path.steps):
                    answer = path.steps[i + 1][1]
                decisions.append({
                    "node_id": node_id,
                    "question": node.text,
                    "answer": answer,
                })

            elif node.is_process():
                actions.append({
                    "node_id": node_id,
                    "action": node.text,
                })

            elif node.is_document():
                subdocs.append({
                    "node_id": node_id,
                    "reference": node.text,
                })

            elif node.is_terminator():
                resolution = {
                    "node_id": node_id,
                    "text": node.text,
                }

        return {
            "decisions": decisions,
            "actions": actions,
            "subdoc_references": subdocs,
            "resolution": resolution,
            "total_steps": len(path) - 1,  # exclude start
        }

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def load_scenarios(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        existing = self.load_scenarios_from_disk(graph_ids)
        if existing:
            return existing
        return self.generate_scenarios(graph_ids)

    def generate_scenarios(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        all_graphs = self.loader.load_all()
        if graph_ids:
            graphs = {gid: g for gid, g in all_graphs.items() if gid in graph_ids}
        else:
            graphs = all_graphs

        scenarios = []
        for gid, graph in graphs.items():
            if graph.complexity == "linear":
                continue
            scenarios.extend(self._generate_for_graph(graph))
        return scenarios

    def _generate_for_graph(self, graph: Graph) -> list[Scenario]:
        paths = GraphUtils.enumerate_paths(graph)
        scenarios = []

        for path_idx, path in enumerate(paths):
            reference_trace = self._build_reference_trace(graph, path)

            # Build a structured summary of the completed path
            # (this is what the agent receives as input)
            completed_path_summary = self._summarise_path(graph, path)

            terminal_id = path.node_ids[-1]
            terminal = graph.get_node(terminal_id)

            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t5_{path_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "fault_context": (
                        f"Diagnostic procedure completed for: {graph.description}. "
                        f"Resolution reached: {terminal.text}"
                    ),
                    "completed_path_summary": completed_path_summary,
                    "graph_description": graph.description,
                },
                ground_truth={
                    "reference_trace": reference_trace,
                    "terminal_node_id": terminal_id,
                    "terminal_text": terminal.text,
                    "n_decisions": len(reference_trace["decisions"]),
                    "n_actions": len(reference_trace["actions"]),
                    "n_subdocs": len(reference_trace["subdoc_references"]),
                },
                metadata={
                    "complexity": graph.complexity,
                    "path_length": len(path),
                    "graph_description": graph.description,
                },
            ))

        return scenarios

    def _summarise_path(self, graph: Graph, path: GraphPath) -> str:
        """
        Build a natural language summary of the completed diagnostic path.
        This is given to the agent as input for trace generation.
        """
        lines = ["The following diagnostic steps were completed:"]
        start_id = graph.get_start_node().node_id
        step_num = 1

        for i, (node_id, incoming_label) in enumerate(path.steps):
            node = graph.get_node(node_id)
            if node_id == start_id:
                continue
            label_str = f"[{incoming_label}] " if incoming_label else ""
            lines.append(
                f"  {step_num}. {label_str}[{node_id}] "
                f"({node.node_type}): {node.text[:80]}"
            )
            step_num += 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        system = self._load_system_prompt()
        fault_context = scenario.input["fault_context"]
        path_summary = scenario.input["completed_path_summary"]

        prompt = f"""{system}

---

FAULT CONTEXT:
{fault_context}

COMPLETED DIAGNOSTIC PATH:
{path_summary}

---

TASK:
Generate a complete, structured, auditable trace log of the diagnostic
session described above. The trace log must document:

1. Every decision point encountered and the Yes/No answer given
2. Every action step performed
3. Any sub-procedure references consulted
4. The final resolution reached
5. A brief justification for each decision

The trace log will be used for industrial audit and compliance purposes.
Every step must be traceable back to the procedure.

Respond ONLY with a valid JSON object matching this exact template:

{{
  "procedure_id": "<graph ID>",
  "fault_summary": "<one sentence describing the fault>",
  "decisions": [
    {{
      "node_id": "<node ID>",
      "question": "<decision question text>",
      "answer": "<Yes or No>",
      "justification": "<one sentence explaining this answer>"
    }}
  ],
  "actions_performed": [
    {{
      "node_id": "<node ID>",
      "action": "<action text>",
      "completed": <true or false>
    }}
  ],
  "subdoc_references": [
    {{
      "node_id": "<node ID>",
      "reference": "<sub-procedure name>"
    }}
  ],
  "resolution": {{
    "node_id": "<terminal node ID>",
    "text": "<resolution text>",
    "escalation_required": <true or false>
  }},
  "trace_complete": <true or false>
}}"""
        return prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        result: TaskResult,
        scenario: Scenario,
        llm_judge=None,
    ) -> dict:
        """
        Score a Task 5 result.

        llm_judge: optional callable that takes (agent_trace, reference_trace)
                   and returns a float 0-1 for justification quality.
                   If None, justification_quality defaults to 0.5.
        """
        gt = scenario.ground_truth
        parsed = result.parsed_output
        ref_trace = gt["reference_trace"]

        if parsed is None:
            return self._zero_scores()

        # 1. Decision coverage
        gt_decisions = ref_trace["decisions"]
        pred_decisions = parsed.get("decisions", [])
        decision_coverage = self._compute_coverage(
            gt_items=[d["node_id"] for d in gt_decisions],
            pred_items=[d.get("node_id", "") for d in pred_decisions],
        )

        # 2. Action coverage
        gt_actions = ref_trace["actions"]
        pred_actions = parsed.get("actions_performed", [])
        action_coverage = self._compute_coverage(
            gt_items=[a["node_id"] for a in gt_actions],
            pred_items=[a.get("node_id", "") for a in pred_actions],
        )

        # 3. Resolution accuracy
        gt_terminal = gt["terminal_node_id"]
        pred_resolution = parsed.get("resolution", {})
        pred_terminal = pred_resolution.get("node_id", "")
        resolution_accuracy = 1.0 if pred_terminal == gt_terminal else 0.0

        # 4. Trace completeness — weighted average of coverage scores
        n_gt_items = (len(gt_decisions) + len(gt_actions)
                      + len(ref_trace["subdoc_references"]))
        if n_gt_items > 0:
            subdoc_coverage = self._compute_coverage(
                gt_items=[s["node_id"] for s in ref_trace["subdoc_references"]],
                pred_items=[s.get("node_id", "")
                           for s in parsed.get("subdoc_references", [])],
            )
            trace_completeness = (
                len(gt_decisions) * decision_coverage
                + len(gt_actions) * action_coverage
                + len(ref_trace["subdoc_references"]) * subdoc_coverage
            ) / n_gt_items
        else:
            trace_completeness = 1.0

        # 5. Justification quality (LLM-as-judge or default)
        if llm_judge is not None:
            justification_quality = llm_judge(parsed, ref_trace)
        else:
            # Heuristic: check that justifications are non-empty
            pred_dec_list = parsed.get("decisions", [])
            if pred_dec_list:
                non_empty = sum(
                    1 for d in pred_dec_list
                    if d.get("justification", "").strip()
                )
                justification_quality = non_empty / len(pred_dec_list)
            else:
                justification_quality = 0.0

        # 6. Composite
        composite = (
            self.WEIGHTS["trace_completeness"] * trace_completeness
            + self.WEIGHTS["resolution_accuracy"] * resolution_accuracy
            + self.WEIGHTS["justification_quality"] * justification_quality
        )

        # 7. Hallucination flag
        hallucination = self._check_hallucination(parsed, scenario)

        return {
            "decision_coverage": round(decision_coverage, 4),
            "action_coverage": round(action_coverage, 4),
            "trace_completeness": round(trace_completeness, 4),
            "resolution_accuracy": resolution_accuracy,
            "justification_quality": round(justification_quality, 4),
            "composite_score": round(composite, 4),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _compute_coverage(
        self,
        gt_items: list[str],
        pred_items: list[str],
    ) -> float:
        """Fraction of GT items present in predicted items."""
        if not gt_items:
            return 1.0
        pred_set = set(pred_items)
        matched = sum(1 for item in gt_items if item in pred_set)
        return matched / len(gt_items)

    def _zero_scores(self) -> dict:
        return {
            "decision_coverage": 0.0,
            "action_coverage": 0.0,
            "trace_completeness": 0.0,
            "resolution_accuracy": 0.0,
            "justification_quality": 0.0,
            "composite_score": 0.0,
            "hallucination_flag": 0,
            "parse_success": 0,
        }

    def _check_hallucination(
        self,
        parsed: dict,
        scenario: Scenario,
    ) -> int:
        """Return 1 if agent references non-existent node IDs in trace."""
        graph = self.loader.load(scenario.graph_id)
        valid_ids = set(graph.nodes.keys())

        # Check all node_id fields in the trace
        for decision in parsed.get("decisions", []):
            if decision.get("node_id", "") not in valid_ids:
                return 1
        for action in parsed.get("actions_performed", []):
            if action.get("node_id", "") not in valid_ids:
                return 1
        for subdoc in parsed.get("subdoc_references", []):
            if subdoc.get("node_id", "") not in valid_ids:
                return 1
        resolution = parsed.get("resolution", {})
        if resolution.get("node_id", "") not in valid_ids:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Aggregate scoring
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_scores(results: list[TaskResult]) -> dict:
        if not results:
            return {}
        scored = [r for r in results if r.scores]
        if not scored:
            return {}

        metrics = [
            "decision_coverage", "action_coverage",
            "trace_completeness", "resolution_accuracy",
            "justification_quality", "composite_score",
            "hallucination_flag", "parse_success",
        ]
        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated[f"mean_{metric}"] = round(sum(values) / len(values), 4)

        for tier in ["short_tree", "medium_chain", "deep_ladder"]:
            tier_results = [r for r in scored
                           if r.metadata.get("complexity") == tier]
            if tier_results:
                vals = [r.scores.get("composite_score", 0.0)
                        for r in tier_results]
                aggregated[f"mean_composite_{tier}"] = round(
                    sum(vals) / len(vals), 4
                )
                aggregated[f"n_{tier}"] = len(tier_results)

        aggregated["n_scenarios"] = len(scored)
        return aggregated
