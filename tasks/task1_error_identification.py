"""
task1_error_identification.py
------------------------------
Task 1: Error Identification and Classification (Perception stage)

CHANGES FROM v2:
  - Removed error_classification_score (ground truth leakage)
  - Symptom derived from process nodes and terminal text only
  - Fallback for graphs starting directly with decision node
  - Leading question words stripped from fallback symptom
  - Equal weighting of branch_mapping and initial_decision
  - Composite = Error Identification Accuracy (EIA)
"""

from __future__ import annotations

import re
from typing import Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils, GraphPath
from tasks.base_task import BaseTask, Scenario, TaskResult, GraphSerializer


class Task1ErrorIdentification(BaseTask):

    TASK_ID = "task1"

    WEIGHTS = {
        "branch_mapping": 0.50,
        "initial_decision": 0.50,
    }

    QUESTION_STARTERS = (
        "is ", "are ", "was ", "were ", "does ", "do ",
        "did ", "has ", "have ", "had ", "can ", "could ",
        "should ", "would ", "will ",
    )

    @property
    def task_id(self):
        return self.TASK_ID

    def load_scenarios(self, graph_ids=None):
        existing = self.load_scenarios_from_disk(graph_ids)
        if existing:
            return existing
        return self.generate_scenarios(graph_ids)

    def generate_scenarios(self, graph_ids=None):
        all_graphs = self.loader.load_all()
        if graph_ids:
            graphs = {gid: all_graphs[gid]
                      for gid in graph_ids if gid in all_graphs}
        else:
            graphs = all_graphs

        scenarios = []
        for gid, graph in graphs.items():
            if graph.complexity == "linear":
                continue
            scenarios.extend(self._generate_for_graph(graph))
        return scenarios

    def _generate_for_graph(self, graph):
        paths = GraphUtils.enumerate_paths(graph)
        scenarios = []

        for path_idx, path in enumerate(paths):
            terminal_id = path.node_ids[-1]
            terminal = graph.get_node(terminal_id)

            entry_decision_id = None
            entry_decision_label = None
            for i, (node_id, label) in enumerate(path.steps):
                node = graph.get_node(node_id)
                if node.is_decision():
                    entry_decision_id = node_id
                    if i + 1 < len(path.steps):
                        entry_decision_label = path.steps[i + 1][1]
                    break

            if entry_decision_id is None:
                continue

            symptom = self._derive_symptom(graph, path, terminal)

            scenarios.append(Scenario(
                scenario_id=graph.graph_id + "_t1_" + str(path_idx).zfill(3),
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "symptom_description": symptom,
                    "operator_role": "Operator",
                },
                ground_truth={
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
            ))

        return scenarios

    def _strip_question_starter(self, text):
        """Strip leading question words so text follows 'whether' naturally."""
        text_lower = text.lower()
        for starter in self.QUESTION_STARTERS:
            if text_lower.startswith(starter):
                return text[len(starter):]
        return text

    def _derive_symptom(self, graph, path, terminal):
        """
        Derive operator-facing symptom from observable process nodes
        and terminal outcome — no graph description to avoid leakage.

        For graphs starting directly with a decision node (no preceding
        process nodes), falls back to rephrasing the first decision
        question as an observable condition, stripping leading question
        words so the phrasing is grammatically correct after 'whether'.
        """
        process_texts = []
        for node_id, _ in path.steps:
            node = graph.get_node(node_id)
            if node.is_decision():
                break
            if node.is_process():
                process_texts.append(node.text)
            if len(process_texts) >= 2:
                break

        terminal_text = terminal.text

        if process_texts:
            return (
                "During production, the following was observed: "
                + process_texts[0]
                + ". The situation has led to: "
                + terminal_text
            )

        first_decision = None
        for node_id, _ in path.steps:
            node = graph.get_node(node_id)
            if node.is_decision():
                first_decision = node
                break

        if first_decision:
            decision_text = first_decision.text.rstrip("?")
            decision_stripped = self._strip_question_starter(decision_text)
            return (
                "During production, an issue was detected. "
                "The operator needs to assess whether "
                + decision_stripped
                + ". The situation has led to: "
                + terminal_text
            )

        return (
            "During production, the following situation "
            "was reached: " + terminal_text
        )

    def build_prompt(self, scenario, graph):
        system = self._load_system_prompt()
        graph_context = self._get_graph_context(graph)
        symptom = scenario.input["symptom_description"]
        role = scenario.input.get("operator_role", "Operator")

        prompt = (
            system
            + "\n\n---\n\n"
            + "You are evaluating a fault reported by a "
            + role
            + " in a manufacturing environment.\n\n"
            + "DIAGNOSTIC PROCEDURE:\n"
            + graph_context
            + "\n\n---\n\n"
            + "REPORTED SYMPTOM:\n"
            + symptom
            + "\n\n---\n\n"
            + "TASK:\n"
            + "Based on the reported symptom and the diagnostic "
            + "procedure above:\n\n"
            + "1. Identify the ENTRY DECISION NODE - the FIRST Yes/No "
            + "decision point in the procedure where diagnosis begins. "
            + "This is the decision node that appears earliest in the "
            + "procedure flow, closest to the Start node. "
            + "Provide its node ID (e.g. box_1).\n\n"
            + "2. Provide the INITIAL DECISION - your Yes or No answer "
            + "to that first decision node, based on the reported "
            + "symptom.\n\n"
            + "Respond ONLY with a valid JSON object:\n\n"
            + "{\n"
            + '  "entry_decision_node": '
            + '"<node ID of the FIRST decision node>",\n'
            + '  "entry_decision_text": '
            + '"<the question text of that decision node>",\n'
            + '  "initial_decision": "<Yes or No>",\n'
            + '  "reasoning": "<one sentence explaining your assessment>"\n'
            + "}"
        )
        return prompt

    def score(self, result, scenario):
        gt = scenario.ground_truth
        parsed = result.parsed_output

        if parsed is None:
            return {
                "branch_mapping_score": 0.0,
                "initial_decision_score": 0.0,
                "composite_score": 0.0,
                "hallucination_flag": 0,
                "parse_success": 0,
            }

        pred_node = parsed.get("entry_decision_node", "").strip()
        gt_node = gt["entry_decision_node"].strip()
        branch_score = 1.0 if pred_node == gt_node else 0.0

        pred_decision = (
            parsed.get("initial_decision", "")
            .strip()
            .capitalize()
            .rstrip(".")
        )
        gt_decision = str(gt["initial_decision"]).strip().capitalize()
        decision_score = 1.0 if pred_decision == gt_decision else 0.0

        composite = (
            self.WEIGHTS["branch_mapping"] * branch_score
            + self.WEIGHTS["initial_decision"] * decision_score
        )

        hallucination = self._check_hallucination(parsed, result, scenario)

        return {
            "branch_mapping_score": branch_score,
            "initial_decision_score": decision_score,
            "composite_score": round(composite, 4),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _check_hallucination(self, parsed, result, scenario):
        graph = self.loader.load(scenario.graph_id)
        valid_node_ids = set(graph.nodes.keys())

        pred_node = parsed.get("entry_decision_node", "")
        if pred_node and pred_node not in valid_node_ids:
            return 1

        reasoning = parsed.get("reasoning", "")
        referenced = re.findall(r'\bbox_\d+\b', reasoning)
        for ref in referenced:
            if ref not in valid_node_ids:
                return 1

        return 0

    @staticmethod
    def aggregate_scores(results):
        if not results:
            return {}
        scored = [r for r in results if r.scores]
        if not scored:
            return {}

        metrics = [
            "branch_mapping_score",
            "initial_decision_score",
            "composite_score",
            "hallucination_flag",
            "parse_success",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated["mean_" + metric] = round(
                sum(values) / len(values), 4
            )

        aggregated["n_scenarios"] = len(scored)
        return aggregated
