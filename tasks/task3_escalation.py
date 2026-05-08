"""
task3_escalation.py
--------------------
Task 3: Role-Based Reasoning and Escalation (Planning stage)

The agent receives:
  - A fault scenario with a stated user role
  - The full diagnostic procedure graph
  - The diagnosed error and terminal node (from Task 1/2, or standalone)

The agent must:
  1. Determine whether the stated role is appropriate to handle the situation
  2. If not, identify the correct role to escalate to
  3. If the situation is out of scope for all operator roles, flag for
     Maintenance Engineer (Level 4) — the final human escalation point

Three scenario cases per graph path:
  Case A — No escalation needed: user role matches required role
  Case B — Escalation required: user role is too low, escalate to correct level
  Case C — Out of scope: situation requires Maintenance Engineer (Level 4)

Scoring:
  - escalation_decision_correct : 1.0 if escalate/no-escalate decision is right
  - target_role_correct         : 1.0 if correct escalation target identified
  - out_of_scope_detection      : 1.0 if correctly flags Level 4 situations
  - composite_score             : weighted average
  - hallucination_flag          : 1 if agent references invalid roles/nodes
"""

from __future__ import annotations

import re
from typing import Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils
from framework.roles import RoleHierarchy, ROLES
from tasks.base_task import BaseTask, Scenario, TaskResult


class Task3Escalation(BaseTask):
    """
    Task 3: Role-Based Reasoning and Escalation.

    Evaluates whether the agent correctly determines escalation need
    and identifies the appropriate authority level.
    """

    TASK_ID = "task3"

    WEIGHTS = {
        "escalation_decision": 0.40,
        "target_role": 0.35,
        "out_of_scope_detection": 0.25,
    }

    # Case labels
    CASE_NO_ESCALATION = "no_escalation"
    CASE_ESCALATION    = "escalation"
    CASE_OUT_OF_SCOPE  = "out_of_scope"

    @property
    def task_id(self) -> str:
        return self.TASK_ID

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    def _infer_required_role(self, graph: Graph, terminal_node_id: str) -> int:
        """
        Infer the required role level for a given terminal node.

        Heuristic based on terminal node text:
          - Simple QA notification / batch change → Level 1 (Operator)
          - Technical check / minor intervention   → Level 2 (Technical Operator)
          - Maintenance team / mechanic action     → Level 3 (Mechanic)
          - Maintenance engineer / critical fault  → Level 4 (Maintenance Engineer)
        """
        terminal = graph.get_node(terminal_node_id)
        text = terminal.text.lower()

        if any(x in text for x in [
            "maintenance engineer", "escalate to maintenance engineer",
            "critical", "clear work cell", "structural"
        ]):
            return 4
        elif any(x in text for x in [
            "maintenance team", "maintenance technician",
            "replace", "repair", "retrain", "realign", "correct the"
        ]):
            return 3
        elif any(x in text for x in [
            "technical", "adjust", "calibrate", "diagnose",
            "inspect the conveyor", "check the pressing"
        ]):
            return 2
        else:
            # Default: operator-level (notify QA, batch change, resume production)
            return 1

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
        scenarios = []
        scenario_idx = 0

        for terminal in graph.get_terminal_nodes():
            required_level = self._infer_required_role(graph, terminal.node_id)
            required_role = ROLES[required_level]

            # Derive fault description from terminal text + graph description
            fault_description = (
                f"During operation involving '{graph.description}', "
                f"the following situation was reached: {terminal.text}"
            )

            # Case A: No escalation — user role matches required
            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t3_{scenario_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "fault_description": fault_description,
                    "terminal_node_id": terminal.node_id,
                    "terminal_text": terminal.text,
                    "user_role_level": required_level,
                    "user_role_name": required_role.name,
                    "case_type": self.CASE_NO_ESCALATION,
                },
                ground_truth={
                    "escalation_needed": False,
                    "required_role_level": required_level,
                    "required_role_name": required_role.name,
                    "escalation_target_level": None,
                    "escalation_target_name": None,
                    "out_of_scope": required_level >= 4,
                },
                metadata={
                    "complexity": graph.complexity,
                    "case_type": self.CASE_NO_ESCALATION,
                    "required_level": required_level,
                    "graph_description": graph.description,
                },
            ))
            scenario_idx += 1

            # Case B: Escalation needed — user role is one level below required
            if required_level > 1:
                user_level = required_level - 1
                user_role = ROLES[user_level]
                scenarios.append(Scenario(
                    scenario_id=f"{graph.graph_id}_t3_{scenario_idx:03d}",
                    graph_id=graph.graph_id,
                    task_id=self.TASK_ID,
                    input={
                        "fault_description": fault_description,
                        "terminal_node_id": terminal.node_id,
                        "terminal_text": terminal.text,
                        "user_role_level": user_level,
                        "user_role_name": user_role.name,
                        "case_type": self.CASE_ESCALATION,
                    },
                    ground_truth={
                        "escalation_needed": True,
                        "required_role_level": required_level,
                        "required_role_name": required_role.name,
                        "escalation_target_level": required_level,
                        "escalation_target_name": required_role.name,
                        "out_of_scope": False,
                    },
                    metadata={
                        "complexity": graph.complexity,
                        "case_type": self.CASE_ESCALATION,
                        "required_level": required_level,
                        "user_level": user_level,
                        "graph_description": graph.description,
                    },
                ))
                scenario_idx += 1

            # Case C: Out of scope — always present as Level 1 user on Level 4 task
            if required_level >= 4:
                user_role = ROLES[1]
                scenarios.append(Scenario(
                    scenario_id=f"{graph.graph_id}_t3_{scenario_idx:03d}",
                    graph_id=graph.graph_id,
                    task_id=self.TASK_ID,
                    input={
                        "fault_description": fault_description,
                        "terminal_node_id": terminal.node_id,
                        "terminal_text": terminal.text,
                        "user_role_level": 1,
                        "user_role_name": user_role.name,
                        "case_type": self.CASE_OUT_OF_SCOPE,
                    },
                    ground_truth={
                        "escalation_needed": True,
                        "required_role_level": 4,
                        "required_role_name": "Maintenance Engineer",
                        "escalation_target_level": 4,
                        "escalation_target_name": "Maintenance Engineer",
                        "out_of_scope": True,
                    },
                    metadata={
                        "complexity": graph.complexity,
                        "case_type": self.CASE_OUT_OF_SCOPE,
                        "required_level": 4,
                        "user_level": 1,
                        "graph_description": graph.description,
                    },
                ))
                scenario_idx += 1

        return scenarios

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        system = self._load_system_prompt()
        graph_context = self._get_graph_context(graph)

        user_role_level = scenario.input["user_role_level"]
        user_role_name  = scenario.input["user_role_name"]
        fault           = scenario.input["fault_description"]
        terminal_text   = scenario.input["terminal_text"]

        role_hierarchy = RoleHierarchy.role_summary()

        prompt = f"""{system}

---

ROLE HIERARCHY:
{role_hierarchy}

---

DIAGNOSTIC PROCEDURE:
{graph_context}

---

SITUATION:
{fault}

Current resolution point reached:
"{terminal_text}"

The worker reporting this situation is:
  Role: {user_role_name} (Level {user_role_level})

---

TASK:
Based on the situation and the role hierarchy:

1. Determine whether the current worker role ({user_role_name}, Level {user_role_level})
   is authorised to handle this resolution.
2. If not, identify the correct role to escalate to.
3. If the situation exceeds all operator-level competence, flag it as out of scope
   and escalate to the Maintenance Engineer (Level 4).

Respond ONLY with a valid JSON object matching this exact template:

{{
  "escalation_needed": <true or false>,
  "required_role_level": <integer 1-4>,
  "required_role_name": "<role name>",
  "escalation_target_level": <integer 1-4 or null if no escalation>,
  "escalation_target_name": "<role name or null if no escalation>",
  "out_of_scope": <true if Maintenance Engineer required, false otherwise>,
  "reasoning": "<one sentence justifying the escalation decision>"
}}"""
        return prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, result: TaskResult, scenario: Scenario) -> dict:
        gt = scenario.ground_truth
        parsed = result.parsed_output
        case_type = scenario.input.get("case_type", self.CASE_NO_ESCALATION)

        if parsed is None:
            return {
                "escalation_decision_correct": 0.0,
                "target_role_correct": 0.0,
                "out_of_scope_detection": 0.0,
                "composite_score": 0.0,
                "hallucination_flag": 0,
                "parse_success": 0,
            }

        # 1. Escalation decision correct
        pred_esc = parsed.get("escalation_needed", None)
        gt_esc   = gt["escalation_needed"]
        esc_correct = 1.0 if pred_esc == gt_esc else 0.0

        # 2. Target role correct
        pred_target_level = parsed.get("escalation_target_level")
        gt_target_level   = gt["escalation_target_level"]

        if gt_target_level is None:
            # No escalation expected — correct if agent also says no target
            target_correct = 1.0 if pred_target_level is None else 0.0
        else:
            target_correct = 1.0 if pred_target_level == gt_target_level else 0.0

        # 3. Out of scope detection
        pred_oos = parsed.get("out_of_scope", False)
        gt_oos   = gt["out_of_scope"]
        if case_type == self.CASE_OUT_OF_SCOPE:
            oos_score = 1.0 if pred_oos == gt_oos else 0.0
        else:
            # Not an out-of-scope case — penalise false positives
            oos_score = 1.0 if not pred_oos else 0.0

        # 4. Composite
        composite = (
            self.WEIGHTS["escalation_decision"] * esc_correct
            + self.WEIGHTS["target_role"] * target_correct
            + self.WEIGHTS["out_of_scope_detection"] * oos_score
        )

        # 5. Hallucination — invalid role level or node references
        hallucination = self._check_hallucination(parsed, scenario)

        return {
            "escalation_decision_correct": esc_correct,
            "target_role_correct": target_correct,
            "out_of_scope_detection": oos_score,
            "composite_score": round(composite, 4),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _check_hallucination(
        self,
        parsed: dict,
        scenario: Scenario,
    ) -> int:
        """Return 1 if agent references invalid role levels or node IDs."""
        # Check role level is within valid range 1-4
        pred_level = parsed.get("required_role_level")
        if pred_level is not None:
            try:
                if int(pred_level) not in range(1, 5):
                    return 1
            except (ValueError, TypeError):
                return 1

        pred_target = parsed.get("escalation_target_level")
        if pred_target is not None:
            try:
                if int(pred_target) not in range(1, 5):
                    return 1
            except (ValueError, TypeError):
                return 1

        # Check for hallucinated node references in reasoning
        graph = self.loader.load(scenario.graph_id)
        valid_node_ids = set(graph.nodes.keys())
        reasoning = parsed.get("reasoning", "")
        referenced = re.findall(r'\bbox_\d+\b', reasoning)
        for ref in referenced:
            if ref not in valid_node_ids:
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
            "escalation_decision_correct",
            "target_role_correct",
            "out_of_scope_detection",
            "composite_score",
            "hallucination_flag",
            "parse_success",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated[f"mean_{metric}"] = round(sum(values) / len(values), 4)

        # Per case type breakdown
        for case in [Task3Escalation.CASE_NO_ESCALATION,
                     Task3Escalation.CASE_ESCALATION,
                     Task3Escalation.CASE_OUT_OF_SCOPE]:
            case_results = [r for r in scored
                           if r.metadata.get("case_type") == case]
            if case_results:
                vals = [r.scores.get("composite_score", 0.0)
                        for r in case_results]
                aggregated[f"mean_composite_{case}"] = round(
                    sum(vals) / len(vals), 4
                )
                aggregated[f"n_{case}"] = len(case_results)

        aggregated["n_scenarios"] = len(scored)
        return aggregated
