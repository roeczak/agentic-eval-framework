"""
task2_instruction_retrieval.py
-------------------------------
Task 2: Instruction Retrieval and Clarification (Perception -> Planning)

The agent receives:
  - A user query describing what they need to do (incomplete or ambiguous)
  - The full diagnostic procedure graph
  - Optionally: a partially completed diagnostic state (for pipeline mode)

The agent must:
  1. Identify the relevant process node(s) that answer the query
  2. Retrieve the correct instruction text
  3. If the relevant branch leads to a sub-procedure reference (document node),
     recognise this and indicate that a sub-procedure must be consulted
  4. If the query is ambiguous, ask a targeted clarification question

Scoring:
  - retrieval_accuracy      : 1.0 if correct node(s) retrieved, 0.0 otherwise
  - instruction_correctness : semantic similarity between retrieved and GT instruction
  - subdoc_recognition      : 1.0 if agent correctly identifies sub-procedure need
                              (only scored when ground truth requires it)
  - clarification_quality   : 1.0 if agent asks appropriate clarification on
                              ambiguous queries, 0.0 if it answers without clarifying
  - interaction_efficiency  : number of turns needed (1 = retrieved correctly first time)
  - hallucination_flag      : 1 if agent references non-existent nodes/procedures
  - composite_score         : weighted average

Scenario types:
  Type A — Direct retrieval: unambiguous query maps to one process node
  Type B — Ambiguous query: query could map to multiple nodes, clarification needed
  Type C — Sub-procedure: correct answer requires referencing a document node
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils
from tasks.base_task import BaseTask, Scenario, TaskResult, GraphSerializer


class Task2InstructionRetrieval(BaseTask):
    """
    Task 2: Instruction Retrieval and Clarification.

    Evaluates the agent's ability to:
      - Retrieve the correct procedural instruction for a given query
      - Recognise when a sub-procedure reference must be followed
      - Ask targeted clarification questions when the query is ambiguous
    """

    TASK_ID = "task2"

    WEIGHTS = {
        "retrieval_accuracy": 0.40,
        "instruction_correctness": 0.25,
        "subdoc_recognition": 0.20,
        "clarification_quality": 0.15,
    }

    # Scenario type labels
    TYPE_DIRECT    = "direct"
    TYPE_AMBIGUOUS = "ambiguous"
    TYPE_SUBDOC    = "subdoc"

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

        # --- Type A: Direct retrieval ---
        # One scenario per process node — query asks how to perform the step
        for node in graph.get_process_nodes():
            query = self._derive_query(node.text, query_type="direct")
            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t2_{scenario_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "user_query": query,
                    "scenario_type": self.TYPE_DIRECT,
                },
                ground_truth={
                    "target_node_id": node.node_id,
                    "target_node_type": node.node_type,
                    "instruction_text": node.text,
                    "requires_subdoc": False,
                    "subdoc_reference": None,
                    "clarification_needed": False,
                },
                metadata={
                    "complexity": graph.complexity,
                    "scenario_type": self.TYPE_DIRECT,
                    "graph_description": graph.description,
                },
            ))
            scenario_idx += 1

        # --- Type B: Ambiguous query ---
        # For each decision node, generate a query that could apply to
        # either branch — agent should ask for clarification
        for node in graph.get_decision_nodes():
            ctx = GraphUtils.get_decision_context(graph, node.node_id)
            if not ctx["yes_branch"] or not ctx["no_branch"]:
                continue
            query = self._derive_ambiguous_query(ctx)
            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t2_{scenario_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "user_query": query,
                    "scenario_type": self.TYPE_AMBIGUOUS,
                },
                ground_truth={
                    "target_node_id": node.node_id,
                    "target_node_type": "decision",
                    "instruction_text": node.text,
                    "requires_subdoc": False,
                    "subdoc_reference": None,
                    "clarification_needed": True,
                    "clarification_question": node.text,
                },
                metadata={
                    "complexity": graph.complexity,
                    "scenario_type": self.TYPE_AMBIGUOUS,
                    "graph_description": graph.description,
                },
            ))
            scenario_idx += 1

        # --- Type C: Sub-procedure reference ---
        # For each document node, query asks how to handle the issue
        # that branches to that sub-procedure
        for ref in GraphUtils.get_subdocument_references(graph):
            doc_node_id = ref["node_id"]
            doc_node = graph.get_node(doc_node_id)
            parent_texts = [p["text"] for p in ref["referenced_from"]]
            if not parent_texts:
                continue
            query = self._derive_subdoc_query(doc_node.text, parent_texts[0])
            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t2_{scenario_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "user_query": query,
                    "scenario_type": self.TYPE_SUBDOC,
                },
                ground_truth={
                    "target_node_id": doc_node_id,
                    "target_node_type": "document",
                    "instruction_text": doc_node.text,
                    "requires_subdoc": True,
                    "subdoc_reference": doc_node.text,
                    "clarification_needed": False,
                },
                metadata={
                    "complexity": graph.complexity,
                    "scenario_type": self.TYPE_SUBDOC,
                    "graph_description": graph.description,
                    "referenced_from": parent_texts[0][:80],
                },
            ))
            scenario_idx += 1

        return scenarios

    # ------------------------------------------------------------------
    # Query derivation helpers
    # ------------------------------------------------------------------

    def _derive_query(self, instruction_text: str, query_type: str) -> str:
        """Derive a user query from a process node instruction."""
        text = instruction_text.strip()
        # Frame as an operator asking what to do
        return (
            f"What should I do to: {text[:120]}? "
            f"Please provide the specific steps from the procedure."
        )

    def _derive_ambiguous_query(self, ctx: dict) -> str:
        """
        Derive an ambiguous query from a decision node context.
        The query intentionally omits the Yes/No condition so the
        agent should ask for clarification.
        """
        question = ctx["question"]
        yes_action = ctx["yes_branch"]["text"][:60] if ctx["yes_branch"] else ""
        no_action = ctx["no_branch"]["text"][:60] if ctx["no_branch"] else ""
        return (
            f"I need guidance on the following check: '{question}'. "
            f"What should I do next?"
        )

    def _derive_subdoc_query(self, doc_text: str, parent_text: str) -> str:
        """Derive a query for a sub-procedure reference node."""
        return (
            f"During the diagnostic procedure, I encountered the following condition: "
            f"'{parent_text[:100]}'. "
            f"How should I proceed?"
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        system = self._load_system_prompt()
        graph_context = self._get_graph_context(graph)
        query = scenario.input["user_query"]
        scenario_type = scenario.input.get("scenario_type", self.TYPE_DIRECT)

        # Adjust instructions based on scenario type
        if scenario_type == self.TYPE_AMBIGUOUS:
            task_instruction = (
                "The user query may be ambiguous — if you cannot determine "
                "the correct instruction without more information, ask a single "
                "targeted clarification question instead of guessing."
            )
        elif scenario_type == self.TYPE_SUBDOC:
            task_instruction = (
                "If the correct answer requires consulting a referenced sub-procedure "
                "or external document, identify the sub-procedure reference and "
                "indicate that it must be consulted."
            )
        else:
            task_instruction = (
                "Retrieve the specific instruction from the procedure that "
                "directly answers the user query."
            )

        prompt = f"""{system}

---

You are assisting a worker in a manufacturing environment with a procedural query.

DIAGNOSTIC PROCEDURE:
{graph_context}

---

USER QUERY:
{query}

---

TASK:
{task_instruction}

Respond ONLY with a valid JSON object matching this exact template:

{{
  "retrieved_node_id": "<node ID of the most relevant node, e.g. box_3>",
  "retrieved_node_type": "<terminator | decision | process | document>",
  "instruction_text": "<the exact text of the retrieved node>",
  "requires_subdoc": <true if a sub-procedure must be consulted, false otherwise>,
  "subdoc_reference": "<name/text of the sub-procedure if requires_subdoc is true, else null>",
  "clarification_needed": <true if you need more information, false otherwise>,
  "clarification_question": "<your clarification question if needed, else null>",
  "reasoning": "<one sentence explaining your retrieval decision>"
}}"""
        return prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, result: TaskResult, scenario: Scenario) -> dict:
        gt = scenario.ground_truth
        parsed = result.parsed_output
        scenario_type = scenario.input.get("scenario_type", self.TYPE_DIRECT)

        if parsed is None:
            return {
                "retrieval_accuracy": 0.0,
                "instruction_correctness": 0.0,
                "subdoc_recognition": 0.0,
                "clarification_quality": 0.0,
                "interaction_efficiency": 0,
                "composite_score": 0.0,
                "hallucination_flag": 0,
                "parse_success": 0,
            }

        # 1. Retrieval accuracy — correct node ID
        pred_node = parsed.get("retrieved_node_id", "").strip()
        gt_node = gt["target_node_id"].strip()
        retrieval_score = 1.0 if pred_node == gt_node else 0.0

        # 2. Instruction correctness — text similarity
        pred_text = parsed.get("instruction_text", "").strip().lower()
        gt_text = gt["instruction_text"].strip().lower()
        instruction_score = self._text_similarity(pred_text, gt_text)

        # 3. Sub-procedure recognition (only scored for TYPE_SUBDOC)
        gt_requires_subdoc = gt.get("requires_subdoc", False)
        pred_requires_subdoc = parsed.get("requires_subdoc", False)
        if scenario_type == self.TYPE_SUBDOC:
            subdoc_score = 1.0 if pred_requires_subdoc == gt_requires_subdoc else 0.0
        else:
            # For non-subdoc scenarios, penalise false positives
            subdoc_score = 1.0 if not pred_requires_subdoc else 0.0

        # 4. Clarification quality (only scored for TYPE_AMBIGUOUS)
        gt_clarification = gt.get("clarification_needed", False)
        pred_clarification = parsed.get("clarification_needed", False)
        if scenario_type == self.TYPE_AMBIGUOUS:
            # Agent should ask for clarification
            clarification_score = 1.0 if pred_clarification else 0.0
        else:
            # Agent should NOT ask for clarification on unambiguous queries
            clarification_score = 1.0 if not pred_clarification else 0.0

        # 5. Interaction efficiency
        # For now: 1 turn if no clarification needed or correctly triggered,
        # 0 if agent asked clarification when not needed (wastes a turn)
        if scenario_type == self.TYPE_AMBIGUOUS:
            efficiency = 1 if pred_clarification else 2
        else:
            efficiency = 1 if not pred_clarification else 2

        # 6. Composite score
        composite = (
            self.WEIGHTS["retrieval_accuracy"] * retrieval_score
            + self.WEIGHTS["instruction_correctness"] * instruction_score
            + self.WEIGHTS["subdoc_recognition"] * subdoc_score
            + self.WEIGHTS["clarification_quality"] * clarification_score
        )

        # 7. Hallucination flag
        hallucination = self._check_hallucination(parsed, scenario)

        return {
            "retrieval_accuracy": retrieval_score,
            "instruction_correctness": round(instruction_score, 4),
            "subdoc_recognition": subdoc_score,
            "clarification_quality": clarification_score,
            "interaction_efficiency": efficiency,
            "composite_score": round(composite, 4),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _text_similarity(self, pred: str, gt: str) -> float:
        """
        Simple token overlap similarity (Jaccard).
        Used for instruction_correctness scoring.
        For a more robust implementation, replace with
        sentence-transformers cosine similarity.
        """
        if not pred or not gt:
            return 0.0
        pred_tokens = set(pred.split())
        gt_tokens = set(gt.split())
        intersection = pred_tokens & gt_tokens
        union = pred_tokens | gt_tokens
        return len(intersection) / len(union) if union else 0.0

    def _check_hallucination(
        self,
        parsed: dict,
        scenario: Scenario,
    ) -> int:
        """Return 1 if agent references non-existent node IDs."""
        graph = self.loader.load(scenario.graph_id)
        valid_node_ids = set(graph.nodes.keys())

        pred_node = parsed.get("retrieved_node_id", "")
        if pred_node and pred_node not in valid_node_ids:
            return 1

        # Check reasoning for hallucinated node references
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
            "retrieval_accuracy",
            "instruction_correctness",
            "subdoc_recognition",
            "clarification_quality",
            "interaction_efficiency",
            "composite_score",
            "hallucination_flag",
            "parse_success",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated[f"mean_{metric}"] = round(sum(values) / len(values), 4)

        # Per scenario type breakdown
        for stype in [Task2InstructionRetrieval.TYPE_DIRECT,
                      Task2InstructionRetrieval.TYPE_AMBIGUOUS,
                      Task2InstructionRetrieval.TYPE_SUBDOC]:
            type_results = [r for r in scored
                           if r.metadata.get("scenario_type") == stype]
            if type_results:
                vals = [r.scores.get("composite_score", 0.0) for r in type_results]
                aggregated[f"mean_composite_{stype}"] = round(
                    sum(vals) / len(vals), 4
                )
                aggregated[f"n_{stype}"] = len(type_results)

        aggregated["n_scenarios"] = len(scored)
        return aggregated
