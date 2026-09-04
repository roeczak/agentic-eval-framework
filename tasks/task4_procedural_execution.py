"""
task4_procedural_execution.py
------------------------------
Task 4: Procedural Execution and Faithfulness (Action stage)

Step-by-step graph navigation: the agent is walked through the diagnostic
procedure one node at a time. At each decision node it must answer Yes or No
based on the fault scenario context. At process and document nodes it receives
the instruction and must acknowledge or flag issues.

The dialogue continues until a terminal node is reached or the agent
is lost (navigates to a non-existent node or exceeds max steps).

Scoring (graph-aware):
  - path_precision          : fraction of agent's visited nodes in GT path
  - path_recall             : fraction of GT path nodes visited by agent
  - path_f1                 : harmonic mean of precision and recall
  - decision_accuracy       : fraction of Yes/No decisions matching GT
  - reached_correct_terminal: 1.0 if agent reaches the GT terminal node
  - hallucination_flag      : 1 if agent references non-existent nodes
  - steps_taken             : number of dialogue turns used
  - composite_score         : weighted combination

Scenario generation:
  One scenario per unique path in each graph.
  Each scenario provides the full fault context and the ground truth path.
  The dialogue is simulated by the runner — not pre-generated.
"""

from __future__ import annotations

import re
from typing import Optional

from framework.graph_loader import Graph, GraphLoader, GraphNode
from framework.graph_utils import GraphUtils, GraphPath
from tasks.base_task import BaseTask, Scenario, TaskResult


# ---------------------------------------------------------------------------
# Dialogue state — tracks agent position in graph during step-by-step run
# ---------------------------------------------------------------------------

class DialogueState:
    """Tracks the agent's position in the graph during step-by-step traversal."""

    def __init__(self, graph: Graph, fault_context: str):
        self.graph = graph
        self.fault_context = fault_context
        self.current_node_id = graph.get_start_node().node_id
        self.visited_nodes: list[str] = [self.current_node_id]
        self.decisions_made: list[tuple[str, str]] = []  # (node_id, Yes/No)
        self.turns: int = 0
        self.finished: bool = False
        self.lost: bool = False     # agent gave invalid response
        self.history: list[dict] = []  # full dialogue log

    @property
    def current_node(self) -> GraphNode:
        return self.graph.get_node(self.current_node_id)

    def to_path(self) -> GraphPath:
        """Convert visited nodes to a GraphPath for scoring."""
        steps = []
        for i, node_id in enumerate(self.visited_nodes):
            label = None
            if i > 0:
                # Find the edge label from previous node to this one
                prev_id = self.visited_nodes[i - 1]
                for edge in self.graph.successors(prev_id):
                    if edge.target == node_id:
                        label = edge.label
                        break
            steps.append((node_id, label))
        return GraphPath(steps)


# ---------------------------------------------------------------------------
# Task 4 implementation
# ---------------------------------------------------------------------------

class Task4ProceduralExecution(BaseTask):
    """
    Task 4: Procedural Execution and Faithfulness.

    Evaluates the agent's ability to navigate a diagnostic flowchart
    step by step, making correct Yes/No decisions at each branch point.
    """

    TASK_ID = "task4"
    MAX_STEPS = 60      # safety limit — prevents infinite loops

    WEIGHTS = {
        "decision_accuracy": 0.40,
        "reached_correct_terminal": 0.30,
        "path_f1": 0.30,
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
        """One scenario per unique path through the graph."""
        paths = GraphUtils.enumerate_paths(graph)
        scenarios = []

        for path_idx, path in enumerate(paths):
            terminal_id = path.node_ids[-1]
            terminal = graph.get_node(terminal_id)

            # Derive fault context from terminal text and graph description
            fault_context = (
                f"You are diagnosing a fault in a manufacturing environment. "
                f"Procedure context: {graph.description}. "
                f"The situation has led to: {terminal.text}"
            )

            # Ground truth decisions: {decision_node_id -> Yes/No}
            gt_decisions = {}
            for i, (node_id, incoming_label) in enumerate(path.steps):
                if incoming_label in ("Yes", "No") and i > 0:
                    prev_node_id = path.node_ids[i - 1]
                    prev_node = graph.get_node(prev_node_id)
                    if prev_node.is_decision():
                        gt_decisions[prev_node_id] = incoming_label

            scenarios.append(Scenario(
                scenario_id=f"{graph.graph_id}_t4_{path_idx:03d}",
                graph_id=graph.graph_id,
                task_id=self.TASK_ID,
                input={
                    "fault_context": fault_context,
                    "graph_description": graph.description,
                },
                ground_truth={
                    "gt_path": path.node_ids,
                    "gt_decisions": gt_decisions,
                    "terminal_node_id": terminal_id,
                    "terminal_text": terminal.text,
                    "path_length": len(path),
                    "n_decisions": len(gt_decisions),
                },
                metadata={
                    "complexity": graph.complexity,
                    "path_length": len(path),
                    "n_decisions": len(gt_decisions),
                    "graph_description": graph.description,
                },
            ))

        return scenarios

    # ------------------------------------------------------------------
    # Prompt construction — step-by-step
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        """
        Build the initial prompt for Task 4.
        This is the opening turn of the dialogue — subsequent turns
        are built by build_step_prompt().
        """
        system = self._load_system_prompt()
        fault_context = scenario.input["fault_context"]

        prompt = f"""{system}

---

FAULT CONTEXT:
{fault_context}

You will now be walked through the diagnostic procedure step by step.
At each decision point, answer Yes or No based on the fault context above.
At each action step, acknowledge the instruction.

Respond ONLY with a valid JSON object at each step.
"""
        return prompt

    def build_step_prompt(
        self,
        state: DialogueState,
        graph: Graph,
        scenario: Scenario,
    ) -> str:
        """
        Build the prompt for a single dialogue step.
        Called repeatedly by run_stepwise().
        """
        node = state.current_node
        fault_context = scenario.input["fault_context"]

        # Build dialogue history summary
        history_lines = []
        for turn in state.history[-5:]:    # last 5 turns for context
            history_lines.append(f"  Step {turn['turn']}: [{turn['node_id']}] "
                                f"{turn['node_type']} — {turn['node_text'][:60]}")
            if turn.get('agent_decision'):
                history_lines.append(f"    Agent answered: {turn['agent_decision']}")
        history_str = "\n".join(history_lines) if history_lines else "  (start)"

        if node.is_decision():
            # Get Yes/No branches for context
            yes_node, no_node = None, None
            for edge in graph.successors(node.node_id):
                target = graph.get_node(edge.target)
                if edge.label == "Yes":
                    yes_node = target.text[:80]
                elif edge.label == "No":
                    no_node = target.text[:80]

            return f"""FAULT CONTEXT: {fault_context}

DIALOGUE HISTORY:
{history_str}

CURRENT STEP — DECISION POINT [{node.node_id}]:
Question: {node.text}

If YES: {yes_node or 'proceed'}
If NO: {no_node or 'proceed'}

Based on the fault context, answer this decision point.

Respond ONLY with:
{{
  "node_id": "{node.node_id}",
  "node_type": "decision",
  "decision": "<Yes or No>",
  "reasoning": "<one sentence>"
}}"""

        elif node.is_process():
            return f"""FAULT CONTEXT: {fault_context}

DIALOGUE HISTORY:
{history_str}

CURRENT STEP — ACTION [{node.node_id}]:
Instruction: {node.text}

Acknowledge this instruction and indicate whether it has been completed
or if there is an issue preventing completion.

Respond ONLY with:
{{
  "node_id": "{node.node_id}",
  "node_type": "process",
  "acknowledged": true,
  "issue_encountered": <true or false>,
  "issue_description": "<description if issue, else null>",
  "reasoning": "<one sentence>"
}}"""

        elif node.is_document():
            return f"""FAULT CONTEXT: {fault_context}

DIALOGUE HISTORY:
{history_str}

CURRENT STEP — SUB-PROCEDURE REFERENCE [{node.node_id}]:
The procedure references: {node.text}

Acknowledge this sub-procedure reference.

Respond ONLY with:
{{
  "node_id": "{node.node_id}",
  "node_type": "document",
  "acknowledged": true,
  "subdoc_name": "{node.text[:100]}",
  "reasoning": "<one sentence>"
}}"""

        else:  # terminator
            return f"""FAULT CONTEXT: {fault_context}

DIALOGUE HISTORY:
{history_str}

PROCEDURE COMPLETE — TERMINAL REACHED [{node.node_id}]:
Resolution: {node.text}

Confirm the resolution.

Respond ONLY with:
{{
  "node_id": "{node.node_id}",
  "node_type": "terminator",
  "resolution_confirmed": true,
  "resolution_text": "{node.text[:150]}",
  "reasoning": "<one sentence>"
}}"""

    # ------------------------------------------------------------------
    # Step-by-step runner
    # ------------------------------------------------------------------

    def run_stepwise(
        self,
        model,
        scenario: Scenario,
        graph: Optional[Graph] = None,
    ) -> TaskResult:
        """
        Run the step-by-step dialogue for Task 4.

        Instead of a single model.generate() call, this method runs a
        dialogue loop: at each step it builds a prompt, calls the model,
        parses the response, and advances the graph state accordingly.

        Returns a TaskResult with the full dialogue log in metadata.
        """
        if graph is None:
            graph = self.loader.load(scenario.graph_id)

        state = DialogueState(
            graph=graph,
            fault_context=scenario.input["fault_context"],
        )

        # Initial prompt (not sent to model — sets context)
        initial_prompt = self.build_prompt(scenario, graph)

        # Advance past start terminator
        start_node = graph.get_start_node()
        start_successors = graph.successors(start_node.node_id)
        if start_successors:
            state.current_node_id = start_successors[0].target
            state.visited_nodes.append(state.current_node_id)

        # Dialogue loop
        while not state.finished and state.turns < self.MAX_STEPS:
            node = state.current_node

            # If we've reached a terminal, we're done
            if node.is_terminator():
                state.finished = True
                state.history.append({
                    "turn": state.turns,
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "node_text": node.text,
                    "agent_decision": None,
                })
                break

            # Build step prompt and get model response
            step_prompt = self.build_step_prompt(state, graph, scenario)
            raw_response = model.generate(step_prompt)
            parsed_response = self._parse_output(raw_response)

            state.turns += 1

            # Log this turn
            turn_log = {
                "turn": state.turns,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "node_text": node.text,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "agent_decision": None,
            }

            if parsed_response is None:
                # Failed to parse — mark as lost
                state.lost = True
                state.history.append(turn_log)
                break

            # Advance graph state based on node type and agent response
            if node.is_decision():
                raw_decision = parsed_response.get("decision", "").strip()
                decision = raw_decision.capitalize().rstrip(".")
                if decision not in ("Yes", "No"):
                    state.lost = True
                    state.history.append(turn_log)
                    break

                turn_log["agent_decision"] = decision
                state.decisions_made.append((node.node_id, decision))

                # Find next node
                next_id = GraphUtils.next_step(graph, node.node_id, decision)
                if next_id is None:
                    state.lost = True
                    state.history.append(turn_log)
                    break

                state.current_node_id = next_id
                state.visited_nodes.append(next_id)

            elif node.is_process() or node.is_document():
                # Advance unconditionally
                next_id = GraphUtils.next_step(graph, node.node_id)
                if next_id is None:
                    # Process/document node with no successor — treat as done
                    state.finished = True
                    state.history.append(turn_log)
                    break

                state.current_node_id = next_id
                state.visited_nodes.append(next_id)

            state.history.append(turn_log)

        # Build TaskResult from dialogue state
        agent_path = state.to_path()

        return TaskResult(
            scenario_id=scenario.scenario_id,
            graph_id=scenario.graph_id,
            task_id=self.TASK_ID,
            model_id=model.model_id,
            context_mode=self.context_mode,
            prompt=initial_prompt,
            raw_output=str([t.get("raw_response", "") for t in state.history]),
            parsed_output={
                "visited_nodes": state.visited_nodes,
                "decisions_made": state.decisions_made,
                "finished": state.finished,
                "lost": state.lost,
                "turns": state.turns,
            },
            metadata={
                "dialogue_history": state.history,
                "agent_path": agent_path.node_ids,
                "complexity": scenario.metadata.get("complexity"),
            },
        )

    # run() override — use run_stepwise for Task 4
    def run(
        self,
        model,
        scenario: Scenario,
        graph: Optional[Graph] = None,
    ) -> TaskResult:
        return self.run_stepwise(model, scenario, graph)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, result: TaskResult, scenario: Scenario) -> dict:
        gt = scenario.ground_truth
        parsed = result.parsed_output

        if parsed is None:
            return self._zero_scores()

        gt_path_nodes = gt["gt_path"]
        gt_decisions = gt["gt_decisions"]
        agent_nodes = parsed.get("visited_nodes", [])
        agent_decisions = dict(parsed.get("decisions_made", []))

        # 1. Path overlap (precision, recall, F1)
        gt_set = set(gt_path_nodes)
        agent_set = set(agent_nodes)

        precision = len(gt_set & agent_set) / len(agent_set) if agent_set else 0.0
        recall    = len(gt_set & agent_set) / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # 2. Decision accuracy
        if gt_decisions:
            correct = sum(
                1 for node_id, gt_label in gt_decisions.items()
                if agent_decisions.get(node_id) == gt_label
            )
            decision_accuracy = correct / len(gt_decisions)
        else:
            decision_accuracy = 1.0

        # 3. Reached correct terminal
        gt_terminal = gt["terminal_node_id"]
        agent_terminal = agent_nodes[-1] if agent_nodes else None
        reached_terminal = 1.0 if agent_terminal == gt_terminal else 0.0

        # 4. Composite
        composite = (
            self.WEIGHTS["decision_accuracy"] * decision_accuracy
            + self.WEIGHTS["reached_correct_terminal"] * reached_terminal
            + self.WEIGHTS["path_f1"] * f1
        )

        # 5. Hallucination flag
        hallucination = self._check_hallucination(parsed, scenario)

        return {
            "path_precision": round(precision, 4),
            "path_recall": round(recall, 4),
            "path_f1": round(f1, 4),
            "decision_accuracy": round(decision_accuracy, 4),
            "reached_correct_terminal": reached_terminal,
            "composite_score": round(composite, 4),
            "steps_taken": parsed.get("turns", 0),
            "finished": int(parsed.get("finished", False)),
            "lost": int(parsed.get("lost", False)),
            "hallucination_flag": hallucination,
            "parse_success": 1,
        }

    def _zero_scores(self) -> dict:
        return {
            "path_precision": 0.0,
            "path_recall": 0.0,
            "path_f1": 0.0,
            "decision_accuracy": 0.0,
            "reached_correct_terminal": 0.0,
            "composite_score": 0.0,
            "steps_taken": 0,
            "finished": 0,
            "lost": 0,
            "hallucination_flag": 0,
            "parse_success": 0,
        }

    def _check_hallucination(
        self,
        parsed: dict,
        scenario: Scenario,
    ) -> int:
        """Return 1 if agent visited non-existent node IDs."""
        graph = self.loader.load(scenario.graph_id)
        valid_node_ids = set(graph.nodes.keys())
        for node_id in parsed.get("visited_nodes", []):
            if node_id not in valid_node_ids:
                return 1
        for node_id, _ in parsed.get("decisions_made", []):
            if node_id not in valid_node_ids:
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
            "path_precision", "path_recall", "path_f1",
            "decision_accuracy", "reached_correct_terminal",
            "composite_score", "steps_taken",
            "finished", "lost", "hallucination_flag", "parse_success",
        ]
        aggregated = {}
        for metric in metrics:
            values = [r.scores.get(metric, 0.0) for r in scored]
            aggregated[f"mean_{metric}"] = round(sum(values) / len(values), 4)

        # Per complexity tier
        for tier in ["short_tree", "medium_chain", "deep_ladder"]:
            tier_results = [r for r in scored
                           if r.metadata.get("complexity") == tier]
            if tier_results:
                vals = [r.scores.get("composite_score", 0.0) for r in tier_results]
                aggregated[f"mean_composite_{tier}"] = round(
                    sum(vals) / len(vals), 4
                )
                aggregated[f"n_{tier}"] = len(tier_results)

        aggregated["n_scenarios"] = len(scored)
        return aggregated
