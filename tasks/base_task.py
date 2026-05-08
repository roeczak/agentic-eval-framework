"""
base_task.py
------------
Abstract base class for all evaluation tasks in the framework.

Every task inherits from BaseTask and implements:
  - load_scenarios()  : load or generate test scenarios from GRAPH documents
  - build_prompt()    : construct the model prompt for a given scenario
  - run()             : execute the task against a model
  - score()           : evaluate model output against ground truth

Design principles:
  - context_mode controls how the graph is presented to the model:
      'raw' : original JSON structure (default)
      'nl'  : structured natural language serialisation
  - All task outputs are dicts for easy serialisation to JSON
  - Scoring is always separate from running — raw outputs are saved first
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from framework.graph_loader import Graph, GraphLoader
from framework.graph_utils import GraphUtils
from framework.roles import RoleHierarchy


# ---------------------------------------------------------------------------
# Scenario and result data classes
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single test scenario for any task."""
    scenario_id: str            # e.g. "GRAPH21_t1_001"
    graph_id: str               # source graph
    task_id: str                # e.g. "task1"
    input: dict                 # task-specific input fields
    ground_truth: dict          # expected output
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "graph_id": self.graph_id,
            "task_id": self.task_id,
            "input": self.input,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        return cls(
            scenario_id=d["scenario_id"],
            graph_id=d["graph_id"],
            task_id=d["task_id"],
            input=d["input"],
            ground_truth=d["ground_truth"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class TaskResult:
    """Raw output from running a model on a single scenario."""
    scenario_id: str
    graph_id: str
    task_id: str
    model_id: str
    context_mode: str           # 'raw' | 'nl'
    prompt: str
    raw_output: str
    parsed_output: Optional[dict]
    scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "graph_id": self.graph_id,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "context_mode": self.context_mode,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "scores": self.scores,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Graph serialisation helpers
# ---------------------------------------------------------------------------

class GraphSerializer:
    """
    Converts a Graph object into a string representation
    suitable for inclusion in a model prompt.

    Two modes:
      'raw' : compact JSON of nodes and edges (default)
      'nl'  : structured natural language description
    """

    @staticmethod
    def serialize(graph: Graph, mode: str = "raw") -> str:
        if mode == "raw":
            return GraphSerializer._to_json(graph)
        elif mode == "nl":
            return GraphSerializer._to_nl(graph)
        else:
            raise ValueError(f"Unknown context_mode: '{mode}'. Use 'raw' or 'nl'.")

    @staticmethod
    def _to_json(graph: Graph) -> str:
        data = {
            "id": graph.graph_id,
            "description": graph.description,
            "nodes": {
                nid: {"type": n.node_type, "text": n.text}
                for nid, n in graph.nodes.items()
            },
            "edges": [
                {"source": e.source, "target": e.target, "label": e.label}
                for e in graph.edges
            ],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def _to_nl(graph: Graph) -> str:
        lines = [
            f"PROCEDURE: {graph.description}",
            f"COMPLEXITY: {graph.complexity.replace('_', ' ').title()}",
            "",
            "DECISION POINTS:",
        ]
        for node in graph.get_decision_nodes():
            context = GraphUtils.get_decision_context(graph, node.node_id)
            yes = context["yes_branch"]
            no = context["no_branch"]
            lines.append(f"  [{node.node_id}] {node.text}")
            if yes:
                lines.append(f"    -> YES: {yes['text'][:100]}")
            if no:
                lines.append(f"    -> NO: {no['text'][:100]}")

        lines += ["", "ACTION STEPS:"]
        for node in graph.get_process_nodes():
            lines.append(f"  [{node.node_id}] {node.text}")

        if graph.has_subdocument_references:
            lines += ["", "SUB-PROCEDURE REFERENCES:"]
            for node in graph.get_document_nodes():
                lines.append(f"  [{node.node_id}] {node.text}")

        lines += ["", "RESOLUTION ENDPOINTS:"]
        for node in graph.get_terminal_nodes():
            lines.append(f"  [{node.node_id}] {node.text}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base task
# ---------------------------------------------------------------------------

class BaseTask(ABC):
    """
    Abstract base class for all evaluation tasks.

    Subclasses must implement:
      - task_id        : str property
      - load_scenarios : generate Scenario objects from GRAPH documents
      - build_prompt   : construct the prompt string for a given scenario
      - score          : compute metric scores for a TaskResult
    """

    def __init__(
        self,
        graphs_dir: str = "data/graphs/",
        scenarios_dir: str = "data/scenarios/",
        context_mode: str = "raw",
    ):
        self.graphs_dir = Path(graphs_dir)
        self.scenarios_dir = Path(scenarios_dir)
        self.context_mode = context_mode
        self.loader = GraphLoader(graphs_dir)
        self.serializer = GraphSerializer()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def task_id(self) -> str:
        ...

    @abstractmethod
    def load_scenarios(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        ...

    @abstractmethod
    def build_prompt(self, scenario: Scenario, graph: Graph) -> str:
        ...

    @abstractmethod
    def score(self, result: TaskResult, scenario: Scenario) -> dict:
        ...

    # ------------------------------------------------------------------
    # Concrete run logic
    # ------------------------------------------------------------------

    def run(
        self,
        model,
        scenario: Scenario,
        graph: Optional[Graph] = None,
    ) -> TaskResult:
        """Run the model on a single scenario. Scoring done separately."""
        if graph is None:
            graph = self.loader.load(scenario.graph_id)

        prompt = self.build_prompt(scenario, graph)
        raw_output = model.generate(prompt)
        parsed_output = self._parse_output(raw_output)

        return TaskResult(
            scenario_id=scenario.scenario_id,
            graph_id=scenario.graph_id,
            task_id=self.task_id,
            model_id=model.model_id,
            context_mode=self.context_mode,
            prompt=prompt,
            raw_output=raw_output,
            parsed_output=parsed_output,
        )

    def run_and_score(
        self,
        model,
        scenario: Scenario,
        graph: Optional[Graph] = None,
    ) -> TaskResult:
        """Convenience: run then score in one call."""
        result = self.run(model, scenario, graph)
        result.scores = self.score(result, scenario)
        return result

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw_output: str) -> Optional[dict]:
        """
        Attempt to parse model output as JSON.
        Strips markdown code fences if present.
        Returns None if parsing fails.
        """
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Scenario persistence
    # ------------------------------------------------------------------

    def save_scenarios(self, scenarios: list[Scenario]):
        task_dir = self.scenarios_dir / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            fpath = task_dir / f"{scenario.scenario_id}.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(scenario.to_dict(), f, indent=2)

    def load_scenarios_from_disk(
        self,
        graph_ids: Optional[list[str]] = None,
    ) -> list[Scenario]:
        task_dir = self.scenarios_dir / self.task_id
        if not task_dir.exists():
            return []
        scenarios = []
        for fpath in sorted(task_dir.glob("*.json")):
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            scenario = Scenario.from_dict(d)
            if graph_ids is None or scenario.graph_id in graph_ids:
                scenarios.append(scenario)
        return scenarios

    def save_results(self, results: list[TaskResult], output_dir: str):
        out_dir = Path(output_dir) / self.task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            fpath = out_dir / f"{result.scenario_id}_{result.model_id}.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Shared prompt utilities
    # ------------------------------------------------------------------

    def _get_graph_context(self, graph: Graph) -> str:
        return self.serializer.serialize(graph, mode=self.context_mode)

    def _load_system_prompt(self) -> str:
        import yaml
        prompt_path = Path("prompts/system.yaml")
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                data = yaml.safe_load(f)
            return data.get("system_prompt", "")
        return (
            "You are an industrial AI assistant supporting operators and technicians "
            "in a manufacturing environment. You help diagnose equipment errors, "
            "navigate standard operating procedures, and determine appropriate "
            "escalation actions based on role hierarchies."
        )
