"""
pipeline_runner.py
------------------
Chains Tasks 1-5 in pipeline evaluation mode.

Updated to run ALL scenarios per graph (not just the first path),
making pipeline evaluation comparable to unit evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from framework.graph_loader import GraphLoader
from tasks.base_task import Scenario, TaskResult
from tasks.task1_error_identification import Task1ErrorIdentification
from tasks.task2_instruction_retrieval import Task2InstructionRetrieval
from tasks.task3_escalation import Task3Escalation
from tasks.task4_procedural_execution import Task4ProceduralExecution
from tasks.task5_trace_logging import Task5TraceLogging
from scoring.metrics import FrameworkMetrics


class PipelineRunner:

    def __init__(
        self,
        graphs_dir: str = "data/graphs/",
        scenarios_dir: str = "data/scenarios/",
        output_dir: str = "results/raw/pipeline/",
        context_mode: str = "raw",
        graph_ids: Optional[list[str]] = None,
    ):
        self.graphs_dir = graphs_dir
        self.scenarios_dir = scenarios_dir
        self.output_dir = Path(output_dir)
        self.context_mode = context_mode
        self.graph_ids = graph_ids
        self.loader = GraphLoader(graphs_dir)

        kwargs = dict(
            graphs_dir=graphs_dir,
            scenarios_dir=scenarios_dir,
            context_mode=context_mode,
        )
        self.task1 = Task1ErrorIdentification(**kwargs)
        self.task2 = Task2InstructionRetrieval(**kwargs)
        self.task3 = Task3Escalation(**kwargs)
        self.task4 = Task4ProceduralExecution(**kwargs)
        self.task5 = Task5TraceLogging(**kwargs)

    def run(
        self,
        model,
        save_results: bool = True,
    ) -> list[dict]:
        all_graphs = self.loader.load_all()
        if self.graph_ids:
            graphs = {gid: all_graphs[gid]
                      for gid in self.graph_ids if gid in all_graphs}
        else:
            graphs = {gid: g for gid, g in all_graphs.items()
                      if g.complexity != "linear"}

        print(f"\n[PipelineRunner] Starting pipeline evaluation")
        print(f"  Model:  {model.model_id}")
        print(f"  Graphs: {len(graphs)}\n")

        pipeline_runs = []

        for gid, graph in tqdm(graphs.items(), desc="Graphs"):
            try:
                # Run all scenarios for this graph
                graph_runs = self._run_all_scenarios_for_graph(
                    model, gid, graph
                )
                pipeline_runs.extend(graph_runs)

                if save_results:
                    for run in graph_runs:
                        self._save_pipeline_run(run, model.model_id)

            except Exception as e:
                print(f"\n[PipelineRunner] Error on {gid}: {e}")
                continue

        print(f"\n[PipelineRunner] Completed {len(pipeline_runs)} pipeline runs")
        return pipeline_runs

    def _run_all_scenarios_for_graph(
        self,
        model,
        graph_id: str,
        graph,
    ) -> list[dict]:
        """
        Run the full Task 1-5 pipeline for ALL scenarios in a graph.
        Each scenario corresponds to one root-to-leaf path.
        """
        # Generate all scenarios for each task for this graph
        t1_scenarios = self.task1.generate_scenarios(graph_ids=[graph_id])
        t2_scenarios = self.task2.generate_scenarios(graph_ids=[graph_id])
        t3_scenarios = self.task3.generate_scenarios(graph_ids=[graph_id])
        t4_scenarios = self.task4.generate_scenarios(graph_ids=[graph_id])
        t5_scenarios = self.task5.generate_scenarios(graph_ids=[graph_id])

        # Index scenarios by path index for alignment
        # T1, T4, T5 are path-based (one per path)
        # T2, T3 are type-based (direct/ambiguous/subdoc, no-esc/esc/oos)
        # We run T1/T4/T5 per path, T2/T3 use their full scenario sets

        runs = []

        for path_idx, t1_s in enumerate(t1_scenarios):
            context = {}
            task_results = {}

            # ── Task 1 ──
            try:
                r1 = self.task1.run(model, t1_s, graph)
                r1.scores = self.task1.score(r1, t1_s)
                r1.metadata["pipeline_stage"] = "task1"
                r1.metadata["pipeline_path_idx"] = path_idx
                task_results["task1"] = r1
                if r1.parsed_output:
                    context["error_category"] = r1.parsed_output.get(
                        "error_category", ""
                    )
                    context["entry_node"] = r1.parsed_output.get(
                        "entry_decision_node", ""
                    )
            except Exception as e:
                print(f"  T1 error on {t1_s.scenario_id}: {e}")
                continue

            # ── Task 2 — use matching scenario type or first direct ──
            t2_s = self._get_matching_t2(t2_scenarios, context)
            if t2_s:
                try:
                    t2_s_enriched = self._enrich(t2_s, context)
                    r2 = self.task2.run(model, t2_s_enriched, graph)
                    r2.scores = self.task2.score(r2, t2_s)
                    r2.metadata["pipeline_stage"] = "task2"
                    r2.metadata["pipeline_path_idx"] = path_idx
                    task_results["task2"] = r2
                    if r2.parsed_output:
                        context["retrieved_instruction"] = r2.parsed_output.get(
                            "instruction_text", ""
                        )
                except Exception as e:
                    print(f"  T2 error on path {path_idx}: {e}")

            # ── Task 3 ──
            t3_s = t3_scenarios[path_idx % len(t3_scenarios)] \
                if t3_scenarios else None
            if t3_s:
                try:
                    t3_s_enriched = self._enrich(t3_s, context)
                    r3 = self.task3.run(model, t3_s_enriched, graph)
                    r3.scores = self.task3.score(r3, t3_s)
                    r3.metadata["pipeline_stage"] = "task3"
                    r3.metadata["pipeline_path_idx"] = path_idx
                    task_results["task3"] = r3
                    if r3.parsed_output:
                        context["escalation_needed"] = r3.parsed_output.get(
                            "escalation_needed", False
                        )
                        context["required_role"] = r3.parsed_output.get(
                            "required_role_name", ""
                        )
                except Exception as e:
                    print(f"  T3 error on path {path_idx}: {e}")

            # ── Task 4 — match by path index ──
            t4_s = t4_scenarios[path_idx] \
                if path_idx < len(t4_scenarios) else None
            if t4_s:
                try:
                    t4_s_enriched = self._enrich(t4_s, context)
                    r4 = self.task4.run_stepwise(model, t4_s_enriched, graph)
                    r4.scores = self.task4.score(r4, t4_s)
                    r4.metadata["pipeline_stage"] = "task4"
                    r4.metadata["pipeline_path_idx"] = path_idx
                    task_results["task4"] = r4
                    if r4.parsed_output:
                        context["visited_nodes"] = r4.parsed_output.get(
                            "visited_nodes", []
                        )
                except Exception as e:
                    print(f"  T4 error on path {path_idx}: {e}")

            # ── Task 5 — match by path index ──
            t5_s = t5_scenarios[path_idx] \
                if path_idx < len(t5_scenarios) else None
            if t5_s:
                try:
                    t5_s_enriched = self._enrich(t5_s, context)
                    r5 = self.task5.run(model, t5_s_enriched, graph)
                    r5.scores = self.task5.score(r5, t5_s)
                    r5.metadata["pipeline_stage"] = "task5"
                    r5.metadata["pipeline_path_idx"] = path_idx
                    task_results["task5"] = r5
                except Exception as e:
                    print(f"  T5 error on path {path_idx}: {e}")

            runs.append({
                "graph_id": graph_id,
                "path_idx": path_idx,
                "scenario_id": t1_s.scenario_id,
                "complexity": graph.complexity,
                "task_results": {
                    tid: r.to_dict()
                    for tid, r in task_results.items()
                },
                "pipeline_scores": {
                    tid: r.scores
                    for tid, r in task_results.items()
                },
                "pipeline_context": context,
            })

        return runs

    def _get_matching_t2(
        self,
        t2_scenarios: list,
        context: dict,
    ):
        """Get the most appropriate T2 scenario — prefer direct type."""
        direct = [s for s in t2_scenarios
                  if s.input.get("scenario_type") == "direct"]
        return direct[0] if direct else (t2_scenarios[0] if t2_scenarios else None)

    def _enrich(self, scenario: Scenario, context: dict) -> Scenario:
        """Inject pipeline context into scenario input."""
        enriched_input = dict(scenario.input)
        if context:
            enriched_input["pipeline_context"] = context
        return Scenario(
            scenario_id=scenario.scenario_id,
            graph_id=scenario.graph_id,
            task_id=scenario.task_id,
            input=enriched_input,
            ground_truth=scenario.ground_truth,
            metadata=scenario.metadata,
        )

    def _save_pipeline_run(self, run: dict, model_id: str):
        out_dir = self.output_dir / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"{run['graph_id']}_path{run['path_idx']:03d}.json"
        with open(fpath, "w") as f:
            json.dump(run, f, indent=2)

    def aggregate_pipeline_results(
        self,
        pipeline_runs: list[dict],
    ) -> dict:
        """Aggregate scores across all pipeline runs."""
        task_scores: dict[str, list[float]] = {
            "task1": [], "task2": [], "task3": [],
            "task4": [], "task5": []
        }
        primary_metrics = {
            "task1": "composite_score",
            "task2": "composite_score",
            "task3": "composite_score",
            "task4": "decision_accuracy",
            "task5": "trace_completeness",
        }

        for run in pipeline_runs:
            for task_id, primary in primary_metrics.items():
                scores = run.get("pipeline_scores", {}).get(task_id, {})
                if scores:
                    task_scores[task_id].append(scores.get(primary, 0.0))

        aggregated = {}
        for task_id, scores in task_scores.items():
            if scores:
                aggregated[f"mean_{task_id}"] = round(
                    sum(scores) / len(scores), 4
                )
                aggregated[f"n_{task_id}"] = len(scores)

        # Error propagation: T1 vs T4 divergence
        t1 = task_scores.get("task1", [])
        t4 = task_scores.get("task4", [])
        if t1 and t4:
            min_len = min(len(t1), len(t4))
            diffs = [abs(a - b) for a, b in zip(t1[:min_len], t4[:min_len])]
            aggregated["mean_t1_t4_divergence"] = round(
                sum(diffs) / len(diffs), 4
            )

        # Per complexity breakdown
        for tier in ["short_tree", "medium_chain", "deep_ladder"]:
            tier_runs = [r for r in pipeline_runs
                        if r.get("complexity") == tier]
            if tier_runs:
                tier_scores = []
                for run in tier_runs:
                    s = run.get("pipeline_scores", {}).get("task1", {})
                    if s:
                        tier_scores.append(s.get("composite_score", 0.0))
                if tier_scores:
                    aggregated[f"mean_task1_{tier}"] = round(
                        sum(tier_scores) / len(tier_scores), 4
                    )
                    aggregated[f"n_{tier}"] = len(tier_runs)

        aggregated["n_pipeline_runs"] = len(pipeline_runs)
        return aggregated
