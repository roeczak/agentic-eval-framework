"""
unit_runner.py
--------------
Runs tasks independently (unit evaluation mode).

Each task is evaluated in isolation against its own ground truth.
Tasks 1-5 are run sequentially. Results are saved after each task.

Usage:
    runner = UnitRunner(
        graphs_dir='data/graphs/',
        output_dir='results/raw/',
        context_mode='raw',
    )
    results = runner.run(model, tasks=['task1', 'task2', 'task3'])
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from framework.graph_loader import GraphLoader
from tasks.base_task import TaskResult, Scenario
from tasks.task1_error_identification import Task1ErrorIdentification
from tasks.task2_instruction_retrieval import Task2InstructionRetrieval
from tasks.task3_escalation import Task3Escalation
from tasks.task4_procedural_execution import Task4ProceduralExecution
from tasks.task5_trace_logging import Task5TraceLogging
from scoring.metrics import FrameworkMetrics


TASK_CLASSES = {
    "task1": Task1ErrorIdentification,
    "task2": Task2InstructionRetrieval,
    "task3": Task3Escalation,
    "task4": Task4ProceduralExecution,
    "task5": Task5TraceLogging,
}


class UnitRunner:
    """
    Runs tasks independently in unit evaluation mode.
    Each task is evaluated against its own ground truth in isolation.
    """

    def __init__(
        self,
        graphs_dir: str = "data/graphs/",
        scenarios_dir: str = "data/scenarios/",
        output_dir: str = "results/raw/",
        context_mode: str = "raw",
        graph_ids: Optional[list[str]] = None,
    ):
        self.graphs_dir = graphs_dir
        self.scenarios_dir = scenarios_dir
        self.output_dir = Path(output_dir)
        self.context_mode = context_mode
        self.graph_ids = graph_ids
        self.loader = GraphLoader(graphs_dir)

    def run(
        self,
        model,
        tasks: Optional[list[str]] = None,
        save_results: bool = True,
        run_type: Optional[str] = None,
        run_index: Optional[int] = None,
    ) -> dict[str, list[TaskResult]]:
        task_ids = tasks or list(TASK_CLASSES.keys())
        all_results: dict[str, list[TaskResult]] = {}

        print(f"\n[UnitRunner] Starting unit evaluation")
        print(f"  Model:        {model.model_id}")
        print(f"  Tasks:        {task_ids}")
        print(f"  Context mode: {self.context_mode}")
        print(f"  Graphs:       {self.graph_ids or 'all'}")
        if run_type:
            print(f"  Run type:     {run_type} (index {run_index})")
        print()

        for task_id in task_ids:
            if task_id not in TASK_CLASSES:
                continue

            task = TASK_CLASSES[task_id](
                graphs_dir=self.graphs_dir,
                scenarios_dir=self.scenarios_dir,
                context_mode=self.context_mode,
            )

            scenarios = task.load_scenarios(graph_ids=self.graph_ids)
            if not scenarios:
                continue

            print(f"[UnitRunner] Running {task_id} ({len(scenarios)} scenarios)...")
            results = []

            for scenario in tqdm(scenarios, desc=f"  {task_id}"):
                graph = self.loader.load(scenario.graph_id)
                try:
                    result = task.run(model, scenario, graph) if task_id != "task4" \
                        else task.run_stepwise(model, scenario, graph)
                    result.scores = task.score(result, scenario)
                    result.metadata["complexity"] = scenario.metadata.get("complexity", "unknown")

                    # Tag for consistency/robustness tracking
                    if run_type:
                        result.metadata["run_type"] = run_type
                    if run_index is not None:
                        result.metadata["run_index"] = run_index

                except Exception as e:
                    print(f"\n  [ERROR] {scenario.scenario_id}: {e}")
                    continue
                results.append(result)

            all_results[task_id] = results

            if save_results:
                self._save_task_results(results, task_id, model.model_id, run_index)

        return all_results

    def _save_task_results(
        self,
        results: list[TaskResult],
        task_id: str,
        model_id: str,
        run_index: Optional[int] = None,
        suffix: str = "",
    ):
    # Use run-indexed subdirectory for repeated runs
        if suffix:
            out_dir = self.output_dir / suffix / task_id / model_id
        elif run_index is not None and run_index > 0:
            out_dir = self.output_dir / task_id / model_id / f"run_{run_index}"
        else:
            out_dir = self.output_dir / task_id / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            fpath = out_dir / f"{result.scenario_id}.json"
            with open(fpath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

    def run_perturbed(
        self,
        model,
        task_ids: Optional[list[str]] = None,
    ) -> list[TaskResult]:
        """
        Run tasks with meaning-preserving input perturbations.
        Used to compute Robustness Index (Metric 9).
        Drops articles and modal verbs from operator-facing input fields.
        """
        task_ids = task_ids or list(TASK_CLASSES.keys())
        all_results = []

        print(f"\n[UnitRunner] Starting perturbed evaluation")
        print(f"  Model:  {model.model_id}")
        print(f"  Tasks:  {task_ids}\n")

        for task_id in task_ids:
            if task_id not in TASK_CLASSES:
                continue

            task = TASK_CLASSES[task_id](
                graphs_dir=self.graphs_dir,
                scenarios_dir=self.scenarios_dir,
                context_mode=self.context_mode,
            )

            scenarios = task.load_scenarios(graph_ids=self.graph_ids)
            if not scenarios:
                continue

        # Perturb all scenarios
            perturbed = [self._perturb_scenario(s) for s in scenarios]
            print(f"[UnitRunner] Perturbed {task_id}: {len(perturbed)} scenarios")

            results = []
            for scenario in tqdm(perturbed, desc=f"  {task_id} (perturbed)"):
                graph = self.loader.load(scenario.graph_id)
                try:
                    result = task.run_stepwise(model, scenario, graph) \
                        if task_id == "task4" \
                        else task.run(model, scenario, graph)
                    result.scores = task.score(result, scenario)
                    result.metadata["complexity"] = scenario.metadata.get(
                        "complexity", "unknown"
                    )
                    result.metadata["run_type"] = "perturbed"
                except Exception as e:
                    print(f"\n  [ERROR] {scenario.scenario_id}: {e}")
                    continue
                results.append(result)

            self._save_task_results(
                results, task_id, model.model_id, suffix="perturbed"
            )
            all_results.extend(results)

        return all_results


    def run_repeated(
        self,
        model,
        task_ids: Optional[list[str]] = None,
        n_runs: int = 3,
        ) -> list[TaskResult]:
        """
        Run tasks multiple times on the same scenarios.
        Used to compute General Consistency (Metric 7).
        Only meaningful with temperature > 0 — greedy decoding
        (temperature=0) will always produce identical outputs.
        """
        task_ids = task_ids or list(TASK_CLASSES.keys())
        all_results = []

        print(f"\n[UnitRunner] Starting repeated evaluation ({n_runs} runs)")
        print(f"  Model:  {model.model_id}")
        print(f"  Tasks:  {task_ids}\n")

        for run_idx in range(n_runs):
            print(f"\n[UnitRunner] Run {run_idx + 1}/{n_runs}")
            results = self.run(
                model,
                tasks=task_ids,
                save_results=True,
                run_type="repeated",
                run_index=run_idx,
            )
            flat = [r for rs in results.values() for r in rs]
            all_results.extend(flat)

        return all_results


    def _perturb_scenario(self, scenario: Scenario) -> Scenario:
        """
    Apply meaning-preserving perturbation to operator-facing input fields.
    Drops articles and modal verbs to simulate noisy operator input.
        """
        DROP_WORDS = {
            "a", "an", "the", "some", "any",
            "should", "would", "could", "might", "may",
            "please", "kindly", "just",
        }

        def perturb_text(text: str) -> str:
            words = text.split()
            kept = [w for w in words
                    if w.lower().rstrip(".,?!") not in DROP_WORDS]
            return " ".join(kept)

        perturbed_input = {
            k: perturb_text(v) if isinstance(v, str) else v
            for k, v in scenario.input.items()
        }

        return Scenario(
            scenario_id=f"{scenario.scenario_id}_perturbed",
            graph_id=scenario.graph_id,
            task_id=scenario.task_id,
            input=perturbed_input,
            ground_truth=scenario.ground_truth,
            metadata={**scenario.metadata, "run_type": "perturbed"},
        )



    def compute_framework_metrics(
        self,
        all_results: dict[str, list[TaskResult]],
        model_id: str = "",
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Compute all 9 framework metrics from unit evaluation results.
        Metrics 7, 8, 9 will be 0.0 unless additional inputs provided.
        """
        flat_results = [r for results in all_results.values() for r in results]
        fm = FrameworkMetrics()
        report = fm.compute(flat_results)
        print(fm.format_report(report, model_id))
        if save_path:
            fm.save_report(report, save_path, model_id)
        return report
