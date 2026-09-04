"""
metrics.py
----------
Aggregates all 9 framework metrics across tasks and models.

The nine metrics defined in the evaluation framework:
  1. Error Identification Accuracy     (Task 1)
  2. Interaction Efficiency            (Task 2)
  3. Escalation Appropriateness        (Task 3)
  4. Procedural Accuracy               (Task 4)
  5. Hallucination Rate                (All tasks)
  6. Trace Completeness                (Task 5)
  7. General Consistency               (All tasks — requires repeated runs)
  8. Cross-Lingual Consistency         (Task 6)
  9. Robustness Index                  (All tasks — requires perturbed inputs)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from tasks.base_task import TaskResult

TASK_PRIMARY_METRICS = {
    "task1": "composite_score",
    "task2": "composite_score",
    "task3": "composite_score",
    "task4": "decision_accuracy",
    "task5": "trace_completeness",
}


class FrameworkMetrics:

    @staticmethod
    def error_identification_accuracy(results: list[TaskResult]) -> float:
        t1 = [r for r in results if r.task_id == "task1" and r.scores]
        if not t1:
            return 0.0
        scores = [r.scores.get("branch_mapping_score", 0.0) for r in t1]
        return round(sum(scores) / len(scores) * 100, 2)

    @staticmethod
    def interaction_efficiency(results: list[TaskResult]) -> float:
        t2 = [r for r in results if r.task_id == "task2" and r.scores]
        if not t2:
            return 0.0
        turns = [r.scores.get("interaction_efficiency", 1) for r in t2]
        return round(sum(turns) / len(turns), 2)

    @staticmethod
    def escalation_appropriateness(results: list[TaskResult]) -> float:
        t3 = [r for r in results if r.task_id == "task3" and r.scores]
        if not t3:
            return 0.0
        scores = [r.scores.get("escalation_decision_correct", 0.0) for r in t3]
        return round(sum(scores) / len(scores) * 100, 2)

    @staticmethod
    def procedural_accuracy(results: list[TaskResult]) -> float:
        t4 = [r for r in results if r.task_id == "task4" and r.scores]
        if not t4:
            return 0.0
        scores = [r.scores.get("decision_accuracy", 0.0) for r in t4]
        return round(sum(scores) / len(scores) * 100, 2)

    @staticmethod
    def hallucination_rate(results: list[TaskResult]) -> float:
        scored = [r for r in results if r.scores]
        if not scored:
            return 0.0
        flags = [r.scores.get("hallucination_flag", 0) for r in scored]
        return round(sum(flags) / len(flags) * 100, 2)

    @staticmethod
    def trace_completeness(results: list[TaskResult]) -> float:
        t5 = [r for r in results if r.task_id == "task5" and r.scores]
        if not t5:
            return 0.0
        scores = [r.scores.get("trace_completeness", 0.0) for r in t5]
        return round(sum(scores) / len(scores) * 100, 2)

    @staticmethod
    def general_consistency(
        repeated_results: dict[str, list[TaskResult]],
    ) -> float:
        """
        Stability across repeated runs of the same scenario.
        repeated_results: {scenario_id -> [TaskResult, TaskResult, ...]}
        """
        if not repeated_results:
            return 0.0
        agreement_scores = []
        for scenario_id, runs in repeated_results.items():
            if len(runs) < 2:
                continue
            task_id = runs[0].task_id
            primary = TASK_PRIMARY_METRICS.get(task_id, "composite_score")
            values = [r.scores.get(primary, 0.0) for r in runs if r.scores]
            if not values:
                continue
            mean_val = sum(values) / len(values)
            if mean_val == 0:
                agreement_scores.append(1.0)
                continue
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            cv = (variance ** 0.5) / mean_val
            agreement_scores.append(max(0.0, 1.0 - cv))
        if not agreement_scores:
            return 0.0
        return round(sum(agreement_scores) / len(agreement_scores) * 100, 2)

    @staticmethod
    def cross_lingual_consistency(
        english_results: list[TaskResult],
        translated_results: list[TaskResult],
    ) -> float:
        if not english_results or not translated_results:
            return 0.0
        ratios = []
        for task_id, primary in TASK_PRIMARY_METRICS.items():
            en = [r for r in english_results if r.task_id == task_id and r.scores]
            tr = [r for r in translated_results if r.task_id == task_id and r.scores]
            if not en or not tr:
                continue
            en_mean = sum(r.scores.get(primary, 0.0) for r in en) / len(en)
            tr_mean = sum(r.scores.get(primary, 0.0) for r in tr) / len(tr)
            if en_mean > 0:
                ratios.append(tr_mean / en_mean)
        if not ratios:
            return 0.0
        return round(sum(ratios) / len(ratios) * 100, 2)

    @staticmethod
    def robustness_index(
        clean_results: list[TaskResult],
        perturbed_results: list[TaskResult],
    ) -> float:
        if not clean_results or not perturbed_results:
            return 0.0
        ratios = []
        for task_id, primary in TASK_PRIMARY_METRICS.items():
            clean = [r for r in clean_results if r.task_id == task_id and r.scores]
            pert = [r for r in perturbed_results if r.task_id == task_id and r.scores]
            if not clean or not pert:
                continue
            clean_mean = sum(r.scores.get(primary, 0.0) for r in clean) / len(clean)
            pert_mean = sum(r.scores.get(primary, 0.0) for r in pert) / len(pert)
            if clean_mean > 0:
                ratios.append(min(pert_mean / clean_mean, 1.0))
        if not ratios:
            return 0.0
        return round(sum(ratios) / len(ratios) * 100, 2)

    def compute(
        self,
        results: list[TaskResult],
        repeated_results: Optional[dict[str, list[TaskResult]]] = None,
        english_results: Optional[list[TaskResult]] = None,
        translated_results: Optional[list[TaskResult]] = None,
        clean_results: Optional[list[TaskResult]] = None,
        perturbed_results: Optional[list[TaskResult]] = None,
    ) -> dict:
        return {
            "error_identification_accuracy":
                self.error_identification_accuracy(results),
            "interaction_efficiency":
                self.interaction_efficiency(results),
            "escalation_appropriateness":
                self.escalation_appropriateness(results),
            "procedural_accuracy":
                self.procedural_accuracy(results),
            "hallucination_rate":
                self.hallucination_rate(results),
            "trace_completeness":
                self.trace_completeness(results),
            "general_consistency":
                self.general_consistency(repeated_results or {}),
            "cross_lingual_consistency":
                self.cross_lingual_consistency(
                    english_results or [], translated_results or []),
            "robustness_index":
                self.robustness_index(
                    clean_results or results, perturbed_results or []),
        }

    def format_report(self, report: dict, model_id: str = "") -> str:
        lines = [
            f"\n{'='*60}",
            f"FRAMEWORK METRICS{' — ' + model_id if model_id else ''}",
            f"{'='*60}",
            f"  {'Metric':<40} {'Score':>10}  Dir",
            f"  {'-'*54}",
        ]
        rows = [
            ("Error Identification Accuracy",
             "error_identification_accuracy", "↑", "%"),
            ("Interaction Efficiency",
             "interaction_efficiency", "↓", " turns"),
            ("Escalation Appropriateness",
             "escalation_appropriateness", "↑", "%"),
            ("Procedural Accuracy",
             "procedural_accuracy", "↑", "%"),
            ("Hallucination Rate",
             "hallucination_rate", "↓", "%"),
            ("Trace Completeness",
             "trace_completeness", "↑", "%"),
            ("General Consistency",
             "general_consistency", "↑", "%"),
            ("Cross-Lingual Consistency",
             "cross_lingual_consistency", "↑", "%"),
            ("Robustness Index",
             "robustness_index", "↑", "%"),
        ]
        for name, key, direction, unit in rows:
            val = report.get(key, 0.0)
            val_str = f"{val:.1f}{unit}" if "%" in unit else f"{val:.2f}{unit}"
            lines.append(f"  {name:<40} {val_str:>10}  {direction}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def save_report(self, report: dict, output_path: str, model_id: str = ""):
        out = {"model_id": model_id, "metrics": report}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
