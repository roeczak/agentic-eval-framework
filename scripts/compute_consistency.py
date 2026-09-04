"""
compute_consistency.py
----------------------
Computes General Consistency (Metric 7) from repeated runs.
Requires at least 2 runs tagged with run_type='repeated'.

Usage:
    python scripts/compute_consistency.py --model qwen2.5-7b-t07
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.base_task import TaskResult

TASK_PRIMARY = {
    "task1": "branch_mapping_score",
    "task2": "retrieval_accuracy",
    "task3": "escalation_decision_correct",
    "task4": "decision_accuracy",
    "task5": "trace_completeness",
}


def load_all_runs(raw_dir, task_id, model_id):
    """Load results from base dir and all run_N subdirectories."""
    task_dir = Path(raw_dir) / task_id / model_id
    if not task_dir.exists():
        return []
    results = []

    def load_file(fpath):
        with open(fpath) as f:
            d = json.load(f)
        return TaskResult(
            scenario_id=d["scenario_id"],
            graph_id=d["graph_id"],
            task_id=d["task_id"],
            model_id=d["model_id"],
            context_mode=d.get("context_mode", "raw"),
            prompt="", raw_output="",
            parsed_output=d.get("parsed_output"),
            scores=d.get("scores", {}),
            metadata=d.get("metadata", {}),
        )

    # Base directory
    for fpath in sorted(task_dir.glob("*.json")):
        results.append(load_file(fpath))

    # run_N subdirectories
    for run_dir in sorted(task_dir.glob("run_*")):
        for fpath in sorted(run_dir.glob("*.json")):
            results.append(load_file(fpath))

    return results


def compute_consistency_for_task(results, task_id):
    """
    Group results by scenario_id across runs and compute
    consistency = 1 - coefficient_of_variation per scenario,
    then average across all scenarios.
    """
    primary = TASK_PRIMARY.get(task_id, "composite_score")

    # Group by base scenario_id (strip _perturbed suffix if present)
    groups = defaultdict(list)
    for r in results:
        if not r.scores:
            continue
        base_id = r.scenario_id
        val = r.scores.get(primary)
        if val is not None:
            groups[base_id].append(val)

    if not groups:
        return 0.0, 0

    consistencies = []
    for scenario_id, values in groups.items():
        if len(values) < 2:
            continue
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            consistencies.append(1.0 if all(v == 0 for v in values) else 0.0)
            continue
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        cv = std / mean_val
        consistencies.append(max(0.0, 1.0 - cv))

    if not consistencies:
        return 0.0, 0

    return sum(consistencies) / len(consistencies), len(consistencies)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--raw_dir", default="results/raw/unit/")
    p.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"])
    args = p.parse_args()

    print(f"\n[General Consistency] Model: {args.model}")
    print(f"{'='*55}")
    print(f"  {'Task':<10} {'Scenarios':>10} {'Consistency':>12}")
    print(f"  {'-'*35}")

    task_scores = []
    for task_id in args.tasks:
        results = load_all_runs(args.raw_dir, task_id, args.model)
        if not results:
            print(f"  {task_id:<10} {'N/A':>10} {'N/A':>12}")
            continue

        consistency, n = compute_consistency_for_task(results, task_id)
        if n == 0:
            print(f"  {task_id:<10} {'<2 runs':>10} {'N/A':>12}")
            continue

        task_scores.append(consistency)
        flag = "✓" if consistency >= 0.90 else ("~" if consistency >= 0.75 else "⚠")
        print(f"  {task_id:<10} {n:>10} {consistency*100:>11.2f}%  {flag}")

    overall = round(sum(task_scores) / len(task_scores) * 100, 2) \
        if task_scores else 0.0
    print(f"  {'-'*35}")
    print(f"  {'General Consistency':<10} {overall:>12.2f}%")
    print(f"{'='*55}\n")

    out = {
        "model_id": args.model,
        "general_consistency": overall,
    }
    out_path = Path("results/aggregated") / f"{args.model}_consistency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
