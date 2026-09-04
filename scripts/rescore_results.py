"""
rescore_results.py
------------------
Loads raw result files from disk and produces:
  1. A summary JSON with all per-task sub-scores aggregated
  2. A breakdown by complexity tier
  3. A breakdown by scenario type (Task 2) / case type (Task 3)
  4. The 9 framework metrics

Does NOT require re-running the model — works entirely from saved raw results.

Usage:
    # Summarise all tasks for one model
    python scripts/rescore_results.py --model qwen2.5-7b

    # Single task only
    python scripts/rescore_results.py --model qwen2.5-7b --task task1

    # Save summary to file
    python scripts/rescore_results.py --model qwen2.5-7b --save
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.base_task import TaskResult
from scoring.metrics import FrameworkMetrics


# ---------------------------------------------------------------------------
# Sub-score definitions per task
# ---------------------------------------------------------------------------

TASK_SUBSCORES = {
    "task1": [
        "branch_mapping_score",
        "initial_decision_score",
        "error_classification_score",
        "hallucination_flag",
        "parse_success",
    ],
    "task2": [
        "retrieval_accuracy",
        "instruction_correctness",
        "subdoc_recognition",
        "clarification_quality",
        "interaction_efficiency",
        "hallucination_flag",
        "parse_success",
    ],
    "task3": [
        "escalation_decision_correct",
        "target_role_correct",
        "out_of_scope_detection",
        "hallucination_flag",
        "parse_success",
    ],
    "task4": [
        "decision_accuracy",
        "path_f1",
        "path_precision",
        "path_recall",
        "reached_correct_terminal",
        "steps_taken",
        "finished",
        "lost",
        "hallucination_flag",
        "parse_success",
    ],
    "task5": [
        "trace_completeness",
        "decision_coverage",
        "action_coverage",
        "resolution_accuracy",
        "justification_quality",
        "hallucination_flag",
        "parse_success",
    ],
}

COMPLEXITY_TIERS = ["short_tree", "medium_chain", "deep_ladder"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(raw_dir, task_id, model_id, include_runs=True):
    """Load results including all run_N subdirectories."""
    task_dir = Path(raw_dir) / task_id / model_id
    if not task_dir.exists():
        return []
    results = []
    
    # Load base results
    for fpath in sorted(task_dir.glob("*.json")):
        with open(fpath) as f:
            d = json.load(f)
        results.append(_build_result(d))
    
    # Load repeated run subdirectories
    if include_runs:
        for run_dir in sorted(task_dir.glob("run_*")):
            for fpath in sorted(run_dir.glob("*.json")):
                with open(fpath) as f:
                    d = json.load(f)
                results.append(_build_result(d))
    
    return results


def _build_result(d: dict) -> TaskResult:
    return TaskResult(
        scenario_id=d["scenario_id"],
        graph_id=d["graph_id"],
        task_id=d["task_id"],
        model_id=d["model_id"],
        context_mode=d.get("context_mode", "raw"),
        prompt=d.get("prompt", ""),
        raw_output=d.get("raw_output", ""),
        parsed_output=d.get("parsed_output"),
        scores=d.get("scores", {}),
        metadata=d.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def mean(values: list) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_subscores(
    results: list[TaskResult],
    task_id: str,
) -> dict:
    subscores = TASK_SUBSCORES.get(task_id, [])
    agg = {"n": len(results)}
    for metric in subscores:
        values = [r.scores.get(metric, 0.0) for r in results if r.scores]
        agg[metric] = mean(values)
    return agg


def breakdown_by_complexity(
    results: list[TaskResult],
    task_id: str,
) -> dict:
    by_tier = defaultdict(list)
    for r in results:
        tier = r.metadata.get("complexity", "unknown")
        by_tier[tier].append(r)
    breakdown = {}
    for tier in COMPLEXITY_TIERS:
        if tier in by_tier:
            breakdown[tier] = aggregate_subscores(by_tier[tier], task_id)
    return breakdown


def breakdown_by_scenario_type(
    results: list[TaskResult],
    key: str = "scenario_type",
) -> dict:
    by_type = defaultdict(list)
    for r in results:
        stype = r.metadata.get(key) or r.metadata.get("case_type", "unknown")
        by_type[stype].append(r)
    breakdown = {}
    for stype, rs in by_type.items():
        task_id = rs[0].task_id if rs else "unknown"
        breakdown[stype] = aggregate_subscores(rs, task_id)
    return breakdown


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_task_summary(task_id: str, agg: dict, by_complexity: dict, by_type: dict = None):
    print(f"\n{'='*60}")
    print(f"TASK {task_id.upper()} — {agg['n']} scenarios")
    print(f"{'='*60}")

    subscores = TASK_SUBSCORES.get(task_id, [])
    for metric in subscores:
        val = agg.get(metric, 0.0)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {metric:<35} {val:.4f}  |{bar}|")

    if by_complexity:
        print(f"\n  By complexity:")
        primary = subscores[0] if subscores else "n"
        print(f"  {'Tier':<15} {'n':>4}  {primary}")
        print(f"  {'-'*40}")
        for tier in COMPLEXITY_TIERS:
            if tier in by_complexity:
                d = by_complexity[tier]
                print(f"  {tier:<15} {d['n']:>4}  {d.get(primary, 0.0):.4f}")

    if by_type:
        print(f"\n  By scenario/case type:")
        primary = subscores[0] if subscores else "n"
        print(f"  {'Type':<20} {'n':>4}  {primary}")
        print(f"  {'-'*40}")
        for stype, d in by_type.items():
            print(f"  {str(stype):<20} {d['n']:>4}  {d.get(primary, 0.0):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rescore and summarise raw results")
    p.add_argument("--model", required=True, help="Model ID, e.g. qwen2.5-7b")
    p.add_argument("--task", default=None,
                   help="Single task ID (default: all available)")
    p.add_argument("--raw_dir", default="results/raw/unit")
    p.add_argument("--output_dir", default="results/aggregated/")
    p.add_argument("--save", action="store_true",
                   help="Save summary JSON to output_dir")
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = (
        [args.task] if args.task
        else [d.name for d in raw_dir.iterdir()
              if d.is_dir() and (d / args.model).exists()]
    )
    task_ids = sorted(task_ids)

    print(f"\n[rescore_results] Model: {args.model}")
    print(f"Tasks found: {task_ids}")

    all_results = []
    full_summary = {"model_id": args.model, "tasks": {}}

    for task_id in task_ids:
        results = load_results(raw_dir, task_id, args.model)
        if not results:
            print(f"\n  No results found for {task_id}")
            continue

        all_results.extend(results)

        agg = aggregate_subscores(results, task_id)
        by_complexity = breakdown_by_complexity(results, task_id)
        by_type = (
            breakdown_by_scenario_type(results, "scenario_type")
            if task_id == "task2"
            else breakdown_by_scenario_type(results, "case_type")
            if task_id == "task3"
            else None
        )

        print_task_summary(task_id, agg, by_complexity, by_type)

        full_summary["tasks"][task_id] = {
            "overall": agg,
            "by_complexity": by_complexity,
            "by_type": by_type or {},
        }

    # Framework metrics
    if all_results:
        print(f"\n{'='*60}")
        print(f"FRAMEWORK METRICS — {args.model}")
        print(f"{'='*60}")
        fm = FrameworkMetrics()
        report = fm.compute(all_results)
        print(fm.format_report(report, args.model))
        full_summary["framework_metrics"] = report

    # Save
    if args.save:
        out_path = output_dir / f"{args.model}_summary.json"
        with open(out_path, "w") as f:
            json.dump(full_summary, f, indent=2)
        print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
