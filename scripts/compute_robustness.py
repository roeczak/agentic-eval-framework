"""
compute_robustness.py
---------------------
Computes Robustness Index (Metric 9) by comparing clean vs perturbed results.

Usage:
    python scripts/compute_robustness.py --model qwen2.5-7b
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

def load_results(raw_dir, task_id, model_id):
    task_dir = Path(raw_dir) / task_id / model_id
    if not task_dir.exists():
        return []
    results = []
    for fpath in sorted(task_dir.glob("*.json")):
        with open(fpath) as f:
            d = json.load(f)
        results.append(TaskResult(
            scenario_id=d["scenario_id"],
            graph_id=d["graph_id"],
            task_id=d["task_id"],
            model_id=d["model_id"],
            context_mode=d.get("context_mode", "raw"),
            prompt="", raw_output="",
            parsed_output=d.get("parsed_output"),
            scores=d.get("scores", {}),
            metadata=d.get("metadata", {}),
        ))
    return results

def mean(values):
    return round(sum(values) / len(values), 4) if values else 0.0

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--clean_dir", default="results/raw/unit/")
    p.add_argument("--perturbed_dir", default="results/raw/perturbed/perturbed")
    p.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"])
    args = p.parse_args()

    print(f"\n[Robustness Index] Model: {args.model}")
    print(f"{'='*55}")
    print(f"  {'Task':<10} {'Clean':>8} {'Perturbed':>10} {'Ratio':>8}")
    print(f"  {'-'*40}")

    ratios = []
    for task_id in args.tasks:
        primary = TASK_PRIMARY.get(task_id, "composite_score")

        clean = load_results(args.clean_dir, task_id, args.model)
        perturbed = load_results(args.perturbed_dir, task_id, args.model)

        if not clean or not perturbed:
            print(f"  {task_id:<10} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            continue

        clean_mean = mean([r.scores.get(primary, 0.0)
                          for r in clean if r.scores])
        pert_mean  = mean([r.scores.get(primary, 0.0)
                          for r in perturbed if r.scores])

        ratio = round(min(pert_mean / clean_mean, 1.0), 4) \
            if clean_mean > 0 else 0.0
        ratios.append(ratio)

        flag = "✓" if ratio >= 0.90 else ("~" if ratio >= 0.75 else "⚠")
        print(f"  {task_id:<10} {clean_mean:>8.4f} {pert_mean:>10.4f} "
              f"{ratio:>8.4f}  {flag}")

    overall = round(sum(ratios) / len(ratios) * 100, 2) if ratios else 0.0
    print(f"  {'-'*40}")
    print(f"  {'Robustness Index':<10} {overall:>8.2f}%")
    print(f"{'='*55}\n")

    # Save
    out = {
        "model_id": args.model,
        "robustness_index": overall,
        "per_task": {
            t: {"ratio": r} for t, r in zip(args.tasks, ratios)
        }
    }
    out_path = Path("results/aggregated") / f"{args.model}_robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
