"""
run_single_task.py
------------------
Run a single task on a single graph for development and testing.

Usage:
    python scripts/run_single_task.py --task task1 --graph GRAPH06
    python scripts/run_single_task.py --task task3 --graph GRAPH22 --context_mode nl
    python scripts/run_single_task.py --task task4 --graph GRAPH06 --n_scenarios 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hf_model import HFModel
from framework.graph_loader import GraphLoader
from tasks.task1_error_identification import Task1ErrorIdentification
from tasks.task2_instruction_retrieval import Task2InstructionRetrieval
from tasks.task3_escalation import Task3Escalation
from tasks.task4_procedural_execution import Task4ProceduralExecution
from tasks.task5_trace_logging import Task5TraceLogging

TASK_CLASSES = {
    "task1": Task1ErrorIdentification,
    "task2": Task2InstructionRetrieval,
    "task3": Task3Escalation,
    "task4": Task4ProceduralExecution,
    "task5": Task5TraceLogging,
}


def parse_args():
    p = argparse.ArgumentParser(description="Run a single evaluation task")
    p.add_argument("--task", required=True, choices=list(TASK_CLASSES.keys()))
    p.add_argument("--graph", required=True, help="Graph ID, e.g. GRAPH06")
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--model_id", default="qwen2.5-7b")
    p.add_argument("--context_mode", default="raw", choices=["raw", "nl"])
    p.add_argument("--n_scenarios", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--torch_dtype", default="float16")
    p.add_argument("--graphs_dir", default="data/graphs/")
    p.add_argument("--scenarios_dir", default="data/scenarios/")
    p.add_argument("--output_dir", default="results/raw/")
    p.add_argument("--show_prompts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n[run_single_task] {args.task} on {args.graph}")
    print(f"  Model:        {args.model_path}")
    print(f"  Context mode: {args.context_mode}")
    print(f"  Max scenarios:{args.n_scenarios}\n")

    TaskClass = TASK_CLASSES[args.task]
    task = TaskClass(
        graphs_dir=args.graphs_dir,
        scenarios_dir=args.scenarios_dir,
        context_mode=args.context_mode,
    )

    scenarios = task.generate_scenarios(graph_ids=[args.graph])
    scenarios = scenarios[:args.n_scenarios]
    print(f"Scenarios: {len(scenarios)}\n")

    if not scenarios:
        print("No scenarios generated.")
        return

    model = HFModel(
        model_id=args.model_id,
        model_name_or_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
    )
    model.load()
    loader = GraphLoader(args.graphs_dir)

    results = []
    try:
        for i, scenario in enumerate(scenarios):
            graph = loader.load(scenario.graph_id)
            print(f"[{i+1}/{len(scenarios)}] {scenario.scenario_id} "
                  f"({scenario.metadata.get('complexity','?')})")

            if args.show_prompts:
                prompt = task.build_prompt(scenario, graph)
                print(f"  PROMPT:\n{prompt[:600]}...\n")

            if args.task == "task4":
                result = task.run_stepwise(model, scenario, graph)
            else:
                result = task.run(model, scenario, graph)

            result.scores = task.score(result, scenario)
            results.append(result)

            print(f"  Raw output:    {result.raw_output[:150].strip()}")
            print(f"  Scores:        {result.scores}\n")

    finally:
        model.unload()

    print(f"\n{'='*50}")
    print(f"SUMMARY — {args.task} on {args.graph}")
    print(f"{'='*50}")
    if hasattr(TaskClass, "aggregate_scores"):
        agg = TaskClass.aggregate_scores(results)
        for k, v in agg.items():
            print(f"  {k}: {v}")

    out_dir = Path(args.output_dir) / args.task / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        with open(out_dir / f"{r.scenario_id}.json", "w") as f:
            json.dump(r.to_dict(), f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
