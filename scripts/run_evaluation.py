"""
run_evaluation.py
-----------------
Main entry point for running the full evaluation framework.

Usage examples:

  # Unit evaluation — all tasks, Qwen2.5-7B, raw context
  python scripts/run_evaluation.py \\
      --model qwen2.5-7b \\
      --model_path Qwen/Qwen2.5-7B-Instruct \\
      --mode unit \\
      --tasks all

  # Pipeline evaluation — specific graphs
  python scripts/run_evaluation.py \\
      --model qwen2.5-7b \\
      --model_path Qwen/Qwen2.5-7B-Instruct \\
      --mode pipeline \\
      --graphs GRAPH06 GRAPH22 GRAPH21

  # Single task, NL context mode
  python scripts/run_evaluation.py \\
      --model qwen2.5-7b \\
      --model_path Qwen/Qwen2.5-7B-Instruct \\
      --mode unit \\
      --tasks task1 \\
      --context_mode nl

  # vLLM backend
  python scripts/run_evaluation.py \\
      --model qwen2.5-7b \\
      --model_path Qwen/Qwen2.5-7B-Instruct \\
      --backend vllm \\
      --mode unit
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.unit_runner import UnitRunner
from pipeline.pipeline_runner import PipelineRunner
from scoring.metrics import FrameworkMetrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Agentic AI Evaluation Framework"
    )
    p.add_argument("--model", required=True,
                   help="Short model ID for filenames (e.g. qwen2.5-7b)")
    p.add_argument("--model_path", required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--backend", default="hf",
                   choices=["hf", "vllm"],
                   help="Model backend (default: hf)")
    p.add_argument("--mode", default="unit",
                   choices=["unit", "pipeline", "both", "repeated", "perturbed"],
                   help="Evaluation mode (default: unit)")
    p.add_argument("--tasks", nargs="+",
                   default=["all"],
                   help="Tasks to run: task1 task2 ... or 'all'")
    p.add_argument("--graphs", nargs="+", default=None,
                   help="Graph IDs to evaluate on (default: all)")
    p.add_argument("--context_mode", default="raw",
                   choices=["raw", "nl"],
                   help="Graph context serialisation mode (default: raw)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--torch_dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Number of GPUs for vLLM tensor parallelism")
    p.add_argument("--graphs_dir", default="data/graphs/")
    p.add_argument("--scenarios_dir", default="data/scenarios/")
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--run_index", type=int, default=0,
               help="Run index for repeated runs (0, 1, 2...)")
    p.add_argument("--run_type", default=None,
               choices=["repeated", "perturbed"],
               help="Tag results for consistency/robustness scoring")
    p.add_argument("--temperature", type=float, default=0.0,
               help="Sampling temperature (default: 0.0 = greedy)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    if args.backend == "hf":
        from models.hf_model import HFModel
        model = HFModel(
            model_id=args.model,
            model_name_or_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=args.torch_dtype,
            temperature = args.temperature,
        )
    elif args.backend == "vllm":
        from models.vllm_model import VLLMModel
        model = VLLMModel(
            model_id=args.model,
            model_name_or_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.torch_dtype,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    model.load()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve tasks
    task_ids = (
        ["task1", "task2", "task3", "task4", "task5"]
        if "all" in args.tasks else args.tasks
    )

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Agentic AI Evaluation Framework")
    print(f"{'='*60}")
    print(f"Model:        {args.model} ({args.backend})")
    print(f"Mode:         {args.mode}")
    print(f"Tasks:        {task_ids}")
    print(f"Context mode: {args.context_mode}")
    print(f"Graphs:       {args.graphs or 'all'}")
    print(f"{'='*60}\n")

    t_total = time.time()
    model = load_model(args)

    try:
        runner_kwargs = dict(
            graphs_dir=args.graphs_dir,
            scenarios_dir=args.scenarios_dir,
            context_mode=args.context_mode,
            graph_ids=args.graphs,
        )

        # --- Unit mode ---
        if args.mode in ("unit", "both"):
            runner = UnitRunner(
                output_dir=str(raw_dir / "unit"),
                **runner_kwargs,
            )
            unit_results = runner.run(
                model, tasks=task_ids, save_results=True, run_type=args.run_type, run_index=args.run_index,
            )
            # Framework metrics
            report = runner.compute_framework_metrics(
                unit_results,
                model_id=args.model,
                save_path=str(agg_dir / f"{args.model}_unit_metrics.json"),
            )

        # --- Pipeline mode ---
        if args.mode in ("pipeline", "both"):
            pipeline_runner = PipelineRunner(
                output_dir=str(raw_dir / "pipeline"),
                **runner_kwargs,
            )
            pipeline_runs = pipeline_runner.run(model, save_results=True)
            agg = pipeline_runner.aggregate_pipeline_results(pipeline_runs)

            print(f"\n{'='*60}")
            print(f"PIPELINE RESULTS — {args.model}")
            print(f"{'='*60}")
            for k, v in agg.items():
                print(f"  {k}: {v}")

            # Save pipeline aggregate
            with open(agg_dir / f"{args.model}_pipeline_metrics.json", "w") as f:
                json.dump({"model_id": args.model, "pipeline": agg}, f, indent=2)

        elif args.mode == "perturbed":
            runner = UnitRunner(
                output_dir=str(raw_dir / "perturbed"),
                **runner_kwargs,
            )
            results = runner.run_perturbed(
                model,
                task_ids=task_ids,
            )
            # Flatten for metrics
            flat = [r for rs in results.values() for r in rs] \
                if isinstance(results, dict) else results
            report = runner.compute_framework_metrics(
                {"perturbed": flat},
                model_id=args.model,
                save_path=str(agg_dir / f"{args.model}_perturbed_metrics.json"),
            )
        elif args.mode == "repeated":
            runner = UnitRunner(
                output_dir=str(raw_dir / "unit"),
                **runner_kwargs,
            )
            results = runner.run_repeated(
                model,
                task_ids=task_ids,
                n_runs=args.n_runs,
            )
            # Flatten and compute
            flat = [r for rs in results.values() for r in rs] \
                if isinstance(results, dict) else results
            runner.compute_framework_metrics(
                {"repeated": flat},
                model_id=args.model,
                save_path=str(agg_dir / f"{args.model}_repeated_metrics.json"),
            )
    finally:
        model.unload()
        elapsed = time.time() - t_total
        print(f"\n[Done] Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
