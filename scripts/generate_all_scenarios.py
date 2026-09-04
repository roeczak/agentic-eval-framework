"""
generate_all_scenarios.py
--------------------------
Pre-generates and saves all scenarios for all tasks from the GRAPH documents.
Run this once before evaluation to avoid regenerating scenarios on every run.

Usage:
    python scripts/generate_all_scenarios.py
    python scripts/generate_all_scenarios.py --graphs GRAPH06 GRAPH22 GRAPH21
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    p = argparse.ArgumentParser(description="Pre-generate all evaluation scenarios")
    p.add_argument("--graphs_dir", default="data/graphs/")
    p.add_argument("--scenarios_dir", default="data/scenarios/")
    p.add_argument("--graphs", nargs="+", default=None,
                   help="Specific graph IDs (default: all)")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="Tasks to generate scenarios for (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    task_ids = (list(TASK_CLASSES.keys())
                if "all" in args.tasks else args.tasks)

    print(f"\n[generate_all_scenarios]")
    print(f"  Tasks:  {task_ids}")
    print(f"  Graphs: {args.graphs or 'all'}\n")

    for task_id in task_ids:
        TaskClass = TASK_CLASSES[task_id]
        task = TaskClass(
            graphs_dir=args.graphs_dir,
            scenarios_dir=args.scenarios_dir,
        )
        scenarios = task.generate_scenarios(graph_ids=args.graphs)
        task.save_scenarios(scenarios)
        print(f"  {task_id}: {len(scenarios)} scenarios saved to "
              f"{args.scenarios_dir}/{task_id}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
