"""
rescore_task4.py
----------------
Replays saved Task 4 dialogue histories with fixed decision normalisation.
Corrects YES/NO capitalisation and 'Is' echo issues without re-running model.

Usage:
    python scripts/rescore_task4.py --model qwen2.5-7b
    python scripts/rescore_task4.py --model qwen2.5-7b --save
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.graph_loader import GraphLoader
from framework.graph_utils import GraphUtils
from tasks.task4_procedural_execution import Task4ProceduralExecution
from tasks.base_task import TaskResult


def normalise_decision(raw: str) -> str:
    """
    Normalise model decision to 'Yes' or 'No'.
    Handles: YES, NO, yes, no, Yes., No. etc.
    Returns '' if not a valid decision after normalisation.
    """
    if not raw:
        return ""
    cleaned = raw.strip().rstrip(".!?,").capitalize()
    return cleaned if cleaned in ("Yes", "No") else ""


def replay_history(history: list[dict], graph) -> dict:
    """
    Replay a saved dialogue history using agent_decision field,
    with corrected decision normalisation.
    """
    visited_nodes = []
    decisions_made = []
    finished = False
    lost = False
    turns = 0

    # Start from start node
    start = graph.get_start_node()
    visited_nodes.append(start.node_id)

    # Advance past start terminator
    start_successors = graph.successors(start.node_id)
    if not start_successors:
        return {"visited_nodes": visited_nodes, "decisions_made": decisions_made,
                "finished": False, "lost": True, "turns": 0}

    current_node_id = start_successors[0].target
    visited_nodes.append(current_node_id)

    for turn in history:
        node = graph.get_node(current_node_id)
        turns += 1

        if node.is_terminator():
            finished = True
            break

        if node.is_decision():
            # Use agent_decision field (top-level in turn) — this is what was stored
            raw_decision = turn.get("agent_decision", "") or ""

            # Also try parsed_response as fallback
            if not raw_decision:
                parsed = turn.get("parsed_response") or {}
                raw_decision = parsed.get("decision", "") or ""

            decision = normalise_decision(raw_decision)

            if not decision:
                lost = True
                break

            decisions_made.append((current_node_id, decision))
            next_id = GraphUtils.next_step(graph, current_node_id, decision)
            if next_id is None:
                lost = True
                break
            current_node_id = next_id
            visited_nodes.append(current_node_id)

        elif node.is_process() or node.is_document():
            next_id = GraphUtils.next_step(graph, current_node_id)
            if next_id is None:
                finished = True
                break
            current_node_id = next_id
            visited_nodes.append(current_node_id)

    # Final check
    if visited_nodes:
        last_node = graph.get_node(visited_nodes[-1])
        if last_node.is_terminator():
            finished = True
            lost = False

    return {
        "visited_nodes": visited_nodes,
        "decisions_made": decisions_made,
        "finished": finished,
        "lost": lost,
        "turns": turns,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--raw_dir", default="results/raw/unit/")
    p.add_argument("--save", action="store_true")
    p.add_argument("--graphs_dir", default="data/graphs/")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir) / "task4" / args.model
    if not raw_dir.exists():
        print(f"No Task 4 results at {raw_dir}")
        return

    loader = GraphLoader(args.graphs_dir)
    task = Task4ProceduralExecution(graphs_dir=args.graphs_dir)

    result_files = sorted(raw_dir.glob("*.json"))
    print(f"\n[rescore_task4] {len(result_files)} files | model: {args.model}\n")

    before = {"lost": 0, "finished": 0, "da": []}
    after  = {"lost": 0, "finished": 0, "da": []}

    for fpath in result_files:
        with open(fpath) as f:
            d = json.load(f)

        po = d.get("parsed_output", {})
        before["lost"]     += int(po.get("lost", False))
        before["finished"] += int(po.get("finished", False))
        orig_da = d.get("scores", {}).get("decision_accuracy")
        if orig_da is not None:
            before["da"].append(orig_da)

        # Load graph
        try:
            graph = loader.load(d["graph_id"])
        except Exception as e:
            print(f"  Could not load {d['graph_id']}: {e}")
            after["lost"] += 1
            continue

        # Replay with fixed normalisation
        history = d.get("metadata", {}).get("dialogue_history", [])
        if not history:
            after["lost"] += int(po.get("lost", False))
            after["finished"] += int(po.get("finished", False))
            continue

        new_parsed = replay_history(history, graph)
        d["parsed_output"] = new_parsed

        after["lost"]     += int(new_parsed.get("lost", False))
        after["finished"] += int(new_parsed.get("finished", False))

        # Rescore
        scenarios = task.generate_scenarios(graph_ids=[d["graph_id"]])
        matching = next((s for s in scenarios if s.scenario_id == d["scenario_id"]), None)

        if matching:
            result_obj = TaskResult(
                scenario_id=d["scenario_id"],
                graph_id=d["graph_id"],
                task_id=d["task_id"],
                model_id=d["model_id"],
                context_mode=d.get("context_mode", "raw"),
                prompt=d.get("prompt", ""),
                raw_output=d.get("raw_output", ""),
                parsed_output=new_parsed,
                scores={},
                metadata=d.get("metadata", {}),
            )
            new_scores = task.score(result_obj, matching)
            d["scores"] = new_scores
            d["metadata"]["rescored"] = True
            if new_scores.get("decision_accuracy") is not None:
                after["da"].append(new_scores["decision_accuracy"])

        if args.save:
            with open(fpath, "w") as f:
                json.dump(d, f, indent=2)

    # Print summary
    n = len(result_files)
    print(f"{'='*55}")
    print(f"RESCORE SUMMARY — {args.model}")
    print(f"{'='*55}")
    print(f"  {'Metric':<35} {'Before':>8} {'After':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Lost':<35} {before['lost']:>8} {after['lost']:>8}")
    print(f"  {'Finished':<35} {before['finished']:>8} {after['finished']:>8}")
    b_da = sum(before['da'])/len(before['da']) if before['da'] else 0.0
    a_da = sum(after['da'])/len(after['da'])   if after['da']  else 0.0
    print(f"  {'Decision accuracy (mean)':<35} {b_da:>8.4f} {a_da:>8.4f}")
    print(f"  {'Total scenarios':<35} {n:>8} {n:>8}")
    print()
    if not args.save:
        print("Run with --save to write corrected scores back to files.")


if __name__ == "__main__":
    main()
