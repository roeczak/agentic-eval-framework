"""
test_task1_model.py
-------------------
End-to-end test of Task 1 with Qwen2.5-7B-Instruct via HuggingFace Transformers.

Runs on a small sample of scenarios (GRAPH06 only — short tree, manageable context)
and prints raw outputs, parsed results, and scores.

Usage:
    python test_task1_model.py

Optional arguments (edit the CONFIG block below):
    MODEL_PATH   : HuggingFace model name or local path
    GRAPH_IDS    : which graphs to test on
    N_SCENARIOS  : max scenarios to run (keep small for first test)
    CONTEXT_MODE : 'raw' or 'nl'
"""

import sys
import json
sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# CONFIG — edit these as needed
# ---------------------------------------------------------------------------

MODEL_ID         = "qwen2.5-7b"
MODEL_PATH       = "Qwen/Qwen2.5-7B-Instruct"  # or local path on HPC
GRAPH_IDS        = ["GRAPH06", "GRAPH22"]        # start small
N_SCENARIOS      = 5                             # max scenarios per graph
CONTEXT_MODE     = "raw"                         # 'raw' or 'nl'
DEVICE           = "cuda"                        # 'cuda' or 'cpu'
MAX_NEW_TOKENS   = 512

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from models.hf_model import HFModel
from tasks.task1_error_identification import Task1ErrorIdentification
from framework.graph_loader import GraphLoader

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Task 1 Model Test")
print(f"Model:        {MODEL_PATH}")
print(f"Graphs:       {GRAPH_IDS}")
print(f"Max scenarios:{N_SCENARIOS} per graph")
print(f"Context mode: {CONTEXT_MODE}")
print(f"{'='*60}\n")

# Load task and generate scenarios
task = Task1ErrorIdentification(
    graphs_dir='data/graphs/',
    scenarios_dir='data/scenarios/',
    context_mode=CONTEXT_MODE,
)
loader = GraphLoader('data/graphs/')

all_scenarios = task.generate_scenarios(graph_ids=GRAPH_IDS)

# Cap per graph
from collections import defaultdict
per_graph = defaultdict(list)
for s in all_scenarios:
    per_graph[s.graph_id].append(s)

scenarios = []
for gid in GRAPH_IDS:
    scenarios.extend(per_graph[gid][:N_SCENARIOS])

print(f"Running {len(scenarios)} scenarios total\n")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model = HFModel(
    model_id=MODEL_ID,
    model_name_or_path=MODEL_PATH,
    device=DEVICE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.0,
)
model.load()

# ---------------------------------------------------------------------------
# Run and score
# ---------------------------------------------------------------------------

results = []
for i, scenario in enumerate(scenarios):
    graph = loader.load(scenario.graph_id)
    print(f"[{i+1}/{len(scenarios)}] {scenario.scenario_id} ({scenario.metadata['complexity']})")

    result = task.run(model, scenario, graph)
    result.scores = task.score(result, scenario)
    results.append(result)

    # Print raw output and scores
    print(f"  Raw output:    {result.raw_output[:200].strip()}")
    print(f"  Parse success: {result.scores.get('parse_success', 0)}")
    print(f"  Scores:        {result.scores}")
    print()

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

model.unload()

print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")

agg = Task1ErrorIdentification.aggregate_scores(results)
for k, v in agg.items():
    print(f"  {k}: {v}")

# Per-graph breakdown
print(f"\nPer-graph breakdown:")
for gid in GRAPH_IDS:
    gid_results = [r for r in results if r.graph_id == gid]
    if gid_results:
        gid_agg = Task1ErrorIdentification.aggregate_scores(gid_results)
        print(f"  {gid}: composite={gid_agg.get('mean_composite_score', 0):.3f} "
              f"| parse_success={gid_agg.get('mean_parse_success', 0):.2f} "
              f"| hallucination={gid_agg.get('mean_hallucination_flag', 0):.2f} "
              f"| n={gid_agg.get('n_scenarios', 0)}")

# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")

parse_rate = agg.get('mean_parse_success', 0)
composite  = agg.get('mean_composite_score', 0)
halluc     = agg.get('mean_hallucination_flag', 0)

if parse_rate < 0.5:
    print("⚠ LOW PARSE RATE — model is not returning valid JSON.")
    print("  → Consider adding stronger JSON formatting instructions to the prompt.")
    print("  → Try context_mode='nl' which produces a shorter prompt.")
elif parse_rate < 0.8:
    print("⚠ MODERATE PARSE RATE — model sometimes returns invalid JSON.")
    print("  → Try lowering MAX_NEW_TOKENS or simplifying the output schema.")
else:
    print("✓ Parse rate is good.")

if composite < 0.3:
    print("⚠ LOW COMPOSITE SCORE — model struggles with this task.")
    print("  → Check if branch_mapping_score is the bottleneck.")
    print("  → Consider testing with context_mode='nl'.")
elif composite < 0.6:
    print("~ MODERATE COMPOSITE SCORE — room for improvement.")
else:
    print("✓ Composite score is reasonable for a 7B model.")

if halluc > 0.2:
    print("⚠ HIGH HALLUCINATION RATE — model is inventing node IDs.")
    print("  → This is expected on deep ladder graphs; check per-graph breakdown.")

print(f"\nDone. Raw results available in memory — add save_results() call if needed.")
