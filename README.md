# Agentic AI Evaluation Framework for Manufacturing SOPs and SMPs

A multi-dimensional evaluation framework for assessing LLM-based agentic AI systems in industrial procedural control environments governed by Standard Operating Procedures (SOPs) and Standard Maintenance Procedures (SMPs).

This repository accompanies the journal paper:

> **Agentic AI in Industrial Procedural Control: A Multi-Dimensional Evaluation Framework**
> Anastasios Koukas, Laura Maruster, Christos Emmanouilidis
> *Annual Reviews in Control* (under review)

---

## Overview

Manufacturing environments governed by SOPs and SMPs represent a class of procedural control problems where correctness, safety, and role-appropriate escalation are non-negotiable. This framework evaluates whether LLM-based agentic systems can navigate these demands reliably.

The framework grounds evaluation in the **agentic perception–planning–action–reflection cycle**, adapted to manufacturing error-handling workflows, and defines **nine evaluation metrics** assessed across both **unit** and **pipeline** evaluation modes.

Key features:
- **Graph-aware evaluation** — SOPs/SMPs are represented as directed branching graphs (TOCAP/OCAP flowcharts), not linear step lists
- **Unit and pipeline modes** — tasks can be evaluated independently or chained to expose error propagation
- **Model-agnostic** — supports any LLM via vLLM or HuggingFace Transformers
- **Data-agnostic** — plug in any GRAPH JSON dataset without modifying the framework
- **Cross-lingual evaluation** — Dutch/English consistency testing across all tasks

---

## Repository Structure

```
agentic-eval-framework/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── graphs/                          # 50 GRAPH JSON documents (TOCAP/OCAP flowcharts)
│   ├── scenarios/                       # Generated test scenarios per task (auto-generated)
│   └── ground_truth/                    # Ground truth labels per task (auto-generated)
│
├── framework/
│   ├── __init__.py
│   ├── graph_loader.py                  # Load, validate, traverse GRAPH JSONs
│   ├── graph_utils.py                   # Path enumeration, node lookup, edge helpers
│   └── roles.py                         # Role hierarchy definitions and escalation rules
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py                     # Abstract base class all tasks inherit from
│   ├── task1_error_identification.py
│   ├── task2_instruction_retrieval.py
│   ├── task3_escalation.py
│   ├── task4_procedural_execution.py
│   ├── task5_trace_logging.py
│   └── task6_crosslingual.py            # Wrapper over tasks 1–5
│
├── prompts/
│   ├── system.yaml                      # Shared system prompt / agent context
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   ├── task4.yaml
│   └── task5.yaml
│
├── models/
│   ├── __init__.py
│   ├── base_model.py                    # Abstract model interface
│   ├── vllm_model.py                    # vLLM backend
│   └── hf_model.py                      # HuggingFace Transformers backend
│
├── scoring/
│   ├── __init__.py
│   ├── metrics.py                       # All 9 metric implementations
│   ├── graph_scorer.py                  # Graph-aware path scoring for Task 4
│   └── llm_judge.py                     # LLM-as-judge for Task 5 (Llama-3.2-3B)
│
├── pipeline/
│   ├── __init__.py
│   ├── unit_runner.py                   # Run tasks independently (unit mode)
│   ├── pipeline_runner.py               # Chain tasks 1→5 (pipeline mode)
│   └── crosslingual_runner.py           # Run all tasks with translated inputs
│
├── scenario_generation/
│   ├── __init__.py
│   ├── generate_scenarios.py            # Generate test scenarios from GRAPH JSONs
│   └── translate.py                     # Dutch/English translation for Task 6
│
├── results/
│   ├── raw/                             # Raw model outputs (auto-generated)
│   └── aggregated/                      # Scored results and summary tables (auto-generated)
│
├── scripts/
│   ├── run_evaluation.py                # Main entry point — full evaluation run
│   ├── run_single_task.py               # Run a single task for development/testing
│   └── generate_all_scenarios.py        # Pre-generate all scenarios from graphs
│
├── notebooks/
│   ├── dataset_analysis.ipynb           # Dataset characterisation
│   └── results_analysis.ipynb           # Results visualisation
│
└── tests/
    ├── test_graph_loader.py
    ├── test_scoring.py
    └── test_tasks.py
```

---

## Dataset

The framework is validated on **50 synthetic GRAPH JSON documents** derived from real TOCAP/OCAP flowcharts used in a large manufacturing environment. Each document represents a troubleshooting or maintenance procedure encoded as a directed branching graph.

### Document structure

Each GRAPH JSON has the following format:

```json
{
  "id": "GRAPH21",
  "description": "Expanded procedure for tracing the root cause of damaged inserts.",
  "graph": {
    "nodes": {
      "box_0": { "type": "terminator", "text": "Start" },
      "box_1": { "type": "decision",   "text": "Is the insert supply from the hopper correct?" },
      "box_2": { "type": "process",    "text": "Check whether inserts come out of the hopper evenly." },
      "box_3": { "type": "document",   "text": "Refer to hopper supply diagnostic procedure." },
      "box_4": { "type": "terminator", "text": "Escalate to maintenance team." }
    },
    "edges": [
      { "source": "box_0", "target": "box_1", "label": null },
      { "source": "box_1", "target": "box_2", "label": "No" },
      { "source": "box_1", "target": "box_4", "label": "Yes" },
      { "source": "box_2", "target": "box_1", "label": null }
    ]
  }
}
```

**Node types:**
| Type | Description |
|---|---|
| `terminator` | Start or end of procedure (including escalation/resolution endpoints) |
| `decision` | Yes/No branching condition |
| `process` | Action or diagnostic step |
| `document` | Reference to a sub-procedure or external document |

### Dataset statistics

| Property | Value |
|---|---|
| Total documents | 50 |
| Average nodes per document | 31.3 |
| Average decision nodes | 8.4 |
| Average edges | 36.2 |
| Documents with sub-procedure references | 40 / 50 |
| Graph patterns | Linear: 1, Short tree (1–3 dec): 3, Medium chain (4–7 dec): 21, Deep ladder (8+ dec): 25 |
| Escalation types | Escalation: 25, Delegate (maintenance): 10, Fixed (QA/Eng): 9, Role-inferred: 3, Other: 3 |

---

## Evaluation Framework

### Agentic cycle

The framework maps the four stages of the agentic AI cycle to manufacturing error-handling:

| Stage | Manufacturing mapping |
|---|---|
| **Perception** | Identify error type, interpret SOP/SMP data |
| **Planning** | Determine action path and escalation level |
| **Action** | Generate or guide through required operational steps |
| **Reflection** | Verify outcomes and log results |

### Role hierarchy

| Level | Role | Scope |
|---|---|---|
| Level 1 | Operator | Sets up and operates machinery under supervision |
| Level 2 | Technical Operator | Minor technical interventions per defined procedures |
| Level 3 | Mechanic | Diagnoses machine issues, reports to manufacturing leader |
| Level 4 | Maintenance Engineer | Advanced diagnostics — final human escalation point |

### Evaluation metrics

| Metric | Cycle stage | Description | Range | Tasks |
|---|---|---|---|---|
| Error Identification Accuracy | Perception | % of errors correctly classified | 0–100 ↑ | 1 |
| Interaction Efficiency | Perception → Planning | Avg clarification turns to correct instruction | 1–10 ↓ | 2 |
| Escalation Appropriateness | Planning | % escalated to correct authority level | 0–100 ↑ | 3 |
| Procedural Accuracy | Action | % SOP steps correctly executed vs ground truth | 0–100 ↑ | 4 |
| Hallucination Rate | All | % scenarios with hallucinated advice | 0–100 ↓ | All |
| Trace Completeness | Reflection | Coverage vs reference trace (LLM-as-judge) | 0–100 ↑ | 5 |
| General Consistency | All | Output stability across repeated runs | 0–100 ↑ | All |
| Cross-Lingual Consistency | All | Performance parity across Dutch/English input | 0–100 ↑ | 6 |
| Robustness Index | All | Performance under noisy/incomplete input | 0–100 ↑ | All |

### Evaluation modes

**Unit mode** — each task is evaluated independently against isolated ground truth. Used to characterise per-task, per-model capability.

**Pipeline mode** — tasks are chained (Task 1 → 2 → 3 → 4 → 5). The output of each task becomes part of the context for the next. Used to expose error propagation across the full agentic workflow.

---

## Tasks

### Task 1 — Error Identification and Classification *(Perception)*
The agent detects anomalies from textual descriptions and maps them to the correct branch of the diagnostic flowchart. Scored on classification accuracy and branch-mapping correctness.

### Task 2 — Instruction Retrieval and Clarification *(Perception → Planning)*
The agent retrieves the correct diagnostic instructions from a multi-document context, using chain-of-thought or dialogue if instructions are incomplete. Scored on retrieval accuracy and interaction efficiency.

### Task 3 — Role-Based Reasoning and Escalation *(Planning)*
The agent determines whether the user role is appropriate for the detected error and escalates to the correct level if not. Three cases tested per scenario: no escalation needed, escalation required, out-of-scope (escalate to Maintenance Engineer).

### Task 4 — Procedural Execution and Faithfulness *(Action)*
The agent navigates the branching TOCAP/OCAP graph, following the correct Yes/No path at each decision node. Scored with graph-aware branch precision — tracking correct node traversal, not just final answer correctness.

### Task 5 — Outcome Reporting and Trace Logging *(Action → Reflection)*
The agent summarises its decisions and generates an auditable trace log. Scored by an LLM-as-judge (Llama-3.2-3B) against a reference trace, assessing decision justification alignment.

### Task 6 — Cross-Lingual Consistency *(Cross-phase)*
Tasks 1–5 are re-run with operator inputs translated to Dutch. Per-task consistency profiles measure performance degradation under language transfer.

---

## Models

The framework is model-agnostic. Any model accessible via vLLM or HuggingFace Transformers can be plugged in. Validated models include:

| Model | Parameters | Backend |
|---|---|---|
| Qwen2.5-7B-Instruct | 7B | vLLM / HF |
| Qwen2.5-14B-Instruct | 14B | vLLM / HF |
| Llama-3.1-8B-Instruct | 8B | vLLM / HF |
| Mistral-7B-Instruct-v0.3 | 7B | vLLM / HF |

---

## Installation

```bash
git clone https://github.com/roeczak/agentic-eval-framework.git
cd agentic-eval-framework
pip install -r requirements.txt
```

### Requirements

```
vllm
transformers
torch
pyyaml
numpy
pandas
scikit-learn
deep-translator
tqdm
pytest
```

---

## Usage

### Generate scenarios from GRAPH documents

```bash
python scripts/generate_all_scenarios.py --data_dir data/graphs/ --output_dir data/scenarios/
```

### Run full evaluation (all tasks, all models, unit mode)

```bash
python scripts/run_evaluation.py \
  --model qwen2.5-7b-instruct \
  --mode unit \
  --tasks all \
  --scenarios_dir data/scenarios/ \
  --output_dir results/
```

### Run full pipeline evaluation

```bash
python scripts/run_evaluation.py \
  --model qwen2.5-7b-instruct \
  --mode pipeline \
  --scenarios_dir data/scenarios/ \
  --output_dir results/
```

### Run a single task during development

```bash
python scripts/run_single_task.py \
  --task task1 \
  --model qwen2.5-7b-instruct \
  --graph GRAPH21 \
  --output_dir results/raw/
```

---

## Project Context

This framework was developed as part of the **AIXPERT** project (Horizon Europe, ID 101214389) at the University of Groningen. The industrial co-creation study was conducted with a manufacturing partner operating complex SOP/SMP-governed production lines.

The conference paper precursor to this work was presented at IFAC 2026 World Congress:
> Koukas, A., Raza, S., Maruster, L., Emmanouilidis, C. (2026). * An Evaluation Framework for Agentic AI in Manufacturing Standard Operating and Maintenance Procedures*

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{koukas2026agentic,
  title={Agentic AI in Industrial Procedural Control: A Multi-Dimensional Evaluation Framework},
  author={Koukas, Anastasios and Maruster, Laura and Emmanouilidis, Christos},
  journal={Annual Reviews in Control},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
