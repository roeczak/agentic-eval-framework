"""
task6_crosslingual.py
----------------------
Task 6: Cross-Lingual Consistency (Cross-phase)

Task 6 is not a standalone task — it is a wrapper that re-runs Tasks 1–5
with operator inputs translated into non-English languages (Dutch, German,
Polish) while keeping the procedure graph in English.

This reflects the real industrial setting: TOCAP/OCAP documents are in
English (or the plant's primary language), but operators may report faults
in their native language.

For each base scenario from Tasks 1–5, Task 6 generates N translated
variants — one per target language. The translated variant is scored
identically to its English counterpart. Cross-lingual consistency is then
computed as the performance ratio between translated and English variants:

  consistency_score = mean_score_translated / mean_score_english

A score of 1.0 means no degradation under language transfer.
A score < 1.0 quantifies degradation.

Per-task consistency profiles are reported separately, testing whether
lexically demanding tasks (T2, T4) degrade more than classification
tasks (T1, T3).

Translation:
  - Uses deep-translator (Google Translate backend) by default
  - Translation is cached to avoid redundant API calls
  - translate.py in scenario_generation/ handles the translation pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tasks.base_task import BaseTask, Scenario, TaskResult
from tasks.task1_error_identification import Task1ErrorIdentification
from tasks.task2_instruction_retrieval import Task2InstructionRetrieval
from tasks.task3_escalation import Task3Escalation
from tasks.task4_procedural_execution import Task4ProceduralExecution
from tasks.task5_trace_logging import Task5TraceLogging
from framework.graph_loader import Graph


# Fields to translate per task
TASK_INPUT_FIELDS = {
    "task1": ["symptom_description"],
    "task2": ["user_query"],
    "task3": ["fault_description"],
    "task4": ["fault_context"],
    "task5": ["fault_context", "completed_path_summary"],
}

SUPPORTED_LANGUAGES = {
    "nl": "Dutch",
    "de": "German",
    "pl": "Polish",
}


class Task6CrossLingual:
    """
    Cross-lingual consistency wrapper for Tasks 1–5.

    Usage:
        wrapper = Task6CrossLingual(
            base_tasks={
                'task1': Task1ErrorIdentification(...),
                'task2': Task2InstructionRetrieval(...),
                ...
            },
            target_languages=['nl', 'de'],
            translation_cache_dir='data/translations/',
        )

        # Generate translated scenarios from base scenarios
        translated = wrapper.generate_translated_scenarios(
            base_scenarios,   # list of Scenario from any task
        )

        # Run and score translated scenarios
        results = wrapper.run_and_score(model, translated_scenarios)

        # Compute consistency profiles
        consistency = wrapper.compute_consistency(
            english_results, translated_results
        )
    """

    TASK_ID = "task6"

    def __init__(
        self,
        base_tasks: dict,
        target_languages: Optional[list[str]] = None,
        translation_cache_dir: str = "data/translations/",
        graphs_dir: str = "data/graphs/",
    ):
        self.base_tasks = base_tasks   # dict: task_id -> BaseTask instance
        self.target_languages = target_languages or ["nl", "de", "pl"]
        self.translation_cache_dir = Path(translation_cache_dir)
        self.translation_cache_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir = graphs_dir
        self._translation_cache: dict[str, dict[str, str]] = {}
        self._load_translation_cache()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _load_translation_cache(self):
        """Load cached translations from disk."""
        cache_file = self.translation_cache_dir / "translation_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                self._translation_cache = json.load(f)

    def _save_translation_cache(self):
        """Persist translation cache to disk."""
        cache_file = self.translation_cache_dir / "translation_cache.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self._translation_cache, f, indent=2, ensure_ascii=False)

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text to target_lang.
        Results are cached to avoid redundant API calls.

        Falls back gracefully if deep-translator is unavailable.
        """
        cache_key = f"{target_lang}::{text[:200]}"
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(
                source="en",
                target=target_lang,
            ).translate(text)
        except Exception as e:
            print(f"[Task6] Translation failed ({target_lang}): {e}. "
                  f"Using original text.")
            translated = text

        self._translation_cache[cache_key] = translated
        self._save_translation_cache()
        return translated

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
    ) -> list[str]:
        """Translate a batch of texts, using cache where possible."""
        return [self.translate(t, target_lang) for t in texts]

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def generate_translated_scenarios(
        self,
        base_scenarios: list[Scenario],
        target_languages: Optional[list[str]] = None,
    ) -> list[Scenario]:
        """
        Generate translated variants of base scenarios.

        For each base scenario, creates one variant per target language.
        Only the operator-facing input fields are translated
        (see TASK_INPUT_FIELDS). The graph context and ground truth
        remain in English.
        """
        languages = target_languages or self.target_languages
        translated_scenarios = []

        for scenario in base_scenarios:
            task_id = scenario.task_id
            fields_to_translate = TASK_INPUT_FIELDS.get(task_id, [])

            for lang_code in languages:
                lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)

                # Translate relevant input fields
                translated_input = dict(scenario.input)
                for field in fields_to_translate:
                    if field in translated_input:
                        original_text = translated_input[field]
                        translated_input[field] = self.translate(
                            original_text, lang_code
                        )

                translated_scenario = Scenario(
                    scenario_id=f"{scenario.scenario_id}_{lang_code}",
                    graph_id=scenario.graph_id,
                    task_id=task_id,
                    input=translated_input,
                    ground_truth=scenario.ground_truth,  # unchanged
                    metadata={
                        **scenario.metadata,
                        "language": lang_code,
                        "language_name": lang_name,
                        "base_scenario_id": scenario.scenario_id,
                        "is_translated": True,
                    },
                )
                translated_scenarios.append(translated_scenario)

        return translated_scenarios

    # ------------------------------------------------------------------
    # Run and score
    # ------------------------------------------------------------------

    def run_and_score(
        self,
        model,
        scenarios: list[Scenario],
    ) -> list[TaskResult]:
        """
        Run and score translated scenarios using the appropriate base task.
        """
        from framework.graph_loader import GraphLoader
        loader = GraphLoader(self.graphs_dir)
        results = []

        for scenario in scenarios:
            task_id = scenario.task_id
            base_task = self.base_tasks.get(task_id)
            if base_task is None:
                print(f"[Task6] No base task found for {task_id}, skipping.")
                continue

            graph = loader.load(scenario.graph_id)

            # Task 4 uses step-by-step runner
            if task_id == "task4" and hasattr(base_task, "run_stepwise"):
                result = base_task.run_stepwise(model, scenario, graph)
            else:
                result = base_task.run(model, scenario, graph)

            result.scores = base_task.score(result, scenario)

            # Tag result as cross-lingual
            result.metadata["language"] = scenario.metadata.get("language")
            result.metadata["language_name"] = scenario.metadata.get("language_name")
            result.metadata["base_scenario_id"] = scenario.metadata.get(
                "base_scenario_id"
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Consistency computation
    # ------------------------------------------------------------------

    def compute_consistency(
        self,
        english_results: list[TaskResult],
        translated_results: list[TaskResult],
    ) -> dict:
        """
        Compute cross-lingual consistency scores.

        For each task and language, computes:
          consistency = mean_score_translated / mean_score_english

        Returns a nested dict:
          {task_id: {lang_code: {metric: consistency_score}}}
        """
        primary_metrics = {
            "task1": "composite_score",
            "task2": "composite_score",
            "task3": "composite_score",
            "task4": "decision_accuracy",
            "task5": "trace_completeness",
        }

        # Index English results by (task_id, scenario_id)
        en_by_task: dict[str, list[TaskResult]] = {}
        for r in english_results:
            en_by_task.setdefault(r.task_id, []).append(r)

        consistency = {}

        for task_id, metric in primary_metrics.items():
            en_results = en_by_task.get(task_id, [])
            if not en_results:
                continue

            en_mean = self._mean_metric(en_results, metric)
            if en_mean == 0:
                continue

            consistency[task_id] = {}

            for lang_code in self.target_languages:
                lang_results = [
                    r for r in translated_results
                    if r.task_id == task_id
                    and r.metadata.get("language") == lang_code
                ]
                if not lang_results:
                    continue

                lang_mean = self._mean_metric(lang_results, metric)
                consistency_score = lang_mean / en_mean

                consistency[task_id][lang_code] = {
                    "primary_metric": metric,
                    "english_mean": round(en_mean, 4),
                    "translated_mean": round(lang_mean, 4),
                    "consistency_score": round(consistency_score, 4),
                    "n_english": len(en_results),
                    "n_translated": len(lang_results),
                }

        return consistency

    def _mean_metric(
        self,
        results: list[TaskResult],
        metric: str,
    ) -> float:
        scored = [r for r in results if r.scores and metric in r.scores]
        if not scored:
            return 0.0
        return sum(r.scores[metric] for r in scored) / len(scored)

    # ------------------------------------------------------------------
    # Aggregate consistency report
    # ------------------------------------------------------------------

    def consistency_report(
        self,
        consistency: dict,
    ) -> str:
        """
        Format the consistency results as a readable report.
        Suitable for printing during experiments.
        """
        lines = ["\n=== Cross-Lingual Consistency Report ===\n"]
        for task_id, lang_scores in consistency.items():
            lines.append(f"  {task_id}:")
            for lang_code, scores in lang_scores.items():
                lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
                cs = scores["consistency_score"]
                en = scores["english_mean"]
                tr = scores["translated_mean"]
                flag = "✓" if cs >= 0.90 else ("~" if cs >= 0.75 else "⚠")
                lines.append(
                    f"    {flag} {lang_name:8s}: {cs:.3f} "
                    f"(EN={en:.3f} → {lang_name[:2].upper()}={tr:.3f})"
                )
        lines.append("")
        return "\n".join(lines)
