"""
llm_judge.py
------------
LLM-as-judge for Task 5 trace completeness scoring.

Uses Llama-3.2-3B-Instruct as the judge model to evaluate
the quality and completeness of agent-generated trace logs
against reference traces.

The judge scores on two dimensions:
  - coverage    : does the trace cover all required decisions/actions?
  - justification: are the justifications accurate and relevant?

Returns a float 0.0-1.0 per trace.
"""

from __future__ import annotations
import json
from typing import Optional


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of
diagnostic trace logs in a manufacturing environment.
You will be given a reference trace and an agent-generated trace.
Evaluate the agent trace on two dimensions:
1. Coverage: does it document all key decisions and actions from the reference?
2. Justification quality: are the justifications accurate and relevant?
Return ONLY a JSON object with your scores."""

JUDGE_PROMPT_TEMPLATE = """REFERENCE TRACE:
{reference}

AGENT TRACE:
{agent}

Rate the agent trace on:
1. coverage (0.0-1.0): fraction of reference decisions/actions documented
2. justification_quality (0.0-1.0): accuracy and relevance of reasoning

Respond ONLY with:
{{
  "coverage": <float 0.0-1.0>,
  "justification_quality": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""


class LLMJudge:
    """
    LLM-as-judge for Task 5 trace evaluation.

    Usage:
        judge = LLMJudge(model_name_or_path="meta-llama/Llama-3.2-3B-Instruct")
        judge.load()
        score = judge.score(agent_trace, reference_trace)
        judge.unload()
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def load(self):
        from transformers import pipeline
        print(f"[LLMJudge] Loading {self.model_name_or_path}...")
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
            max_new_tokens=self.max_new_tokens,
        )
        print("[LLMJudge] Judge loaded.")

    def unload(self):
        del self._pipeline
        self._pipeline = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def score(
        self,
        agent_trace: dict,
        reference_trace: dict,
    ) -> float:
        """
        Score an agent trace against the reference.
        Returns a float 0.0-1.0 (mean of coverage and justification_quality).
        Falls back to heuristic scoring if model is not loaded.
        """
        if self._pipeline is None:
            return self._heuristic_score(agent_trace, reference_trace)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            reference=json.dumps(reference_trace, indent=2)[:800],
            agent=json.dumps(agent_trace, indent=2)[:800],
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self._pipeline(messages, return_full_text=False)
            raw = result[0]["generated_text"].strip()
            # Strip markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])
            parsed = json.loads(raw)
            coverage = float(parsed.get("coverage", 0.0))
            justification = float(parsed.get("justification_quality", 0.0))
            return round((coverage + justification) / 2, 4)
        except Exception as e:
            print(f"[LLMJudge] Scoring failed: {e}. Using heuristic.")
            return self._heuristic_score(agent_trace, reference_trace)

    def _heuristic_score(
        self,
        agent_trace: dict,
        reference_trace: dict,
    ) -> float:
        """
        Fallback heuristic when judge model is unavailable.
        Checks non-empty justifications in agent trace decisions.
        """
        decisions = agent_trace.get("decisions", [])
        if not decisions:
            return 0.0
        non_empty = sum(
            1 for d in decisions
            if d.get("justification", "").strip()
        )
        return round(non_empty / len(decisions), 4)

    def __call__(
        self,
        agent_trace: dict,
        reference_trace: dict,
    ) -> float:
        """Allow judge to be called directly as a function."""
        return self.score(agent_trace, reference_trace)
