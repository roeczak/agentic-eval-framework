"""
vllm_model.py
-------------
vLLM backend for high-throughput inference on GPU.
Recommended for actual evaluation runs on the HPC.

vLLM provides significantly faster inference than HuggingFace pipelines
for batch evaluation, especially for larger models.
"""

from __future__ import annotations
from .base_model import BaseModel


class VLLMModel(BaseModel):
    """
    vLLM backend using the LLM class for offline batch inference.

    Usage:
        model = VLLMModel(
            model_id="qwen2.5-7b",
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,   # set to number of GPUs
        )
        model.load()
        output = model.generate("What is the error type?")
        model.unload()
    """

    def __init__(
        self,
        model_id: str,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ):
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self._llm = None
        self._sampling_params = None

    def load(self):
        from vllm import LLM, SamplingParams

        print(f"[VLLMModel] Loading {self.model_name_or_path} "
              f"(tensor_parallel={self.tensor_parallel_size})...")

        self._llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
        )
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        print(f"[VLLMModel] Model loaded.")

    def unload(self):
        del self._llm
        self._llm = None
        self._sampling_params = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[VLLMModel] Model unloaded.")

    def generate(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format as chat-style prompt using tokenizer's apply_chat_template
        # vLLM accepts raw strings; we wrap in a simple user turn
        outputs = self._llm.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate responses for a batch of prompts in one call.
        More efficient than calling generate() in a loop.
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
