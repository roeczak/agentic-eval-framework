"""
hf_model.py
-----------
HuggingFace Transformers backend.
Useful for development, testing, and HPC runs without vLLM.

Supports any instruction-tuned model available on HuggingFace Hub
or loaded from a local path.
"""

from __future__ import annotations
import torch
from .base_model import BaseModel


class HFModel(BaseModel):
    """
    HuggingFace Transformers backend using the text-generation pipeline.

    Usage:
        model = HFModel(
            model_id="qwen2.5-7b",
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            device="cuda",
        )
        model.load()
        output = model.generate("What is the error type?")
        model.unload()
    """

    def __init__(
        self,
        model_id: str,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,       # 0.0 = greedy (deterministic)
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.temperature = temperature
        self.torch_dtype = getattr(torch, torch_dtype)
        self._pipeline = None

    def load(self):
        from transformers import pipeline, AutoTokenizer

        print(f"[HFModel] Loading {self.model_name_or_path} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name_or_path,
            tokenizer=self._tokenizer,
            device_map="auto",
            torch_dtype=self.torch_dtype,
        )
        print(f"[HFModel] Model loaded.")

    def unload(self):
        del self._pipeline
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[HFModel] Model unloaded.")

    def generate(self, prompt: str) -> str:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format as chat message
        messages = [
            {"role": "user", "content": prompt}
        ]

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        result = self._pipeline(messages, **gen_kwargs)
        return result[0]["generated_text"].strip()
