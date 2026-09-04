"""
hf_model.py
-----------
HuggingFace Transformers backend using direct model/tokenizer loading
instead of the pipeline() API, to avoid generation_config conflicts.
"""

from __future__ import annotations
import torch
from .base_model import BaseModel


class HFModel(BaseModel):

    def __init__(
        self,
        model_id: str,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        torch_dtype: str = "float16",
    ):
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.temperature = temperature
        self.torch_dtype = getattr(torch, torch_dtype)
        self._model = None
        self._tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[HFModel] Loading {self.model_name_or_path} on {self.device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            clean_up_tokenization_spaces=False,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self._model.eval()
        print(f"[HFModel] Model loaded.")

    def unload(self):
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[HFModel] Model unloaded.")

    def generate(self, prompt: str) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format as chat message using tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
        ).to(self._model.device)

        # Build generation kwargs explicitly — no generation_config conflict
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
