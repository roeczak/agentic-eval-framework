"""
translate.py
------------
Translation utilities for Task 6 cross-lingual scenario generation.

Handles batch translation of scenario input fields from English to
Dutch, German, and Polish using deep-translator (Google Translate backend).

Results are cached to disk to avoid redundant API calls across runs.

Usage:
    from scenario_generation.translate import TranslationPipeline

    pipeline = TranslationPipeline(cache_dir='data/translations/')
    dutch_text = pipeline.translate("Check whether the cap is damaged", "nl")
    batch = pipeline.translate_batch(texts, "de")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class TranslationPipeline:
    """
    Wraps deep-translator with disk-based caching.
    Falls back gracefully if deep-translator is not installed.
    """

    SUPPORTED = {
        "nl": "Dutch",
        "de": "German",
        "pl": "Polish",
    }

    def __init__(self, cache_dir: str = "data/translations/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self._cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def _cache_key(self, text: str, lang: str) -> str:
        return f"{lang}::{text[:300]}"

    def translate(self, text: str, target_lang: str) -> str:
        """Translate a single string. Uses cache if available."""
        if target_lang not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported language: '{target_lang}'. "
                f"Supported: {list(self.SUPPORTED.keys())}"
            )

        key = self._cache_key(text, target_lang)
        if key in self._cache:
            return self._cache[key]

        translated = self._call_translator(text, target_lang)
        self._cache[key] = translated
        self._save_cache()
        return translated

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        show_progress: bool = False,
    ) -> list[str]:
        """Translate a list of strings, using cache where available."""
        results = []
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                print(f"  Translating {i+1}/{len(texts)} to {target_lang}...")
            results.append(self.translate(text, target_lang))
        return results

    def _call_translator(self, text: str, target_lang: str) -> str:
        """Call deep-translator. Falls back to original text on failure."""
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(
                source="en",
                target=target_lang,
            ).translate(text) or text
        except ImportError:
            print(
                "[TranslationPipeline] deep-translator not installed. "
                "Install with: pip install deep-translator"
            )
            return text
        except Exception as e:
            print(f"[TranslationPipeline] Translation error ({target_lang}): {e}")
            return text

    def translate_scenario_inputs(
        self,
        scenario_input: dict,
        fields: list[str],
        target_lang: str,
    ) -> dict:
        """
        Translate specified fields of a scenario input dict.
        Returns a new dict with translated fields.
        """
        translated = dict(scenario_input)
        for field in fields:
            if field in translated and isinstance(translated[field], str):
                translated[field] = self.translate(translated[field], target_lang)
        return translated

    def cache_stats(self) -> dict:
        """Return stats about the translation cache."""
        by_lang = {}
        for key in self._cache:
            lang = key.split("::")[0]
            by_lang[lang] = by_lang.get(lang, 0) + 1
        return {
            "total_cached": len(self._cache),
            "by_language": by_lang,
        }
