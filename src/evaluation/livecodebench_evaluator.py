import json
import logging
import os
import pickle
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from init import ROOT_DIR, set_seed
from src.models.pretrained import (
    load_gpt_oss_20b,
    load_granite_2b,
    load_gemma_1b,
    load_llama3_8b,
    load_deepseek_coder_1_3b
)


@dataclass
class LiveCodeBenchMetrics:
    total: int
    exact_match: int
    char_matches: int
    char_total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.exact_match / self.total

    @property
    def char_accuracy(self) -> float:
        if self.char_total == 0:
            return 0.0
        return self.char_matches / self.char_total


class LiveCodeBenchEvaluator:
    def __init__(self, cfg, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("livecodebench_eval")
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        if self.cfg.model_name == "llama3":
            model, tokenizer, _ = load_llama3_8b()
        elif self.cfg.model_name == "gpt2":
            model, tokenizer, _ = load_gpt_oss_20b()
        elif self.cfg.model_name == "granite":
            model, tokenizer, _ = load_granite_2b()
        elif self.cfg.model_name == "gemma1":
            model, tokenizer, _ = load_gemma_1b()
        elif self.cfg.model_name == "deepseek_coder":
            model, tokenizer, _ = load_deepseek_coder_1_3b()
        else:
            raise ValueError(f"Unsupported model_name: {self.cfg.model_name}")
        model.eval()
        model.to(self.device)
        return model, tokenizer

    def _build_data_path(self) -> str:
        if self.cfg.data_path:
            return self.cfg.data_path
        base_dir = os.path.join(
            ROOT_DIR,
            "data",
            "livecodebench",
            f"holdout_{self.cfg.holdout_size}",
            f"seed_{self.cfg.data_seed}",
        )
        return os.path.join(base_dir, f"{self.cfg.split}.pkl")

    def _load_samples(self) -> List[Dict[str, Any]]:
        data_path = self._build_data_path()
        with open(data_path, "rb") as f:
            samples = pickle.load(f)
        if self.cfg.max_samples and self.cfg.max_samples > 0:
            return samples[: self.cfg.max_samples]
        return samples

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        code = sample.get("code", "").rstrip()
        inp = sample.get("input", "")
        instruction = (
            "You are given a Python function and an assertion containing an input to "
            "the function. Complete the assertion with a literal (no unsimplified "
            "expressions, no function calls) containing the output when executing the "
            "provided code on the given input, even if the function is incorrect or "
            "incomplete. Do NOT output any extra information. Provide the full "
            "assertion with the correct output in [ANSWER] and [/ANSWER] tags.\n\n"
        )
        instruction_with_examples = (
            "You are given a Python function and an assertion containing an input to "
            "the function. Complete the assertion with a literal (no unsimplified "
            "expressions, no function calls) containing the output when executing the "
            "provided code on the given input, even if the function is incorrect or "
            "incomplete. Do NOT output any extra information. Provide the full "
            "assertion with the correct output in [ANSWER] and [/ANSWER] tags, "
            "following the examples.\n\n"
        )
        fewshot = (
            "[PYTHON]\n"
            "def repeatNumber(number : int) -> int:\n"
            "    return number\n"
            "assert repeatNumber(number = 17) == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
            "assert repeatNumber(number = 17) == 17\n"
            "[/ANSWER]\n\n"
            "[PYTHON]\n"
            "def addCharacterA(string : str) -> str:\n"
            "    return string + \"a\"\n"
            "assert addCharacterA(string = \"x9j\") == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
            "assert addCharacterA(string = \"x9j\") == \"x9ja\"\n"
            "[/ANSWER]\n\n"
        )
        prompt = (
            "[PYTHON]\n"
            f"{code}\n"
            f"assert {inp} == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
        )
        if self.cfg.include_fewshot:
            return instruction_with_examples + fewshot + prompt
        return instruction + prompt

    def _normalize_output(self, output: Any) -> str:
        if isinstance(output, str):
            text = output
        else:
            text = repr(output)
        return text.strip()

    @staticmethod
    def _parse_string_literal(text: str) -> tuple[bool, Optional[str]]:
        """Parse a Python string literal like `'abc'` or `"abc"`.

        Only string literals are accepted. Non-string literals (lists/ints/etc.)
        are intentionally ignored so existing non-string matching behavior is unchanged.
        """
        try:
            value = ast.literal_eval(text)
        except Exception:
            return False, None
        if isinstance(value, str):
            return True, value
        return False, None

    def _is_match(self, pred_literal: Any, gold_output: Any) -> int:
        """Return exact-match flag with tolerant handling for quoted string literals.

        Example accepted equivalence: `pred_literal == "'abc'"` and `gold_output == "abc"`.
        Non-string structures (e.g., lists) still use strict normalized string equality.
        """
        pred_norm = self._normalize_output(pred_literal)
        gold_norm = self._normalize_output(gold_output)
        if pred_norm == gold_norm:
            return 1

        pred_is_str_lit, pred_unquoted = self._parse_string_literal(pred_norm)
        gold_is_str_lit, gold_unquoted = self._parse_string_literal(gold_norm)

        # Only relax when exactly one side is a quoted Python string literal.
        if pred_is_str_lit and not gold_is_str_lit and pred_unquoted == gold_norm:
            return 1
        if gold_is_str_lit and not pred_is_str_lit and gold_unquoted == pred_norm:
            return 1
        return 0

    @staticmethod
    def _extract_answer_literal(text: str) -> str:
        if not text:
            return ""
        end_idx = text.find("[/ANSWER]")
        if end_idx == -1:
            end_idx = len(text)
        segment = text[:end_idx]
        if "==" in segment:
            segment = segment.split("==", 1)[-1]
        return segment.strip()

    def _generate_output(self, prompt: str) -> str:
        return self._generate_outputs([prompt])[0]

    def _generate_outputs(self, prompts: List[str]) -> List[str]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        do_sample = self.cfg.temperature and self.cfg.temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
        outputs = []
        for i in range(output_ids.shape[0]):
            input_len = int(inputs["attention_mask"][i].sum().item())
            gen_ids = output_ids[i][input_len:]
            outputs.append(
                self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            )
        return outputs

    def evaluate(self) -> Dict[str, Any]:
        set_seed(self.cfg.seed)
        samples = self._load_samples()
        metrics = LiveCodeBenchMetrics(total=0, exact_match=0, char_matches=0, char_total=0)
        predictions = []

        self.logger.info("Loaded %d samples for split=%s", len(samples), self.cfg.split)
        batch_size = max(1, int(self.cfg.batch_size))
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            prompts = [self._format_prompt(sample) for sample in batch_samples]
            preds = self._generate_outputs(prompts)
            for offset, (sample, pred) in enumerate(zip(batch_samples, preds), start=1):
                idx = start_idx + offset
                pred_literal = self._extract_answer_literal(pred)
                pred_norm = self._normalize_output(pred_literal)
                gold_norm = self._normalize_output(sample.get("output", ""))
                match = self._is_match(pred_literal, sample.get("output", ""))
                char_matches, char_total = self._char_match_stats(pred_norm, gold_norm)

                metrics.total += 1
                metrics.exact_match += match
                metrics.char_matches += char_matches
                metrics.char_total += char_total
                predictions.append(
                    {
                        "input": sample.get("input", ""),
                        "prediction": pred,
                        "prediction_literal": pred_literal,
                        "gold": sample.get("output", ""),
                        "match": match,
                        "char_matches": char_matches,
                        "char_total": char_total,
                        "char_accuracy": (char_matches / char_total) if char_total else 0.0,
                    }
                )
                if idx % self.cfg.log_every == 0:
                    self.logger.info(
                        "Progress %d/%d | acc=%.4f | char_acc=%.4f",
                        idx,
                        len(samples),
                        metrics.accuracy,
                        metrics.char_accuracy,
                    )
                    preview_count = min(2, len(predictions))
                    if preview_count:
                        for preview in predictions[-preview_count:]:
                            self.logger.info(
                                "Sample input: %s",
                                preview.get("input", ""),
                            )
                            self.logger.info(
                                "Sample pred: %s",
                                preview.get("prediction", ""),
                            )
                            self.logger.info(
                                "Sample gold: %s",
                                preview.get("gold", ""),
                            )

        return {
            "metrics": {
                "total": metrics.total,
                "exact_match": metrics.exact_match,
                "accuracy": metrics.accuracy,
                "char_matches": metrics.char_matches,
                "char_total": metrics.char_total,
                "char_accuracy": metrics.char_accuracy,
            },
            "predictions": predictions,
        }

    @staticmethod
    def _char_match_stats(pred: str, gold: str) -> tuple[int, int]:
        if not pred and not gold:
            return 0, 0
        total = max(len(pred), len(gold))
        matches = sum(
            1 for i in range(min(len(pred), len(gold))) if pred[i] == gold[i]
        )
        return matches, total

    def save_results(self, results: Dict[str, Any]) -> str:
        output_dir = os.path.join(
            ROOT_DIR,
            "models",
            "eval",
            "livecodebench",
            f"holdout_{self.cfg.holdout_size}",
            f"seed_{self.cfg.data_seed}",
            f"model_{self.cfg.model_name}",
            self.cfg.split,
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
