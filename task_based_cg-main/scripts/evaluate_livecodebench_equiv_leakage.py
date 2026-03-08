"""
LiveCodeBench evaluation for equiv_leakage datasets with optional LoRA adapters.
"""
import argparse
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from init import ROOT_DIR, read_config, set_seed
from src.evaluation.livecodebench_evaluator import LiveCodeBenchEvaluator

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


class LiveCodeBenchLoRAEvaluator(LiveCodeBenchEvaluator):
    def _load_model_and_tokenizer(self):
        model, tokenizer = super()._load_model_and_tokenizer()
        if getattr(self.cfg, "lora_path", None):
            if PeftModel is None:
                raise ImportError(
                    "peft is required for LoRA evaluation. Install with `pip install peft`."
                )
            model = PeftModel.from_pretrained(model, self.cfg.lora_path, is_trainable=False)
            self.logger.info("Loaded LoRA adapter from: %s", self.cfg.lora_path)
            model.eval()
            model.to(self.device)
        else:
            self.logger.info("No LoRA adapter provided; running base model only.")
        return model, tokenizer

    def evaluate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._evaluate_samples(samples)

    def _evaluate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        from src.evaluation.livecodebench_evaluator import LiveCodeBenchMetrics

        metrics = LiveCodeBenchMetrics(total=0, exact_match=0, char_matches=0, char_total=0)
        predictions = []

        self.logger.info("Loaded %d samples for split=%s", len(samples), self.cfg.split)
        batch_size = max(1, int(self.cfg.batch_size))
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            prompts = [self._format_prompt(sample) for sample in batch_samples]
            batch_start = time.time()
            preds = self._generate_outputs(prompts)
            batch_elapsed = time.time() - batch_start
            per_sample_time = batch_elapsed / max(1, len(batch_samples))
            for offset, (sample, pred) in enumerate(zip(batch_samples, preds), start=1):
                idx = start_idx + offset
                self.logger.info(
                    "Sample %d/%d runtime: %.4fs",
                    idx,
                    len(samples),
                    per_sample_time,
                )
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


def _random_identifier(rng: random.Random, length: int = 12) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return rng.choice(alphabet) + "".join(rng.choice(alphabet) for _ in range(length - 1))


def _anonymize_sample(sample: Dict[str, Any], new_name: str) -> Dict[str, Any]:
    func_name = sample.get("function_name")
    if not func_name:
        return sample
    pattern = r"\b" + re.escape(func_name) + r"\b"
    code = re.sub(pattern, new_name, sample.get("code", ""))
    inp = re.sub(pattern, new_name, sample.get("input", ""))
    new_sample = dict(sample)
    new_sample["code"] = code
    new_sample["input"] = inp
    new_sample["function_name"] = new_name
    new_sample["anonymized_function_name"] = new_name
    return new_sample


def _load_samples(path: str, max_samples: int) -> List[Dict[str, Any]]:
    import pickle

    with open(path, "rb") as f:
        samples = pickle.load(f)
    if max_samples and max_samples > 0:
        return samples[: max_samples]
    return samples


def _resolve_split_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}.pkl")


def _build_results_path(cfg, data_dir: str, split: str, suffix: str) -> str:
    base_dir = os.path.join(ROOT_DIR, "models", "eval", "livecodebench_equiv_leakage")
    data_subdir = ""
    if data_dir:
        marker = os.path.join("data", "livecodebench") + os.sep
        if marker in data_dir:
            data_subdir = data_dir.split(marker, 1)[-1].strip(os.sep)
        else:
            data_subdir = os.path.basename(os.path.normpath(data_dir))
    if data_subdir:
        base_dir = os.path.join(base_dir, data_subdir)
    model_tag = f"model_{cfg.model_name}"
    base_dir = os.path.join(base_dir, model_tag, split)
    results_subdir = getattr(cfg, "results_subdir", None)
    if results_subdir:
        base_dir = os.path.join(base_dir, results_subdir)
    if getattr(cfg, "lora_path", None):
        lora_tag = os.path.basename(os.path.normpath(cfg.lora_path))
        base_dir = os.path.join(base_dir, f"lora_{lora_tag}")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"results{suffix}.json")


def main(cfg, data_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("livecodebench_eval_equiv")

    set_seed(cfg.seed)
    data_path = _resolve_split_path(data_dir, cfg.split)
    samples = _load_samples(data_path, cfg.max_samples)

    evaluator = LiveCodeBenchLoRAEvaluator(cfg, logger=logger)

    normal_results = evaluator.evaluate_samples(samples)
    normal_path = _build_results_path(cfg, data_dir, cfg.split, "")
    with open(normal_path, "w") as f:
        import json

        json.dump(normal_results, f, indent=2)
    logger.info("Saved results to: %s", normal_path)

    rng = random.Random(cfg.seed)
    anonymized_samples = []
    preview_original = None
    preview_anonymized = None
    for idx, sample in enumerate(samples):
        new_name = _random_identifier(rng)
        anonymized = _anonymize_sample(sample, new_name)
        if preview_original is None:
            preview_original = sample
            preview_anonymized = anonymized
        anonymized_samples.append(anonymized)
    if preview_original and preview_anonymized:
        logger.info("Anonymization preview (function_name: %s -> %s)", preview_original.get("function_name"), preview_anonymized.get("function_name"))
        logger.info("Original code:\n%s", preview_original.get("code", ""))
        logger.info("Anonymized code:\n%s", preview_anonymized.get("code", ""))
        logger.info("Original input: %s", preview_original.get("input", ""))
        logger.info("Anonymized input: %s", preview_anonymized.get("input", ""))
    anonymized_results = evaluator.evaluate_samples(anonymized_samples)
    anon_path = _build_results_path(cfg, data_dir, cfg.split, "_anon")
    with open(anon_path, "w") as f:
        import json

        json.dump(anonymized_results, f, indent=2)
    logger.info("Saved anonymized results to: %s", anon_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=f"{ROOT_DIR}/config/eval/livecodebench.yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing train.pkl/test.pkl.",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--include_fewshot", type=bool, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--results_subdir",
        type=str,
        default=None,
        help="Optional subdir to insert before the lora tag (e.g., shared fraction).",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.split is not None:
        cfg.split = args.split
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.include_fewshot is not None:
        cfg.include_fewshot = args.include_fewshot
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lora_path is not None:
        cfg.lora_path = args.lora_path
    if args.results_subdir is not None:
        cfg.results_subdir = args.results_subdir

    if not args.data_dir:
        raise ValueError("--data_dir is required for equiv_leakage evaluation")

    main(cfg, args.data_dir)
