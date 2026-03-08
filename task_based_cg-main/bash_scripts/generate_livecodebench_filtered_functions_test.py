import argparse
import json
import logging
import os
import pickle
from typing import Any, Dict, List

from datasets import load_dataset
from omegaconf import OmegaConf

from init import ROOT_DIR, read_config, set_seed
from scripts.generate_livecodebench import generate_augmented_samples


def _load_function_names(path: str) -> List[str]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected function names JSON to be a list of strings")
    return payload


def _default_output_dir(functions: List[str], per_function: int, seed: int) -> str:
    return os.path.join(
        ROOT_DIR,
        "data",
        "livecodebench",
        f"heldout_{len(functions)}_{per_function}",
        f"seed_{seed}",
    )


def _select_base_samples(
    records: List[Dict[str, Any]],
    function_names: List[str],
    id_field: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    indices_by_func: Dict[str, List[int]] = {name: [] for name in function_names}
    for idx, record in enumerate(records):
        func_name = record.get("function_name")
        if func_name in indices_by_func:
            indices_by_func[func_name].append(idx)

    base_samples: Dict[str, Dict[str, Any]] = {}
    for func_name in function_names:
        indices = indices_by_func.get(func_name, [])
        if not indices:
            raise ValueError(f"No records found for function '{func_name}'")
        base_idx = indices[-1]
        sample = records[base_idx]
        sample_id = sample.get(id_field, sample.get("id", base_idx))
        logger.info(
            "Selected base sample for %s: index=%d id=%s input=%s",
            func_name,
            base_idx,
            sample_id,
            sample.get("input", ""),
        )
        base_samples[func_name] = sample
    return base_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "config", "gen", "livecodebench.yaml"),
    )
    parser.add_argument(
        "--function_names_json",
        type=str,
        required=True,
        help="JSON list of function names to include (one base sample each).",
    )
    parser.add_argument("--per_function", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output dir. Defaults to data/livecodebench/heldout_<N>_<per_function>/seed_<seed>.",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("livecodebench_filtered_test")

    function_names = _load_function_names(args.function_names_json)
    output_dir = args.output_dir or _default_output_dir(function_names, args.per_function, cfg.seed)

    ds = load_dataset(cfg.dataset_name)
    if isinstance(ds, dict):
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split_name]
    records = list(ds)
    id_field = "id"

    base_samples = _select_base_samples(records, function_names, id_field, logger)

    all_test_samples: List[Dict[str, Any]] = []
    for offset, func_name in enumerate(function_names):
        base_sample = base_samples[func_name]
        test_samples = generate_augmented_samples(
            [base_sample],
            args.per_function,
            cfg,
            cfg.seed + 100 + offset,
            id_field,
            logger,
            label=f"test_{func_name}",
        )
        all_test_samples.extend(test_samples)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Writing test outputs to: %s", output_dir)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(all_test_samples, f)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "seed": cfg.seed,
                "dataset_name": cfg.dataset_name,
                "per_function": args.per_function,
                "function_names": function_names,
                "output_dir": output_dir,
                "generator": "generate_livecodebench_filtered_functions_test",
                "base_selection": "last_occurrence_per_function",
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
