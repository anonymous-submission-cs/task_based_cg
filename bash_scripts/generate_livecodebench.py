import argparse
import ast
import builtins
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Any, Dict, List, Tuple, Optional

from datasets import load_dataset
from omegaconf import OmegaConf

from init import ROOT_DIR, read_config, set_seed


SAFE_IMPORT_ALLOWLIST = {
    "math",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "operator",
    "typing",
    "random",
    "numpy",
}

BUILTINS_BLACKLIST = {
    "eval",
    "exec",
    "open",
    "input",
    "compile",
    "breakpoint",
    "help",
    "exit",
    "quit",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in SAFE_IMPORT_ALLOWLIST:
        raise ImportError(f"Import of '{name}' is not allowed")
    return importlib.import_module(name)


def build_safe_builtins() -> Dict[str, Any]:
    safe_builtins = dict(builtins.__dict__)
    for key in BUILTINS_BLACKLIST:
        safe_builtins.pop(key, None)
    safe_builtins["__import__"] = safe_import
    return safe_builtins


def parse_input_call(input_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    expr = ast.parse(input_str, mode="eval").body
    if not isinstance(expr, ast.Call):
        raise ValueError("Input is not a function call")
    if isinstance(expr.func, ast.Name):
        func_name = expr.func.id
    elif isinstance(expr.func, ast.Attribute):
        func_name = expr.func.attr
    else:
        raise ValueError("Unsupported function call")
    args = [ast.literal_eval(arg) for arg in expr.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
    return func_name, args, kwargs


def _bounded_int_range(value: int, cfg) -> Tuple[int, int]:
    scale = max(10, abs(value) * 2)
    low = -scale if value < 0 else 0
    high = scale
    if cfg.int_low is not None:
        low = max(low, cfg.int_low)
    if cfg.int_high is not None:
        high = min(high, cfg.int_high)
    if low > high:
        low, high = high, low
    return low, high


def _bounded_float_range(value: float, cfg) -> Tuple[float, float]:
    scale = max(10.0, abs(value) * 2.0)
    low = -scale if value < 0 else 0.0
    high = scale
    if cfg.float_low is not None:
        low = max(low, cfg.float_low)
    if cfg.float_high is not None:
        high = min(high, cfg.float_high)
    if low > high:
        low, high = high, low
    return low, high


def randomize_value(value: Any, rng: random.Random, cfg) -> Any:
    if isinstance(value, bool):
        return rng.choice([True, False])
    if isinstance(value, int):
        low, high = _bounded_int_range(value, cfg)
        return rng.randint(low, high)
    if isinstance(value, float):
        low, high = _bounded_float_range(value, cfg)
        return rng.uniform(low, high)
    if isinstance(value, str):
        if not value:
            return value
        return "".join(rng.choice(cfg.string_alphabet) for _ in range(len(value)))
    if isinstance(value, list):
        return [randomize_value(v, rng, cfg) for v in value]
    if isinstance(value, tuple):
        return tuple(randomize_value(v, rng, cfg) for v in value)
    if isinstance(value, dict):
        return {k: randomize_value(v, rng, cfg) for k, v in value.items()}
    return value


def build_input_str(func_name: str, args: List[Any], kwargs: Dict[str, Any]) -> str:
    args_str = ", ".join(repr(a) for a in args)
    kwargs_str = ", ".join(f"{k} = {repr(v)}" for k, v in kwargs.items())
    if args_str and kwargs_str:
        return f"{func_name}({args_str}, {kwargs_str})"
    if args_str:
        return f"{func_name}({args_str})"
    return f"{func_name}({kwargs_str})"


def _exec_worker(code: str, func_name: str, args: List[Any], kwargs: Dict[str, Any], result_q):
    try:
        exec_globals = {"__builtins__": build_safe_builtins()}
        exec_globals["typing"] = importlib.import_module("typing")
        exec_globals["List"] = exec_globals["typing"].List
        exec_globals["Dict"] = exec_globals["typing"].Dict
        exec_globals["Tuple"] = exec_globals["typing"].Tuple
        exec_globals["Set"] = exec_globals["typing"].Set
        exec_globals["Optional"] = exec_globals["typing"].Optional
        exec_globals["Deque"] = exec_globals["typing"].Deque
        exec_globals["DefaultDict"] = exec_globals["typing"].DefaultDict
        exec_globals["Counter"] = exec_globals["typing"].Counter
        exec_globals["Iterable"] = exec_globals["typing"].Iterable
        exec_globals["Iterator"] = exec_globals["typing"].Iterator
        exec_globals["Sequence"] = exec_globals["typing"].Sequence
        exec_globals["Mapping"] = exec_globals["typing"].Mapping
        exec_globals["MutableMapping"] = exec_globals["typing"].MutableMapping
        exec_globals["MutableSequence"] = exec_globals["typing"].MutableSequence
        exec_globals["MutableSet"] = exec_globals["typing"].MutableSet
        exec_globals["collections"] = importlib.import_module("collections")
        exec_globals["math"] = importlib.import_module("math")
        exec_globals["functools"] = importlib.import_module("functools")
        exec_globals["itertools"] = importlib.import_module("itertools")
        exec_globals["numpy"] = importlib.import_module("numpy")
        exec_globals["np"] = exec_globals["numpy"]
        exec_globals["heapq"] = importlib.import_module("heapq")
        exec_globals["bisect"] = importlib.import_module("bisect")
        exec_globals["operator"] = importlib.import_module("operator")
        exec_globals["deque"] = exec_globals["collections"].deque
        exec_globals["defaultdict"] = exec_globals["collections"].defaultdict
        exec_globals["Counter"] = exec_globals["collections"].Counter
        exec_globals["gcd"] = exec_globals["math"].gcd
        exec_globals["inf"] = exec_globals["math"].inf
        exec_globals["comb"] = exec_globals["math"].comb
        exec_globals["prod"] = exec_globals["math"].prod
        exec_globals["reduce"] = exec_globals["functools"].reduce
        exec_globals["cache"] = exec_globals["functools"].cache
        exec_globals["lru_cache"] = exec_globals["functools"].lru_cache
        exec_globals["accumulate"] = exec_globals["itertools"].accumulate
        exec_globals["islice"] = exec_globals["itertools"].islice
        exec_globals["combinations"] = exec_globals["itertools"].combinations
        exec_globals["heapify"] = exec_globals["heapq"].heapify
        exec_globals["heappush"] = exec_globals["heapq"].heappush
        exec_globals["heappop"] = exec_globals["heapq"].heappop
        exec(code, exec_globals, exec_globals)
        target = exec_globals.get(func_name)
        if target is None:
            solution_cls = exec_globals.get("Solution")
            if solution_cls is not None:
                target = getattr(solution_cls(), func_name, None)
        if target is None:
            raise ValueError(f"Function '{func_name}' not found")
        result = target(*args, **kwargs)
        result_q.put(("ok", result))
    except Exception as exc:
        result_q.put(("err", repr(exc)))


def run_with_timeout(code: str, func_name: str, args: List[Any], kwargs: Dict[str, Any], timeout: float):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(
        target=_exec_worker,
        args=(code, func_name, args, kwargs, result_q),
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "Timeout"
    if result_q.empty():
        return False, "No result"
    status, payload = result_q.get()
    if status == "ok":
        return True, payload
    return False, payload


def select_split_ids(question_ids: List[Any], cfg, rng: random.Random) -> List[Any]:
    if cfg.holdout_question_ids:
        return list(cfg.holdout_question_ids)
    unique_ids = list(set(question_ids))
    rng.shuffle(unique_ids)
    return unique_ids[: cfg.holdout_size]


def build_equiv_leakage_split(
    records: List[Dict[str, Any]],
    id_field: str,
    heldout_functions: List[str],
    shared_fraction: float,
    holdout_size: int,
    rng: random.Random,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    heldout_set = set(heldout_functions)
    record_ids = [get_record_id(r, idx, id_field) for idx, r in enumerate(records)]
    func_to_indices: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        func_name = record.get("function_name")
        if func_name in heldout_set:
            func_to_indices.setdefault(func_name, []).append(idx)

    test_records: List[Dict[str, Any]] = []
    test_ids: set = set()

    for func_name, indices in func_to_indices.items():
        total = len(indices)
        n_test = max(1, int((1.0 - shared_fraction) * total))
        n_train = total - n_test
        for idx in indices[n_train:]:
            rec_id = record_ids[idx]
            if rec_id in test_ids:
                continue
            test_ids.add(rec_id)
            test_records.append(records[idx])
        logger.info(
            "Leakage split %s: total=%d train=%d test=%d",
            func_name,
            total,
            n_train,
            n_test,
        )

    if len(test_records) < holdout_size:
        remaining_indices = [
            idx
            for idx, rec_id in enumerate(record_ids)
            if rec_id not in test_ids
            and records[idx].get("function_name") not in heldout_set
        ]
        needed = holdout_size - len(test_records)
        if needed > len(remaining_indices):
            logger.warning(
                "Requested holdout_size=%d but only %d records available after leakage split.",
                holdout_size,
                len(test_records) + len(remaining_indices),
            )
            needed = len(remaining_indices)
        extra_indices = rng.sample(remaining_indices, k=needed) if needed > 0 else []
        for idx in extra_indices:
            rec_id = record_ids[idx]
            if rec_id in test_ids:
                continue
            test_ids.add(rec_id)
            test_records.append(records[idx])
    elif len(test_records) > holdout_size:
        logger.warning(
            "Leakage split produced %d test records; truncating to holdout_size=%d.",
            len(test_records),
            holdout_size,
        )
        ordered_indices = []
        for idx, rec_id in enumerate(record_ids):
            if rec_id in test_ids:
                ordered_indices.append(idx)
            if len(ordered_indices) >= holdout_size:
                break
        test_records = [records[idx] for idx in ordered_indices]
        test_ids = {record_ids[idx] for idx in ordered_indices}

    train_records = [
        r for r, rec_id in zip(records, record_ids)
        if rec_id not in test_ids
    ]
    return train_records, test_records, test_ids


def build_equiv_leakage_function_fraction_split(
    records: List[Dict[str, Any]],
    id_field: str,
    heldout_functions: List[str],
    leaked_functions: List[str],
    shared_fraction: float,
    holdout_size: int,
    rng: random.Random,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    heldout_set = set(heldout_functions)
    leaked_set = set(leaked_functions)
    record_ids = [get_record_id(r, idx, id_field) for idx, r in enumerate(records)]
    func_to_indices: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        func_name = record.get("function_name")
        if func_name in heldout_set:
            func_to_indices.setdefault(func_name, []).append(idx)

    test_records: List[Dict[str, Any]] = []
    test_ids: set = set()

    for func_name, indices in func_to_indices.items():
        total = len(indices)
        if func_name in leaked_set:
            n_test = max(1, int((1.0 - shared_fraction) * total))
        else:
            n_test = total
        n_train = total - n_test
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for idx in shuffled[n_train:]:
            rec_id = record_ids[idx]
            if rec_id in test_ids:
                continue
            test_ids.add(rec_id)
            test_records.append(records[idx])
        logger.info(
            "Leakage split %s: leaked=%s total=%d train=%d test=%d",
            func_name,
            func_name in leaked_set,
            total,
            n_train,
            n_test,
        )

    if len(test_records) < holdout_size:
        remaining_indices = [
            idx
            for idx, rec_id in enumerate(record_ids)
            if rec_id not in test_ids
            and records[idx].get("function_name") not in heldout_set
        ]
        needed = holdout_size - len(test_records)
        if needed > len(remaining_indices):
            logger.warning(
                "Requested holdout_size=%d but only %d records available after leakage split.",
                holdout_size,
                len(test_records) + len(remaining_indices),
            )
            needed = len(remaining_indices)
        extra_indices = rng.sample(remaining_indices, k=needed) if needed > 0 else []
        for idx in extra_indices:
            rec_id = record_ids[idx]
            if rec_id in test_ids:
                continue
            test_ids.add(rec_id)
            test_records.append(records[idx])
    elif len(test_records) > holdout_size:
        logger.warning(
            "Leakage split produced %d test records; truncating to holdout_size=%d.",
            len(test_records),
            holdout_size,
        )
        ordered_indices = []
        for idx, rec_id in enumerate(record_ids):
            if rec_id in test_ids:
                ordered_indices.append(idx)
            if len(ordered_indices) >= holdout_size:
                break
        test_records = [records[idx] for idx in ordered_indices]
        test_ids = {record_ids[idx] for idx in ordered_indices}

    train_records = [
        r for r, rec_id in zip(records, record_ids)
        if rec_id not in test_ids
    ]
    return train_records, test_records, test_ids


def get_record_id(record: Dict[str, Any], idx: int, id_field: str) -> Any:
    if id_field == "__index__":
        return idx
    value = record.get(id_field)
    if value is None and id_field != "id":
        value = record.get("id")
    if value is None:
        return idx
    return value


def load_json_list(path: str, key: Optional[str] = None, label: str = "values") -> List[Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if key and key in payload:
            payload = payload[key]
        elif "ids" in payload:
            payload = payload["ids"]
        elif "id" in payload:
            payload = payload["id"]
    if not isinstance(payload, list):
        if key:
            detail = f"a list or dict with a '{key}' field"
        else:
            detail = "a list"
        raise ValueError(f"Expected {label} JSON to be {detail}")
    return payload


def load_filtered_ids(path: str) -> List[Any]:
    return load_json_list(path, label="filtered ids")


def generate_augmented_samples(
    base_samples: List[Dict[str, Any]],
    target_count: int,
    cfg,
    seed: int,
    id_field: str,
    logger: logging.Logger,
    label: str,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    results = []
    seen = set()
    max_attempts = target_count * cfg.max_attempts_per_sample
    attempts = 0
    if not base_samples:
        logger.warning("%s split has no base samples; skipping generation.", label)
        return results
    start_time = time.time()
    log_every = max(1, target_count // 2000)
    attempt_log_every = max(1000, log_every * 5)
    logged_first = False
    logger.info(
        "[%s] starting generation target=%d max_attempts=%d",
        label,
        target_count,
        max_attempts,
    )
    while len(results) < target_count and attempts < max_attempts:
        sample = rng.choice(base_samples)
        sample_id = sample.get(id_field)
        if sample_id is None:
            sample_id = sample.get("id")
        if sample_id is None:
            sample_id = rng.randrange(0, 1_000_000_000)
        try:
            func_name, args, kwargs = parse_input_call(sample["input"])
        except Exception:
            logger.warning("Failed to parse input call: %s", sample["input"])
            attempts += 1
            continue
        target_func_name = sample["function_name"] or func_name

        for _ in range(max(1, cfg.retry_per_sample)):
            if attempts % 1000 == 0 and attempts > 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[%s] heartbeat attempts=%d success=%d rate=%.2f samples/s",
                    label,
                    attempts,
                    len(results),
                    rate,
                )
            if attempts >= max_attempts or len(results) >= target_count:
                break
            attempts += 1
            new_args = [randomize_value(a, rng, cfg) for a in args]
            new_kwargs = {k: randomize_value(v, rng, cfg) for k, v in kwargs.items()}
            new_input = build_input_str(target_func_name, new_args, new_kwargs)
            key = (sample_id, new_input)
            if key in seen:
                continue
            ok, output = run_with_timeout(
                sample["code"],
                target_func_name,
                new_args,
                new_kwargs,
                cfg.timeout_seconds,
            )
            if not ok:
                logger.warning(
                    'Failed to execute sample_id=%s input="%s": %s',
                    sample_id,
                    new_input,
                    output,
                )
                _dump_failure(
                    cfg,
                    sample_id,
                    label,
                    str(output),
                    sample["code"],
                    sample["input"],
                    new_input,
                )
                if attempts % attempt_log_every == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "[%s] attempts=%d success=%d rate=%.2f samples/s last_error=%s",
                        label,
                        attempts,
                        len(results),
                        rate,
                        output,
                    )
                continue
            seen.add(key)
            record = dict(sample)
            record["orig_input"] = sample["input"]
            record["orig_output"] = sample["output"]
            record["split_id"] = sample_id
            record["input"] = new_input
            record["output"] = output
            results.append(record)
            if not logged_first or len(results) % log_every == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[%s] progress %d/%d (attempts=%d, rate=%.2f samples/s)",
                    label,
                    len(results),
                    target_count,
                    attempts,
                    rate,
                )
                logger.info(
                    "[%s] sample input: %s",
                    label,
                    new_input,
                )
                logger.info(
                    "[%s] sample output: %s",
                    label,
                    repr(output),
                )
                logged_first = True
            break
    if attempts >= max_attempts and len(results) < target_count:
        elapsed = time.time() - start_time
        logger.warning(
            "[%s] stopped after max_attempts=%d with %d/%d samples (elapsed=%.1fs)",
            label,
            max_attempts,
            len(results),
            target_count,
            elapsed,
        )
    return results


def get_output_dir(cfg, shard_id: int, num_shards: int) -> str:
    mode = getattr(cfg, "mode", "random")
    mode_suffix = mode
    if mode == "equiv_leakage":
        shared = getattr(cfg, "shared_fraction", None)
        if shared is None:
            shared = "unknown"
        heldout_fraction = getattr(cfg, "heldout_function_fraction", None)
        if heldout_fraction is None:
            mode_suffix = f"{mode}_shared_{shared}"
        else:
            mode_suffix = f"{mode}_shared_{shared}_heldoutfrac_{heldout_fraction}"
    return os.path.join(
        ROOT_DIR,
        "data",
        "livecodebench",
        mode_suffix,
        f"holdout_{cfg.holdout_size}",
        f"seed_{cfg.seed}",
        f"shards_{num_shards}",
        f"shard_{shard_id}",
    )


def get_shard_count(total: int, shard_id: int, num_shards: int) -> int:
    base = total // num_shards
    remainder = total % num_shards
    return base + (1 if shard_id < remainder else 0)


def _dump_failure(
    cfg,
    sample_id: Any,
    label: str,
    reason: str,
    code: str,
    orig_input: str,
    new_input: str,
):
    if not cfg.debug_dump_failures:
        return
    dump_dir = os.path.join(ROOT_DIR, "a_logs", "livecodebench_failures")
    os.makedirs(dump_dir, exist_ok=True)
    fname = f"{label}_sample_{sample_id}.txt"
    path = os.path.join(dump_dir, fname)
    with open(path, "w") as f:
        f.write("reason: " + reason + "\n")
        f.write("orig_input: " + orig_input + "\n")
        f.write("new_input: " + new_input + "\n\n")
        f.write(code)


def main(cfg, shard_id: int, num_shards: int):
    set_seed(cfg.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("livecodebench_gen")
    logger.setLevel(logging.INFO)
    ds = load_dataset(cfg.dataset_name)
    if isinstance(ds, dict):
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split_name]
    records = list(ds)
    id_field = "id"
    if cfg.filtered_ids_json:
        total_records = len(records)
        filtered_ids = set(load_filtered_ids(cfg.filtered_ids_json))
        records = [
            r for idx, r in enumerate(records)
            if get_record_id(r, idx, id_field) in filtered_ids
        ]
        logger.info("Filtered ids file: %s", cfg.filtered_ids_json)
        logger.info("Filtered base size: %d (from %d)", len(records), total_records)
    question_ids = [get_record_id(r, idx, id_field) for idx, r in enumerate(records)]

    rng = random.Random(cfg.seed)
    mode = getattr(cfg, "mode", "random")
    if mode == "equiv_leakage":
        if not getattr(cfg, "heldout_function_names_json", None):
            raise ValueError("equiv_leakage mode requires heldout_function_names_json")
        if getattr(cfg, "shared_fraction", None) is None:
            raise ValueError("equiv_leakage mode requires shared_fraction")
        if not 0.0 <= cfg.shared_fraction <= 1.0:
            raise ValueError("shared_fraction must be between 0 and 1")
        heldout_functions = load_json_list(
            cfg.heldout_function_names_json,
            label="heldout function names",
        )
        heldout_fraction = getattr(cfg, "heldout_function_fraction", None)
        if heldout_fraction is not None:
            if not 0.0 <= heldout_fraction <= 1.0:
                raise ValueError("heldout_function_fraction must be between 0 and 1")
            total_functions = len(heldout_functions)
            target_count = int(math.ceil(heldout_fraction * total_functions))
            rng_functions = random.Random(cfg.seed + 1000)
            leaked_functions = (
                rng_functions.sample(heldout_functions, k=target_count)
                if target_count > 0
                else []
            )
            logger.info(
                "Heldout function fraction: %.3f (%d/%d)",
                heldout_fraction,
                len(leaked_functions),
                total_functions,
            )
            train_base, test_base, holdout_ids = build_equiv_leakage_function_fraction_split(
                records,
                id_field,
                heldout_functions,
                leaked_functions,
                cfg.shared_fraction,
                cfg.holdout_size,
                rng,
                logger,
            )
        else:
            train_base, test_base, holdout_ids = build_equiv_leakage_split(
                records,
                id_field,
                heldout_functions,
                cfg.shared_fraction,
                cfg.holdout_size,
                rng,
                logger,
            )
    else:
        holdout_ids = set(select_split_ids(question_ids, cfg, rng))
        train_base = [
            r for idx, r in enumerate(records)
            if get_record_id(r, idx, id_field) not in holdout_ids
        ]
        test_base = [
            r for idx, r in enumerate(records)
            if get_record_id(r, idx, id_field) in holdout_ids
        ]

    logger.info("Dataset size: %d", len(records))
    logger.info("Split field: %s", id_field)
    logger.info("Split mode: %s", mode)
    logger.info("Holdout size: %d", len(holdout_ids))
    logger.info("Train base size: %d", len(train_base))
    logger.info("Test base size: %d", len(test_base))
    logger.info("Target train samples: %d", cfg.train_size)
    logger.info("Target test samples: %d", cfg.test_size)
    logger.info("Shard %d/%d", shard_id, num_shards)

    shard_train_size = get_shard_count(cfg.train_size, shard_id, num_shards)
    shard_test_size = get_shard_count(cfg.test_size, shard_id, num_shards)
    logger.info("Shard train target: %d", shard_train_size)
    logger.info("Shard test target: %d", shard_test_size)

    train_samples = generate_augmented_samples(
        train_base, shard_train_size, cfg, cfg.seed + 1 + shard_id, id_field, logger, "train"
    )
    test_samples = generate_augmented_samples(
        test_base, shard_test_size, cfg, cfg.seed + 2 + shard_id, id_field, logger, "test"
    )

    output_dir = get_output_dir(cfg, shard_id, num_shards)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Writing outputs to: %s", output_dir)

    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)
    with open(os.path.join(output_dir, "split_question_ids.pkl"), "wb") as f:
        pickle.dump(
            {"holdout": list(holdout_ids), "id_field": id_field, "shard_id": shard_id, "num_shards": num_shards},
            f,
        )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "config", "gen", "livecodebench.yaml"),
    )
    parser.add_argument("--holdout_size", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["random", "equiv_leakage"],
        help="Split mode: random (default) or equiv_leakage.",
    )
    parser.add_argument(
        "--filtered_ids_json",
        type=str,
        default=None,
        help="Optional JSON file containing a list of LiveCodeBench ids to include.",
    )
    parser.add_argument(
        "--heldout_function_names_json",
        type=str,
        default=None,
        help="JSON file with a list of function names used in equiv_leakage mode.",
    )
    parser.add_argument(
        "--shared_fraction",
        type=float,
        default=None,
        help="Fraction of samples per heldout function that remain shared (0-1).",
    )
    parser.add_argument(
        "--heldout_function_fraction",
        type=float,
        default=None,
        help="Fraction of heldout functions to include in leakage split (0-1).",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.holdout_size is not None:
        cfg.holdout_size = args.holdout_size
    if args.train_size is not None:
        cfg.train_size = args.train_size
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.mode is not None:
        cfg.mode = args.mode
    cfg.filtered_ids_json = args.filtered_ids_json
    if args.heldout_function_names_json is not None:
        cfg.heldout_function_names_json = args.heldout_function_names_json
    if args.heldout_function_fraction is not None:
        cfg.heldout_function_fraction = args.heldout_function_fraction
    if args.shared_fraction is not None:
        cfg.shared_fraction = args.shared_fraction
    main(cfg, args.shard_id, args.num_shards)
