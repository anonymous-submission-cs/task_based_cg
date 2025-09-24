import argparse
import logging
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np

from src.data_generation.functions import (
    DIVERSE_FUNCTIONS,
    BaseFunction,
    MapRandom,
    apply_function_composition,
    apply_function_composition_uniform,
)
from src.data_generation.init import read_config

# initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"
from src.data_generation.utils import *


def sample_string():
    """
    Samples a string of the given length.
    """
    n_alphabets = 26
    seq_len = 6
    with_replacement = False
    alph = [chr(i + 97) for i in range(n_alphabets)]
    tokens = np.random.choice(alph, size=seq_len, replace=with_replacement)
    tokens = "".join(tokens)
    return tokens


def get_function_list_decoded(doc, sep_token, tokens):
    all_function_list = []
    for i in range(len(doc)):
        function_list = []
        fn_list = get_function_list(doc[i], sep_token)
        for fn in fn_list:
            function_list.append(tokens[fn])
        all_function_list.append(function_list)
    return all_function_list


def get_unique_functions(function_list):
    function_set = set()
    for i, fn_list in enumerate(function_list):
        function_set.add(tuple(fn_list))
    return [list(fn) for fn in function_set]


def make_config(prompt_mode, prompt_length, function_type, task_max_length):
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    cfg = read_config(cfg_path)
    cfg.prompt_length = prompt_length
    cfg.function.type = function_type
    cfg.task_max_length = task_max_length
    return cfg


def get_outputs(function_list, input_str_1, input_str_2, n_alphabets):
    filter_func = lambda x: x in "aeiou"
    offset = 1
    outputs = apply_function_composition(
        n_alphabets,
        function_list,
        DIVERSE_FUNCTIONS,
        input_str_1,
        input_str_2,
        filter_func,
        offset,
    )
    return outputs


def get_outputs_uniform(n_functions, function_list, input_str_1):
    function_dict = {
        f"map{i}": MapRandom.map_random(seed=i) for i in range(1, n_functions + 1)
    }
    function_dict["identity"] = BaseFunction.identity
    outputs = apply_function_composition_uniform(
        function_list,
        function_dict,
        input_str_1,
    )
    return outputs


def get_train_test_functions(args):
    prompt_mode = args.prompt_mode
    prompt_length = args.prompt_length
    model_split = args.model_split
    eval_split = args.eval_split
    nheads_nlayers = args.nheads_nlayers
    seq_len = args.seq_len
    n_functions = args.n_functions
    n_alphabets = args.n_alphabets
    max_task_length = args.max_task_length
    data_n_alphabets_seq_len_fn_len_max_task_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            n_alphabets, seq_len, n_functions, max_task_length
        )
    )
    train_split = model_split
    eval_split = eval_split
    function_type = args.function_type
    train_data_path = f"{ROOT_DIR}/data/{function_type}/{prompt_length}/{data_n_alphabets_seq_len_fn_len_max_task_length}/{prompt_mode}/{train_split}"
    test_data_path = f"{ROOT_DIR}/data/{function_type}/{prompt_length}/{data_n_alphabets_seq_len_fn_len_max_task_length}/{prompt_mode}/{eval_split}"

    train_docs = np.load(
        os.path.join(train_data_path, f"train_{prompt_mode}_corpus.npy")
    )
    test_docs = np.load(os.path.join(test_data_path, f"test_{prompt_mode}_corpus.npy"))
    token_idx_path = f"{ROOT_DIR}/data/{function_type}/{prompt_length}/{data_n_alphabets_seq_len_fn_len_max_task_length}/{prompt_mode}/{train_split}/token_idx.pkl"
    with open(token_idx_path, "rb") as f:
        token_idx = pickle.load(f)
    tokens_path = f"{ROOT_DIR}/data/{function_type}/{prompt_length}/{data_n_alphabets_seq_len_fn_len_max_task_length}/{prompt_mode}/{train_split}/token.pkl"
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    sep_token = token_idx["<SEP>"]
    train_function_list = get_function_list_decoded(train_docs, sep_token, tokens)
    test_function_list = get_function_list_decoded(test_docs, sep_token, tokens)
    # get all the unique functions in the train and test function lists
    train_functions = get_unique_functions(train_function_list)
    test_functions = get_unique_functions(test_function_list)
    return train_functions, test_functions


def get_equivalence_classes(
    train_functions, test_functions, num_trials=100, n_alphabets=26, max_seq_len=12
):
    equivalence_map = {}
    equivalence_class_map = {}

    for i, test_fn_list in enumerate(test_functions):
        all_trials = {}
        print("===== Test Function List {} =====".format(i))
        for trial in range(num_trials):
            # sample random input strings
            input_str_1 = sample_string()
            input_str_2 = sample_string()

            test_outputs = get_outputs(
                test_fn_list, input_str_1, input_str_2, n_alphabets
            )

            for train_fn_list in train_functions:
                train_outputs = get_outputs(
                    train_fn_list, input_str_1, input_str_2, n_alphabets
                )
                if train_outputs[-1][:max_seq_len] == test_outputs[-1][:max_seq_len]:
                    if tuple(train_fn_list) not in all_trials:
                        all_trials[tuple(train_fn_list)] = 1
                    else:
                        all_trials[tuple(train_fn_list)] += 1

        equivalence_class_map[tuple(test_fn_list)] = all_trials

    return equivalence_class_map


def process_single_test_function(test_fn_data):
    """
    Process a single test function across all trials and train functions.
    This function will be called by each worker process.
    """
    test_fn_list, train_functions, num_trials, n_alphabets, test_idx, max_seq_len = (
        test_fn_data
    )

    all_trials = {}
    print(f"===== Test Function List {test_idx} =====")

    for trial in range(num_trials):
        # sample random input strings
        input_str_1 = sample_string()
        input_str_2 = sample_string()

        test_outputs = get_outputs(test_fn_list, input_str_1, input_str_2, n_alphabets)

        for train_fn_list in train_functions:
            train_outputs = get_outputs(
                train_fn_list, input_str_1, input_str_2, n_alphabets
            )
            if train_outputs[-1][:max_seq_len] == test_outputs[-1][:max_seq_len]:
                if tuple(train_fn_list) not in all_trials:
                    all_trials[tuple(train_fn_list)] = 1
                else:
                    all_trials[tuple(train_fn_list)] += 1
                # first log test task and all the test outputs
                test_task = f"Test task: {test_fn_list}"
                all_test_outputs = []
                for i in range(len(test_outputs)):
                    all_test_outputs.append(test_outputs[i][:max_seq_len])
                all_train_outputs = []
                for i in range(len(train_outputs)):
                    all_train_outputs.append(train_outputs[i][:max_seq_len])
    return tuple(test_fn_list), all_trials


def process_single_test_function_uniform(test_fn_data):
    """
    Process a single test function across all trials and train functions.
    This function will be called by each worker process.
    """
    test_fn_list, train_functions, num_trials, n_functions, test_idx, max_seq_len = (
        test_fn_data
    )

    all_trials = {}
    print(f"===== Test Function List {test_idx} =====")

    for trial in range(num_trials):
        # sample random input strings
        input_str_1 = sample_string()
        test_outputs = get_outputs_uniform(n_functions, test_fn_list, input_str_1)

        for train_fn_list in train_functions:
            train_outputs = get_outputs_uniform(n_functions, train_fn_list, input_str_1)
            if train_outputs[-1][:max_seq_len] == test_outputs[-1][:max_seq_len]:
                if tuple(train_fn_list) not in all_trials:
                    all_trials[tuple(train_fn_list)] = 1
                else:
                    all_trials[tuple(train_fn_list)] += 1

    return tuple(test_fn_list), all_trials


def get_equivalence_classes_parallel_v1(
    train_functions,
    test_functions,
    num_trials=100,
    n_processes=None,
    n_alphabets=26,
    max_seq_len=12,
    n_functions=6,
    function_type="diverse",
    logger=None,
):
    """
    Parallelized version - each test function is processed by a separate worker.
    Good when you have many test functions.
    """
    if n_processes is None:
        n_processes = min(cpu_count(), len(test_functions))

    # Prepare data for parallel processing
    test_fn_data = []
    for i, test_fn_list in enumerate(test_functions):
        if function_type == "diverse":
            test_fn_data.append(
                (test_fn_list, train_functions, num_trials, n_alphabets, i, max_seq_len)
            )
        else:
            test_fn_data.append(
                (test_fn_list, train_functions, num_trials, n_functions, i, max_seq_len)
            )

    # Process in parallel
    with Pool(processes=n_processes) as pool:
        if function_type == "diverse":
            results = pool.map(process_single_test_function, test_fn_data)
        else:
            results = pool.map(process_single_test_function_uniform, test_fn_data)

    # Combine results
    equivalence_class_map = {}
    for test_fn_tuple, all_trials in results:
        equivalence_class_map[test_fn_tuple] = all_trials
    return equivalence_class_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--prompt_length", type=str, default="fixed", help="fixed or variable"
    )
    parser.add_argument(
        "--model_split",
        type=str,
        default="permutation_6_10",
        help="Model training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="permutation_6_10",
        help="Model evaluation split",
    )
    parser.add_argument(
        "--nheads_nlayers",
        type=str,
        default="nh6_nl3",
        help="number of heads and layers",
    )
    parser.add_argument(
        "--n_alphabets", type=int, default=26, help="number of alphabets"
    )
    parser.add_argument("--seq_len", type=int, default=6, help="sequence length")
    parser.add_argument(
        "--n_functions", type=int, default=6, help="number of functions"
    )
    parser.add_argument(
        "--pos_embedding_type", type=str, default="rel_global", help="abs or rel_global"
    )
    parser.add_argument("--num_trials", type=int, default=100, help="number of trials")
    parser.add_argument(
        "--function_type", type=str, default="diverse", help="uniform or diverse"
    )
    parser.add_argument(
        "--max_task_length", type=int, default=6, help="max task length"
    )
    parser.add_argument(
        "--n_processes", type=int, default=4, help="number of processes"
    )

    args = parser.parse_args()
    if args.function_type == "diverse":
        max_seq_len = 2 * args.seq_len
    else:
        max_seq_len = args.seq_len

    train_functions, test_functions = get_train_test_functions(args)

    # save the equivalence class map
    n_alphabets = args.n_alphabets
    seq_len = args.seq_len
    n_functions = args.n_functions
    max_task_length = args.max_task_length
    function_type = args.function_type
    prompt_length = args.prompt_length
    data_n_alphabets_seq_len_fn_len_max_task_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            n_alphabets, seq_len, n_functions, max_task_length
        )
    )
    path = f"{ROOT_DIR}/data/{function_type}/{prompt_length}/{data_n_alphabets_seq_len_fn_len_max_task_length}/{args.prompt_mode}/{args.model_split}/{args.eval_split}"
    os.makedirs(path, exist_ok=True)
    log_file_path = f"{path}/equivalence_class_map.log"
    print(f"Logging to {log_file_path}")
    equivalence_class_map = get_equivalence_classes_parallel_v1(
        train_functions,
        test_functions,
        n_alphabets=args.n_alphabets,
        n_processes=args.n_processes,
        max_seq_len=max_seq_len,
        num_trials=args.num_trials,
        function_type=args.function_type,
        n_functions=args.n_functions,
        logger=logger,
    )
    with open(f"{path}/equivalence_class_map.pkl", "wb") as f:
        pickle.dump(equivalence_class_map, f)
