import argparse
import os
import pickle

import numpy as np
import pandas as pd
from src.evaluation.tsne_plots import plot_tsne_with_equivalence_class_score, plot_tsne_with_equivalence_class_score_with_slider

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"
from src.data_generation.init import read_config
from src.evaluation.tsne_plots import plot_tsne_with_equivalence_class_score, plot_tsne_with_equivalence_class_score_with_slider


def load_data_equivalence_class_map(path_prefix):
    with open(f"{path_prefix}/data_equivalence_class_map.pkl", "rb") as f:
        data_equivalence_class_map = pickle.load(f)
    return data_equivalence_class_map


def load_model_equivalence_class_map(path_prefix):
    with open(f"{path_prefix}/model_equivalence_class_map.pkl", "rb") as f:
        model_equivalence_class_map = pickle.load(f)
    return model_equivalence_class_map


def load_representation_results(path_prefix, replace_strings_repr):
    with open(
        f"{path_prefix}/representation_results_{replace_strings_repr}.pkl", "rb"
    ) as f:
        print(f"Loading representation results from {path_prefix}/representation_results_{replace_strings_repr}.pkl")
        representation_results = pickle.load(f)
    return representation_results


def load_train_test_accs(accs_path, functions_info_path):
    with open(accs_path, "rb") as f:
        train_test_accs = pickle.load(f)

    with open(functions_info_path, "rb") as f:
        functions_info = pickle.load(f)

    ck, accs = train_test_accs[0]

    test_comb_acc = accs["test"]["combination"]["acc"]
    train_comb_acc = accs["train"]["combination"]["acc"]
    train_acc = accs["train"]["total"]["acc"]
    test_acc = accs["test"]["total"]["acc"]

    test_combination_accs = {}
    train_combination_accs = {}
    for cid, acc in test_comb_acc.items():
        # find corresponding function names
        function_id = [k for k, v in functions_info.items() if v == cid][0]
        test_combination_accs[function_id] = acc

    for cid, acc in train_comb_acc.items():
        # find corresponding function names
        function_id = [k for k, v in functions_info.items() if v == cid][0]
        train_combination_accs[function_id] = acc

    return train_acc, test_acc, train_combination_accs, test_combination_accs


def get_representation_df(tsne, replace_strings_repr):
    embeddings = tsne["embeddings"]
    perm_labels = np.array(tsne["perm_labels"])
    dataset_labels = np.array(tsne["dataset_labels"])
    unique_perms = np.array(tsne["unique_perms"], dtype=object)
    simplified_perms = np.array(tsne["simplified_perms"], dtype=object)
    input_strings = np.array(tsne["input_strings"])
    
    predicted_output_strings = []
    if "predicted_output_strings" in tsne.keys():
        predicted_output_label = "predicted_output_strings"
    else:
        predicted_output_label = "output_strings"
    for s in tsne[predicted_output_label]:
        predicted_output_strings.append("".join(s))
    
    if "actual_output_strings" in tsne.keys():
        actual_output_label = "actual_output_strings"
    else:
        actual_output_label = "output_strings"
    actual_output_strings = []
    for s in tsne[actual_output_label]:
        actual_output_strings.append("".join(s))


    df = pd.DataFrame(embeddings, columns=["x", "y"])
    df["perm_label"] = perm_labels
    df["perm_tuple"] = [tuple(row) for row in simplified_perms]
    df["original_perm"] = [tuple(row) for row in tsne["all_function_lists"]]
    df["dataset"] = dataset_labels
    df["input_string"] = input_strings
    df["predicted_output_string"] = [row for row in predicted_output_strings]
    df["actual_output_string"] = [row for row in actual_output_strings]
    # print number of unique input strings
    print("Number of unique input strings:")
    print(len(set(input_strings)))
   
    return df


def get_interested_test_keys_based_on_accuracy(
    test_combination_accs, accuracy_threshold=1.0, filter_type="less_than"
):
    interested_test_keys = []
    for key, value in test_combination_accs.items():
        if filter_type == "less_than" and value < accuracy_threshold:
            interested_test_keys.append(key)
        elif filter_type == "more_than" and value > accuracy_threshold:
            interested_test_keys.append(key)
    return interested_test_keys


def get_equivalent_train_keys_based_on_test_keys(
    interested_test_keys, equivalence_class_map, df
):
    equivalent_train_keys = set()
    equivalent_train_key_value_map = {}
    for interested_test_key in interested_test_keys:
        if interested_test_key in equivalence_class_map.keys():
            for key in equivalence_class_map[interested_test_key].keys():
                equivalent_train_keys.add(key)
                equivalent_train_key_value_map[interested_test_key] = (
                    equivalence_class_map[interested_test_key]
                )
    all_train_keys = list(df[df["dataset"] == "train"]["original_perm"].unique())
    remaining_train_keys = [
        key for key in all_train_keys if key not in equivalent_train_keys
    ]

    return equivalent_train_keys, equivalent_train_key_value_map, remaining_train_keys





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
        default="combination_3",
        help="Model training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="combination_3",
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
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--function_type", type=str, default="diverse", help="uniform or diverse"
    )
    parser.add_argument(
        "--max_task_length", type=int, default=3, help="max task length"
    )
    parser.add_argument(
        "--replace_strings_repr",
        type=bool,
        default=False,
        help="replace strings in representation",
    )
    parser.add_argument(
        "--equivalence_class_analysis_type",
        type=str,
        default="data",
        help="compute equivalence class analysis",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="simple",
        help="plot type",
    )

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = read_config(f"{ROOT_DIR}/config/eval/conf.yaml")
    cfg.tag = args.prompt_mode
    cfg.prompt_mode = args.prompt_mode
    cfg.prompt_length = args.prompt_length
    cfg.model_split = args.model_split
    cfg.eval_split = args.eval_split
    cfg.nheads_nlayers = args.nheads_nlayers
    cfg.seq_len = args.seq_len
    cfg.n_functions = args.n_functions
    cfg.n_alphabets = args.n_alphabets
    cfg.data_n_alphabets_seq_len_fn_len_max_task_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            args.n_alphabets, args.seq_len, args.n_functions, args.max_task_length
        )
    )
    cfg.data_path = f"{ROOT_DIR}/data/{args.function_type}/{args.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{args.prompt_mode}/{args.eval_split}"
    cfg.train_data_path = f"{ROOT_DIR}/data/{args.function_type}/{args.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{args.prompt_mode}/{args.model_split}"
    cfg.pos_embedding_type = args.pos_embedding_type
    cfg.num_runs = args.num_runs
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    cfg.replace_strings_repr = args.replace_strings_repr
    cfg.max_task_length = args.max_task_length

    acc_path = f"{ROOT_DIR}/models/eval/{cfg.function_type}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{cfg.prompt_mode}/{cfg.prompt_length}/model_{cfg.model_split}/eval_{cfg.eval_split}/{cfg.pos_embedding_type}/{cfg.nheads_nlayers}/seed_{cfg.seed}/accs.pkl"
    functions_info_path = f"{ROOT_DIR}/data/{cfg.function_type}/{cfg.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{cfg.prompt_mode}/{cfg.eval_split}/functions_info.pkl"
    train_accs, test_accs, train_combinations, test_combinations = load_train_test_accs(
        acc_path, functions_info_path
    )
    print(train_accs)
    print(test_accs)
    
    for key, value in test_combinations.items():
        print(key, value)
    
    path_prefix = f"{ROOT_DIR}/models/eval/{cfg.function_type}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{cfg.prompt_mode}/{cfg.prompt_length}/model_{cfg.model_split}/eval_{cfg.eval_split}/{cfg.pos_embedding_type}/{cfg.nheads_nlayers}/seed_0/"
    # check if the path exists
    if not os.path.exists(path_prefix):
        raise FileNotFoundError(f"Path {path_prefix} does not exist")

    if args.equivalence_class_analysis_type == "data":
        equivalence_label = "Data"
        equivalence_class_map = load_data_equivalence_class_map(path_prefix)
    elif args.equivalence_class_analysis_type == "model":
        equivalence_label = "Model"
        equivalence_class_map = load_model_equivalence_class_map(path_prefix)
    else:
        raise ValueError(f"Invalid equivalence class analysis type: {args.equivalence_class_analysis_type}")

    
    representation_results = load_representation_results(
        path_prefix, cfg.replace_strings_repr
    )

    repr_df = get_representation_df(representation_results, cfg.replace_strings_repr)

    less_than_interested_test_keys = get_interested_test_keys_based_on_accuracy(
        test_combinations, accuracy_threshold=0.3, filter_type="less_than"
    )
    more_than_interested_test_keys = get_interested_test_keys_based_on_accuracy(
        test_combinations, accuracy_threshold=0.9, filter_type="more_than"
    )
    interested_test_keys = less_than_interested_test_keys + more_than_interested_test_keys
    equivalent_train_keys, equivalent_train_key_value_map, remaining_train_keys = (
        get_equivalent_train_keys_based_on_test_keys(
            interested_test_keys, equivalence_class_map, repr_df
        )
    )

    selected_df = repr_df[repr_df["original_perm"].isin(interested_test_keys)]
    label = f"{cfg.model_split}_{cfg.eval_split}_{cfg.function_type}_{cfg.replace_strings_repr}"
    if args.plot_type == "simple":
        plot_tsne_with_equivalence_class_score(
            repr_df,
            interested_test_keys,
            equivalent_train_keys,
            remaining_train_keys,
            equivalence_class_map,
            f"{equivalence_label} Equivalence Class",
            test_combinations,
            label + f"_{args.equivalence_class_analysis_type}")
    elif args.plot_type == "slider":
        plot_tsne_with_equivalence_class_score_with_slider(
            repr_df,
            interested_test_keys,
            equivalent_train_keys,
            remaining_train_keys,
            equivalence_class_map,
            f"{equivalence_label} Equivalence Class",
            test_combinations,
            label + f"_{args.equivalence_class_analysis_type}")