"""
Evaluation script
"""

import argparse
import glob
import os

import numpy as np
import torch

from src.data_generation.init import read_config, set_seed
from src.evaluation.fixed_evaluator import FixedPromptEvaluator
from src.evaluation.variable_evaluator import VariablePromptEvaluator
from src.models.nanogpt import nanoGPT

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"
MODELS_DIR = "FILL/IN/PATH/TO/MODELS/DIRECTORY"

# see if cuda is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    DEVICE = torch.device("cpu")


def load_net(fname):
    ckpt = torch.load(fname, weights_only=False, map_location=DEVICE)
    net_cfg = ckpt["config"]

    net = nanoGPT(net_cfg.net)

    net.load_state_dict(ckpt["net"])
    return net, net_cfg


def fetch_dirs(cfg, i):

    ckpt_dir = "{}/models/ckpts/{}/{}/{}/{}/{}/{}/{}/seed_{}".format(
        ROOT_DIR,
        cfg.function_type,
        cfg.prompt_length,
        cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
        cfg.tag,
        cfg.model_split,
        cfg.pos_embedding_type,
        cfg.nheads_nlayers,
        cfg.seed,
    )
    if not os.path.exists(ckpt_dir):
        ckpt_dir = "{}/models/ckpts/{}/{}/{}/{}/{}/{}/{}/seed_{}".format(
            MODELS_DIR,
            cfg.function_type,
            cfg.prompt_length,
            cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
            cfg.tag,
            cfg.model_split,
            cfg.pos_embedding_type,
            cfg.nheads_nlayers,
            cfg.seed,
        )
        if not os.path.exists(ckpt_dir):
            raise ValueError("Checkpoint directory does not exist: {}".format(ckpt_dir))

    def itr(ck):
        return int((ck.split("_")[-1]).split(".")[0])

    all_dirs = os.listdir(ckpt_dir)
    all_dirs = [os.path.join(ckpt_dir, d) for d in all_dirs if d.endswith(".pt")]
    all_dirs = [(itr(d), d) for d in all_dirs]
    all_dirs = sorted(all_dirs)

    reduced_alldirs = []
    for it, cdir in all_dirs:
        reduced_alldirs.append((it, cdir))

    return reduced_alldirs


def main(cfg):
    print("Running evaluation with the following configuration:")
    print(cfg)
    set_seed(cfg.seed)
    for i in range(cfg.num_runs):
        sorted_dirs = fetch_dirs(cfg, i)

        _, net_cfg = load_net(sorted_dirs[0][1])
        log_path = "{}/logs/{}".format(ROOT_DIR, cfg.function_type)
        if cfg.prompt_length == "fixed":
            evaluator = FixedPromptEvaluator(cfg, net_cfg, cfg.nbatch, log_path)
        else:
            evaluator = VariablePromptEvaluator(cfg, net_cfg, cfg.nbatch, log_path)

        sorted_dirs = [sorted_dirs[-1]]
        print("Sorted dirs: ", sorted_dirs)
        accs = []
        for i, ck in enumerate(sorted_dirs):
            net, _ = load_net(ck[1])
            if (i == 0) or (i == len(sorted_dirs) - 1):
                verbose = True
            else:
                verbose = False
            if cfg.acc_analysis:
                metrics = evaluator.get_acc(ck, net, verbose=verbose)
                accs.append((ck, metrics))
                for k, v in metrics.items():
                    result_dict = v
                    acc_vals = np.array(result_dict["total"]["acc"])
                    ood_vals = np.array(result_dict["total"]["ood"])
                    try:
                        print(
                            "Iter: ",
                            ck[0],
                            "Split: ",
                            k,
                            "Acc: ",
                            np.mean(acc_vals),
                            "OOD: ",
                            np.mean(ood_vals),
                        )
                    except:
                        print("Iter: ", ck[0], "Split: ", k, "Acc: ", np.mean(acc_vals))

            model_path = "{}/models".format(ROOT_DIR)
            evaluator.save_accs(cfg, accs, model_path, i)
        if cfg.equivalence_class_analysis:
            if (
                cfg.equivalence_class_type == "both"
                or cfg.equivalence_class_type == "data"
            ):
                print("Computing data-based equivalence classes")
                data_equivalence_class_map = (
                    evaluator.compute_data_based_equivalence_classes(net, ck)
                )
                evaluator.save_equivalence_classes(
                    cfg,
                    data_equivalence_class_map,
                    model_path,
                    i,
                    equivalence_class_type="data",
                )
            if (
                cfg.equivalence_class_type == "both"
                or cfg.equivalence_class_type == "model"
            ):
                print("Computing model-based equivalence classes")
                model_equivalence_class_map = (
                    evaluator.compute_model_based_equivalence_classes(net, ck)
                )
                evaluator.save_equivalence_classes(
                    cfg,
                    model_equivalence_class_map,
                    model_path,
                    i,
                    equivalence_class_type="model",
                )

        if cfg.repr_analysis:
            print(
                "Computing representation analysis with replace strings in representation: {}".format(
                    cfg.replace_strings_repr
                )
            )
            results = evaluator.analyze_representations(net, ck)
            evaluator.save_representation_results(cfg, results, model_path, i)


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
        default="combination_6",
        help="Model training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="combination_6",
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
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--max_task_length", type=int, default=6, help="max task length"
    )
    parser.add_argument(
        "--replace_strings_repr",
        type=bool,
        default=False,
        help="replace strings in representation analysis",
    )
    parser.add_argument(
        "--repr_analysis", type=bool, default=False, help="do representation analysis"
    )
    parser.add_argument(
        "--equivalence_class_analysis",
        type=bool,
        default=False,
        help="do equivalence class analysis",
    )
    parser.add_argument(
        "--equivalence_class_type",
        type=str,
        default="both",
        help="model or data or both",
    )
    parser.add_argument(
        "--acc_analysis", type=bool, default=True, help="do accuracy analysis"
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
    cfg.replace_strings_repr = args.replace_strings_repr
    cfg.repr_analysis = args.repr_analysis
    cfg.equivalence_class_analysis = args.equivalence_class_analysis
    cfg.equivalence_class_type = args.equivalence_class_type
    cfg.acc_analysis = args.acc_analysis
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
    main(cfg)
