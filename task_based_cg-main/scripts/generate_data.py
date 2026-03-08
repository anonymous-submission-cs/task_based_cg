import argparse

from src.data_generation.functions import CreateFunctions
from src.data_generation.generator import SyntheticData
from src.data_generation.generator_uniform import SyntheticDataUniform
from src.data_generation.init import read_config, set_seed

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"


def main(cfg):
    set_seed(cfg.seed)
    # Create function object
    function_obj = CreateFunctions(cfg)
    # Create functions
    train_functions, test_functions, functions_info = function_obj.get_train_functions()

    # Create function dictionary
    function_dict = function_obj.function_dict
    # Create synthetic data object
    if cfg.function.type == "uniform":
        synthetic_data = SyntheticDataUniform(
            cfg,
            train_functions,
            test_functions,
            function_dict,
            functions_info,
            function_obj.mappings,
        )
    else:
        synthetic_data = SyntheticData(
            cfg, train_functions, test_functions, function_dict, functions_info
        )
    # Initialize tokens
    synthetic_data.init_tokens()
    print("generate_corpus")
    synthetic_data.generate_corpus()
    # Store the data
    synthetic_data.store_data()


if __name__ == "__main__":
    # Set config in the yaml files

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_length",
        type=str,
        default="fixed",
        help="fixed or variable",
        required=True,
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="combination_6",
        help="Random split of permutations of size K: combination_K; \
        Systematic split of permutations of size K with one relative order in training: permutation_K; \
        Systematic split of size K with T relative orders in training: permutation_T_K",
        required=True,
    )
    parser.add_argument(
        "--n_functions",
        type=int,
        default=6,
        help="number of non-identity functions",
        required=True,
    )
    parser.add_argument(
        "--task_max_length",
        type=int,
        default=7,
        help="compositional task max length",
        required=True,
    )
    parser.add_argument(
        "--n_alphabets", type=int, default=26, help="number of alphabets"
    )
    parser.add_argument(
        "--seq_len", type=int, default=6, help="sequence length for input data string"
    )
    parser.add_argument(
        "--functions_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    # Read config file
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    # read the config file
    cfg = read_config(cfg_path)
    cfg.prompt_length = args.prompt_length
    cfg.function.split.strategy = args.split_strategy
    cfg.n_alphabets = args.n_alphabets
    cfg.seq_len = args.seq_len
    cfg.function.n_functions = args.n_functions
    cfg.task_max_length = args.task_max_length
    cfg.function.type = args.functions_type
    cfg.seed = args.seed
    main(cfg)
