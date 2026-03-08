import importlib.util
import json
import logging
import os
import pickle
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data_generation.functions import (
    DIVERSE_FUNCTIONS,
    DIVERSE_2_FUNCTIONS,
    BaseFunction,
    MapRandom,
    apply_function_composition,
    apply_function_composition_uniform,
)
from src.data_generation.generator import SyntheticData
from src.data_generation.utils import *
from src.evaluation.representation import (
    RepresentationExtractor,
    perform_tsne_and_visualize,
    save_representation_results,
)


class FixedPromptEvaluator(SyntheticData):
    def __init__(self, eval_cfg, net_cfg, nbatch, log_path):
        self._setup_logging(eval_cfg, log_path)
        self._load_data(eval_cfg)
        self._setup_tokens()
        self.eval_cfg = eval_cfg
        self.net_cfg = net_cfg
        self.nbatch = nbatch

    def _setup_logging(self, eval_cfg, log_path):
        """Initialize logging configuration"""
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

        log_file_dir = "{}/{}/{}/{}/model_{}/eval_{}/{}/seed_{}".format(
            log_path,
            eval_cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
            eval_cfg.tag,
            eval_cfg.prompt_length,
            eval_cfg.model_split,
            eval_cfg.eval_split,
            eval_cfg.pos_embedding_type,
            eval_cfg.seed,
        )
        os.makedirs(log_file_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create file handler
        log_file_path = "{}/eval.log".format(log_file_dir)
        print(f"Logging to {log_file_path}")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        # Prevent propagation to root logger (optional, but recommended)
        self.logger.propagate = False

        self.logger.info("Initializing SyntheticEval...")
        self.logger.info(log_file_path)

    def _load_data(self, eval_cfg):
        """Load data files and configuration"""
        self.data_path = eval_cfg.data_path
        data_fname = os.path.join(self.data_path, "config.json")

        self.token_idx = np.load(
            os.path.join(self.data_path, "token_idx.pkl"), allow_pickle=True
        )
        self.token = np.load(
            os.path.join(self.data_path, "token.pkl"), allow_pickle=True
        )
        self.functions_info = np.load(
            os.path.join(self.data_path, "functions_info.pkl"), allow_pickle=True
        )
        self.cfg = OmegaConf.create(json.load(open(data_fname)))
        self.logger.info(self.cfg)

    def _setup_tokens(self):
        """Setup special tokens"""
        self.start_token = "<START>"
        self.space_token = " "
        self.sep_token = "<SEP>"
        self.null_token = "<NULL>"
        self.end_token = "<END>"
        self.special_tokens = [
            self.start_token,
            self.space_token,
            self.sep_token,
            self.null_token,
            self.end_token,
        ]
        self.n_special = len(self.special_tokens)
        self.n_alphabets = self.cfg.n_alphabets

    def _log_predictions(self, print_indices, ck, split, dat, output, labels=None):
        """Log input/prediction pairs for debugging with colored error highlighting"""
        input_data = dat.detach().cpu().numpy()
        predictions = output.detach().cpu().numpy()

        # ANSI color codes for highlighting
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        END = "\033[0m"

        for idx in print_indices:
            self.logger.info("=======================================================")
            self.logger.info("{} {}".format(ck, split))

            # Decode input and prediction
            input_decoded = self.decode(input_data[idx])
            pred_decoded = self.decode(predictions[idx])

            # Highlight differences between input and prediction
            highlighted_pred = self._highlight_differences(
                input_decoded, pred_decoded, RED, END
            )

            self.logger.info("Input: {}".format(input_decoded))
            if labels is not None:
                label_decoded = self.decode(labels[idx])
                self.logger.info("Label: {}".format(label_decoded))
            # print normal prediction first
            self.logger.info("Predi: {}".format(pred_decoded))
            self.logger.info(
                "=======================Colored diffs================================\n"
            )
            self.logger.info("Input: {}".format(input_decoded))
            self.logger.info("Predi: {}".format(highlighted_pred))
            self.logger.info(
                "=======================================================\n\n"
            )

    def _highlight_differences(self, input_str, pred_str, error_color, end_color):
        """Highlight differences between input and prediction strings"""
        highlighted = ""
        min_len = min(len(input_str), len(pred_str))

        for i in range(min_len):
            if input_str[i] != pred_str[i]:
                highlighted += error_color + pred_str[i] + end_color
            else:
                highlighted += pred_str[i]

        # Add remaining characters from prediction if it's longer
        if len(pred_str) > min_len:
            highlighted += error_color + pred_str[min_len:] + end_color

        return highlighted

    def _generate_predictions(self, net, inp_batch, new_length):
        """Generate predictions using the model"""
        for _ in range(new_length):
            logits = net(inp_batch)
            logits = logits[:, -1, :]
            inp_next = torch.argmax(logits, -1, keepdims=True)
            inp_batch = torch.cat((inp_batch, inp_next), dim=1)
        return inp_batch

    def _get_uniform_eval_docs(self, docs, per_sample_count=None):
        """Get uniform eval docs"""

        # get equal samples per combination id such that total number of samples is 10k
        combination_ids, doc_combination_id_map = map_docs_to_combination_id(
            self.token, self.token_idx, self.sep_token, self.functions_info, docs
        )
        unique_combination_ids = np.unique(combination_ids)
        if per_sample_count is None:
            per_sample_count = int(
                self.eval_cfg.nsamples // len(unique_combination_ids)
            )
            if per_sample_count <= 1:
                per_sample_count = 2
        sampled_eval_docs = []
        for cid in unique_combination_ids:
            cid_docs = docs[combination_ids == cid]
            cid_docs = cid_docs[
                np.random.choice(len(cid_docs), size=per_sample_count, replace=False)
            ]
            sampled_eval_docs.append(cid_docs)
        sampled_eval_docs = np.concatenate(sampled_eval_docs)

        return sampled_eval_docs

    def _load_corpus_data(
        self, sampled_train_docs=True, per_sample_count=None, cross_eval_flag=False
    ):
        """Load training and test corpus data"""
        if cross_eval_flag:
            train_docs = np.load(
                os.path.join(
                    self.eval_cfg.train_data_path,
                    f"train_{self.net_cfg.tag}_corpus.npy",
                )
            )
            train_heldout_docs = np.load(
                os.path.join(
                    self.eval_cfg.train_data_path,
                    f"train_heldout_{self.net_cfg.tag}_corpus.npy",
                )
            )
        else:
            train_docs = np.load(
                os.path.join(self.data_path, f"train_{self.net_cfg.tag}_corpus.npy")
            )
            train_heldout_docs = np.load(
                os.path.join(
                    self.data_path, f"train_heldout_{self.net_cfg.tag}_corpus.npy"
                )
            )
        test_docs = np.load(
            os.path.join(self.data_path, f"test_{self.net_cfg.tag}_corpus.npy")
        )
        if sampled_train_docs:
            sampled_train_docs = self._get_uniform_eval_docs(
                train_docs, per_sample_count
            )
        else:
            sampled_train_docs = train_docs
        self.logger.info(
            "Shape of sampled train docs: {}".format(sampled_train_docs.shape)
        )
        self.logger.info("Shape of test docs: {}".format(test_docs.shape))
        return {
            "train": sampled_train_docs,
            "train_heldout": train_heldout_docs,
            "test": test_docs,
        }

    def _log_verbose_results(self, split, docs, doc_combination_id_map, metrics, ck):
        """Log verbose results for standard evaluation"""
        self.logger.info("Evaluating {} documents: {}".format(split, len(docs)))
        combination_acc = metrics["combination"]["acc"]
        combination_ood = metrics["combination"]["ood"]
        total_acc = metrics["total"]["acc"]
        total_ood = metrics["total"]["ood"]
        if combination_acc is not None:
            for cid, acc in combination_acc.items():
                self.logger.info(
                    "Accuracy for combination id {} {} is {:.3f}".format(
                        cid, doc_combination_id_map[cid], acc
                    )
                )

        if total_ood is not None:
            self.logger.info(
                "Iter: {} Split: {} Acc: {:.3f} OOD: {:.3f}".format(
                    ck, split, total_acc, total_ood
                )
            )
        else:
            self.logger.info(
                "Iter: {} Split: {} Acc: {:.3f}".format(ck, split, total_acc)
            )

    def _error_analysis_sbs(self, dat, output_l, targets_l):
        """Optimized step-by-step error analysis using vectorized operations"""
        output_l = output_l.cpu().numpy()
        targets_l = targets_l.cpu().numpy()
        dat_np = dat.cpu().numpy()

        batch_size = dat.shape[0]
        pad_length = 2 * self.eval_cfg.seq_len

        # Pre-allocate structures
        module_wise_acc = defaultdict(lambda: {"acc": [], "total": 0})
        step_by_step_acc = {
            "individual": defaultdict(list),
            "cumulative": defaultdict(list),
        }

        # Process all samples in batch
        all_function_lists = []
        all_decoded_lists = []
        max_functions = 0

        # First pass: extract all function lists and find max length
        for i in range(batch_size):
            function_list = get_function_list(dat_np[i], self.token_idx[self.sep_token])
            decoded_function_list = self.decode(function_list, return_list=True)
            all_function_lists.append(function_list)
            all_decoded_lists.append(decoded_function_list)
            max_functions = max(max_functions, len(decoded_function_list))

        # Create batch accuracy matrix: [batch_size, max_functions]
        batch_accuracies = np.full((batch_size, max_functions), False, dtype=bool)

        # Vectorized slice processing
        for j in range(max_functions):
            # Determine slice positions for this function step
            if j == 0:
                start_pos = 0
                end_pos = pad_length
            else:
                start_pos = j * (pad_length + 1)  # +1 for SEP token
                end_pos = start_pos + pad_length

            # Extract batch slices for this function step
            valid_samples = []
            for i in range(batch_size):
                if j < len(all_decoded_lists[i]):
                    valid_samples.append(i)

            if not valid_samples:
                continue

            # Vectorized accuracy calculation for valid samples
            valid_samples = np.array(valid_samples)
            batch_slice_output = output_l[valid_samples, start_pos:end_pos]
            batch_slice_targets = targets_l[valid_samples, start_pos:end_pos]

            # Check if all tokens in each sequence match
            slice_accuracies = (batch_slice_output == batch_slice_targets).all(axis=1)

            # Update batch accuracy matrix
            batch_accuracies[valid_samples, j] = slice_accuracies

            # Update step-by-step individual accuracies
            step_by_step_acc["individual"][j].extend(slice_accuracies.tolist())

            # Update module-wise accuracies
            for idx, sample_idx in enumerate(valid_samples):
                fn = all_decoded_lists[sample_idx][j]
                module_wise_acc[fn]["acc"].append(slice_accuracies[idx])
                module_wise_acc[fn]["total"] += 1

        # Vectorized cumulative accuracy calculation
        for j in range(max_functions):
            if j == 0:
                # First step: same as individual
                step_by_step_acc["cumulative"][j] = step_by_step_acc["individual"][
                    j
                ].copy()
            else:
                # Cumulative: AND with all previous steps
                cumulative_acc = []
                for i in range(batch_size):
                    if j < len(all_decoded_lists[i]):
                        # Take AND of all steps up to j
                        cum_acc = batch_accuracies[i, : j + 1].all()
                        cumulative_acc.append(cum_acc)
                step_by_step_acc["cumulative"][j] = cumulative_acc

        return dict(module_wise_acc), step_by_step_acc

    def _error_metrics_sbs(self, dat, output, seq_info, combination_ids):
        """Optimized step-by-step error metrics calculation"""
        # Vectorized slicing
        output_l = output[:, seq_info["prompt_pos_end"] :]
        targets_l = dat[:, seq_info["prompt_pos_end"] :]

        # Vectorized accuracy calculation
        acc_l = output_l == targets_l
        sharp_acc = acc_l.all(dim=-1).float().mean().cpu().numpy()

        # Batch OOD detection
        ood_flags = is_ood_prompt(
            self.token, self.token_idx, output_l, targets_l, self.eval_cfg.prompt_length
        )
        ood_flags = ood_flags.cpu().tolist()
        ood_mean = np.array(ood_flags).mean()

        # Vectorized combination accuracy
        (
            sharp_combination_acc,
            ood_combination_acc,
            total_unique_combination_ids,
            print_error_indices,
        ) = calculate_combination_accuracy(
            acc_l, ood_flags, combination_ids, use_sharp=True
        )

        # Optimized module-wise and step-by-step analysis
        module_wise_acc, step_by_step_acc = self._error_analysis_sbs(
            dat, output_l, targets_l
        )

        # Calculate direct metrics once and reuse
        direct_acc_metrics = self._error_metrics_direct(
            dat, output, seq_info, combination_ids
        )

        return {
            "total": {"acc": sharp_acc, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
            "module_wise": module_wise_acc,
            "step_by_step": step_by_step_acc,
            "direct": direct_acc_metrics,
            "total_unique_combination_ids": total_unique_combination_ids,
            "print_error_indices": print_error_indices,
        }

    def _error_metrics_direct(self, dat, output, seq_info, combination_ids):
        """Optimized direct error metrics calculation"""
        # Vectorized slicing
        start_idx = seq_info["last_sep_pos"] + 1
        end_idx = seq_info["end_pos"]
        output_l = output[:, start_idx:end_idx]
        targets_l = dat[:, start_idx:end_idx]

        # Vectorized accuracy calculation
        acc_l = output_l == targets_l
        sharp_acc = acc_l.all(dim=-1).float().mean().cpu().numpy()

        # Batch OOD detection
        ood_flags = is_ood_prompt(
            self.token, self.token_idx, output_l, targets_l, self.eval_cfg.prompt_length
        )
        ood_flags = ood_flags.cpu().tolist()
        ood_mean = np.array(ood_flags).mean()

        # Vectorized combination accuracy
        (
            sharp_combination_acc,
            ood_combination_acc,
            total_unique_combination_ids,
            print_error_indices,
        ) = calculate_combination_accuracy(
            acc_l, ood_flags, combination_ids, use_sharp=True
        )

        return {
            "total": {"acc": sharp_acc, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
            "total_unique_combination_ids": total_unique_combination_ids,
            "print_error_indices": print_error_indices,
        }

    def _calculate_metrics(self, dat, output, seq_info, combination_ids, prompt_mode):
        """Optimized metrics calculation for fixed prompt mode"""
        if prompt_mode == "step_by_step":
            return self._error_metrics_sbs(dat, output, seq_info, combination_ids)
        elif prompt_mode in ["direct", "curriculum"]:
            return self._error_metrics_direct(dat, output, seq_info, combination_ids)
        else:
            raise ValueError(f"Invalid prompt mode: {prompt_mode}")

    def generate_model_outputs(self, split, net, docs, seq_info, device):
        dat = np.array(docs)
        dat = torch.Tensor(dat.astype(int)).long()

        batch_size = self.eval_cfg.nbatch
        shape = dat.shape
        if device == "cuda":
            dat = dat.cuda(non_blocking=True)
        dat = dat.view(-1, shape[-1])
        inp = dat[:, : seq_info["prompt_pos_end"]]
        outputs = []
        for batch_start in range(0, dat.size(0), batch_size):
            batch_end = min(batch_start + batch_size, dat.size(0))
            inp_batch = inp[batch_start:batch_end]
            inp_batch = self._generate_predictions(net, inp_batch, seq_info["new_len"])
            outputs.append(inp_batch)
        output = torch.cat(outputs, dim=0)
        return output

    def evaluate_docs(self, ck, split, net, dat, seq_info, combination_ids, device):
        """Evaluate documents with fixed prompt length"""
        self.logger.info("Evaluating {} documents: {}".format(split, len(dat)))
        self.logger.info("Seq info Type: {}".format(seq_info))

        batch_size = self.eval_cfg.nbatch
        shape = dat.shape

        if device == "cuda":
            dat = dat.cuda(non_blocking=True)
        dat = dat.view(-1, shape[-1])
        inp = dat[:, : seq_info["prompt_pos_end"]]
        # Process in batches
        outputs = []
        for batch_start in range(0, dat.size(0), batch_size):
            batch_end = min(batch_start + batch_size, dat.size(0))
            inp_batch = inp[batch_start:batch_end]
            # Generate predictions
            inp_batch = self._generate_predictions(net, inp_batch, seq_info["new_len"])
            outputs.append(inp_batch)
        output = torch.cat(outputs, dim=0)
        metrics = self._calculate_metrics(
            dat, output, seq_info, combination_ids, self.eval_cfg.prompt_mode
        )
        print_indices = metrics["print_error_indices"]

        self._log_predictions(print_indices, ck, split, dat, output)
        return metrics

    def get_acc(self, ck, net, verbose=False):
        """Get accuracy for standard evaluation"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval()
        if device == "cuda":
            net = net.cuda()

        docs_dict = self._load_corpus_data()
        acc_map = {}

        for split, docs in docs_dict.items():
            sampled_docs = np.array(docs)
            flatten_docs = torch.Tensor(sampled_docs.astype(int)).long()

            combination_ids, doc_combination_id_map = map_docs_to_combination_id(
                self.token,
                self.token_idx,
                self.sep_token,
                self.functions_info,
                sampled_docs,
            )

            sample = torch.Tensor(docs[0]).long()
            seq_info = get_seq_info(
                self.token_idx,
                self.sep_token,
                self.end_token,
                sample,
                self.eval_cfg.function_type,
            )
            metrics = self.evaluate_docs(
                ck, split, net, flatten_docs, seq_info, combination_ids, device
            )

            acc_map[split] = metrics

            if verbose:
                self._log_verbose_results(
                    split, docs, doc_combination_id_map, metrics, ck
                )

        return acc_map

    def save_accs(self, cfg, accs, model_path="models", run_num=None):
        """Save accuracy results"""
        model_split = self.eval_cfg.model_split
        eval_split = self.eval_cfg.eval_split
        nheads_nlayers = f"nh{self.net_cfg.net.n_head}_nl{self.net_cfg.net.n_layer}"

        results_path = "{}/eval/{}/{}/{}/{}/model_{}/eval_{}/{}/{}/seed_{}".format(
            model_path,
            cfg.function_type,
            self.eval_cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
            cfg.tag,
            cfg.prompt_length,
            model_split,
            eval_split,
            cfg.pos_embedding_type,
            nheads_nlayers,
            cfg.seed,
        )

        os.makedirs(results_path, exist_ok=True)
        pickle.dump(accs, open(results_path + "/accs.pkl", "wb"))

    def save_equivalence_classes(
        self,
        cfg,
        equivalence_class_map,
        model_path="models",
        run_num=None,
        equivalence_class_type="model",
    ):
        """Save equivalence classes"""
        model_split = self.eval_cfg.model_split
        eval_split = self.eval_cfg.eval_split
        nheads_nlayers = f"nh{self.net_cfg.net.n_head}_nl{self.net_cfg.net.n_layer}"

        results_path = "{}/eval/{}/{}/{}/{}/model_{}/eval_{}/{}/{}/seed_{}".format(
            model_path,
            cfg.function_type,
            self.eval_cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
            cfg.tag,
            cfg.prompt_length,
            model_split,
            eval_split,
            cfg.pos_embedding_type,
            nheads_nlayers,
            cfg.seed,
        )

        os.makedirs(results_path, exist_ok=True)
        pickle.dump(
            equivalence_class_map,
            open(
                results_path + f"/{equivalence_class_type}_equivalence_class_map.pkl",
                "wb",
            ),
        )
        print(
            f"Equivalence class map saved to {results_path}/{equivalence_class_type}_equivalence_class_map.pkl"
        )

    def _replace_string(self, doc, input_str1, input_str2):
        """Replace string in document"""
        sep_positions = np.where(doc == self.token_idx[self.sep_token])[0]
        doc[sep_positions[0] + 1 : sep_positions[1]] = input_str1
        if len(input_str2) > 0:
            doc[sep_positions[1] + 1 : sep_positions[2]] = input_str2
        return doc

    def compute_model_based_equivalence_classes(self, net, ck):
        self.logger.info("Computing model equivalence classes...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval()
        if device == "cuda":
            net = net.cuda()

        docs_dict = self._load_corpus_data(
            sampled_train_docs=True, per_sample_count=1, cross_eval_flag=True
        )
        test_docs = np.array(docs_dict["test"])
        train_docs = np.array(docs_dict["train"])

        sample = torch.tensor(test_docs[0]).long()
        seq_info = get_seq_info(
            self.token_idx,
            self.sep_token,
            self.end_token,
            sample,
            self.eval_cfg.function_type,
        )

        # Precompute decoded function lists
        test_function_list = self.get_function_list_decoded(
            test_docs, self.token_idx[self.sep_token], self.token
        )
        train_function_list = self.get_function_list_decoded(
            train_docs, self.token_idx[self.sep_token], self.token
        )

        # Precompute outputs once
        test_outputs = self.generate_model_outputs(
            "test", net, test_docs, seq_info, device
        )
        test_outputs = test_outputs[
            :, -seq_info["new_len"] :
        ]  # shape [n_test, new_len]

        equivalence_class_map = {}

        # Extract test input strings only once
        test_inputs = [
            self._extract_inputs_from_doc(
                doc, self.token_idx[self.sep_token], tokens_to_string=False
            )
            for doc in test_docs
        ]
        # Pre-compute all train outputs for all test inputs
        replace_train_docs = []
        for test_idx, (test_fn_list, (input_str1, input_str2)) in enumerate(
            zip(test_function_list, test_inputs)
        ):
            test_fn_tuple = tuple(test_fn_list)
            if test_fn_tuple not in equivalence_class_map:
                equivalence_class_map[test_fn_tuple] = {}

            # Replace inputs in *all* train docs at once
            replaced_train_samples = [
                self._replace_string(train_doc.copy(), input_str1, input_str2)
                for train_doc in train_docs
            ]
            replace_train_docs.append(replaced_train_samples)

        # Generate all train outputs at once
        train_outputs = self.generate_model_outputs(
            "train", net, replace_train_docs, seq_info, device
        )
        train_outputs = train_outputs[
            :, -seq_info["new_len"] :
        ]  # shape [n_test_inputs * n_train_functions, new_len]
        # reshape train_outputs to [n_test_inputs, n_train_functions, new_len]
        train_outputs = train_outputs.reshape(
            len(test_inputs), len(train_function_list), seq_info["new_len"]
        )

        # Vectorized equivalence class computation for all test inputs
        for test_idx, test_fn_list in enumerate(test_function_list):
            test_fn_tuple = tuple(test_fn_list)
            test_output = test_outputs[test_idx]  # shape [new_len]

            # Vectorized match across all train functions for this test input
            matches = (train_outputs[test_idx] == test_output).all(
                axis=1
            )  # shape [n_train_functions]
            # convert from tensor to numpy array
            matches = matches.cpu().numpy()

            # Get indices of matching train functions
            matching_train_indices = np.where(matches)[0]

            # Vectorized update of equivalence class map
            if len(matching_train_indices) > 0:
                # Convert train function lists to tuples for matching indices
                matching_train_tuples = [
                    tuple(train_function_list[idx]) for idx in matching_train_indices
                ]

                # Count occurrences using Counter for efficiency
                tuple_counts = Counter(matching_train_tuples)

                # Update equivalence class map
                for train_fn_tuple, count in tuple_counts.items():
                    equivalence_class_map[test_fn_tuple].setdefault(train_fn_tuple, 0)
                    equivalence_class_map[test_fn_tuple][train_fn_tuple] += count

        return equivalence_class_map

    def generate_actual_outputs(self, function_list, input_str1, input_str2):
        all_outputs = []
        for single_function_list in function_list:
            if self.eval_cfg.function_type == "diverse" or self.eval_cfg.function_type == "diverse2":
                filter_func = lambda x: x in "aeiou"
                offset = 1
                pad_length = 2 * self.eval_cfg.seq_len
                outputs = apply_function_composition(
                    self.eval_cfg.n_alphabets,
                    single_function_list,
                    DIVERSE_FUNCTIONS if self.eval_cfg.function_type == "diverse" else DIVERSE_2_FUNCTIONS,
                    input_str1,
                    input_str2,
                    filter_func,
                    offset,
                )

            elif self.eval_cfg.function_type == "uniform":
                filter_func = lambda x: x in "aeiou"
                offset = 1
                pad_length = self.eval_cfg.seq_len
                function_dict = {
                    f"map{i}": MapRandom.map_random(seed=i)
                    for i in range(1, self.eval_cfg.n_functions + 1)
                }
                function_dict["identity"] = BaseFunction.identity
                outputs = apply_function_composition_uniform(
                    single_function_list,
                    function_dict,
                    input_str1,
                )
               
            else:
                raise ValueError(
                    f"Invalid function type: {self.eval_cfg.function_type}"
                )
            
            for i in range(len(outputs)):
                if len(outputs[i]) < pad_length:
                    outputs[i] = outputs[i] + self.space_token * (
                        pad_length - len(outputs[i])
                    )
                else:
                    outputs[i] = outputs[i][:pad_length]
            output_tokens = self.get_output_tokens(outputs)
            if self.eval_cfg.prompt_mode == "direct":
                output_tokens = output_tokens[-1]
            all_outputs.append(output_tokens)
        return np.array(all_outputs)

    def compute_data_based_equivalence_classes(self, net, ck):
        self.logger.info("Computing model equivalence classes...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval()
        if device == "cuda":
            net = net.cuda()

        docs_dict = self._load_corpus_data(
            sampled_train_docs=True, per_sample_count=1, cross_eval_flag=True
        )
        test_docs = np.array(docs_dict["test"])
        train_docs = np.array(docs_dict["train"])

        sample = torch.tensor(test_docs[0]).long()
        seq_info = get_seq_info(
            self.token_idx,
            self.sep_token,
            self.end_token,
            sample,
            self.eval_cfg.function_type,
        )

        # Precompute decoded function lists
        test_function_list = self.get_function_list_decoded(
            test_docs, self.token_idx[self.sep_token], self.token
        )
        train_function_list = self.get_function_list_decoded(
            train_docs, self.token_idx[self.sep_token], self.token
        )

        # Precompute outputs once
        test_outputs = test_docs[:, -seq_info["new_len"] :]
        # ignore the last token
        test_outputs = test_outputs[:, :-1]

        equivalence_class_map = {}

        # Extract test input strings only once
        test_inputs = [
            self._extract_inputs_from_doc(
                doc, self.token_idx[self.sep_token], tokens_to_string=True
            )
            for doc in test_docs
        ]

        # Pre-compute all train outputs for all test inputs
        # Shape: [n_test_inputs, n_train_functions, output_length]
        all_train_outputs = np.zeros(
            (len(test_inputs), len(train_function_list), test_outputs.shape[1]),
            dtype=test_outputs.dtype,
        )

        for test_idx, (input_str1, input_str2) in enumerate(test_inputs):
            train_outputs = self.generate_actual_outputs(
                train_function_list, input_str1, input_str2
            )
            all_train_outputs[test_idx] = train_outputs

        # Vectorized equivalence class computation
        for test_idx, test_fn_list in enumerate(test_function_list):
            test_fn_tuple = tuple(test_fn_list)
            if test_fn_tuple not in equivalence_class_map:
                equivalence_class_map[test_fn_tuple] = {}

            test_output = test_outputs[test_idx]  # shape [output_length]
            train_outputs = all_train_outputs[
                test_idx
            ]  # shape [n_train_functions, output_length]

            # Vectorized match across all train functions
            matches = (train_outputs == test_output).all(
                axis=1
            )  # shape [n_train_functions]

            # Get indices of matching train functions
            matching_train_indices = np.where(matches)[0]

            # Vectorized update of equivalence class map
            if len(matching_train_indices) > 0:
                # Convert train function lists to tuples for matching indices
                matching_train_tuples = [
                    tuple(train_function_list[idx]) for idx in matching_train_indices
                ]

                # Count occurrences using Counter for efficiency
                tuple_counts = Counter(matching_train_tuples)

                # Update equivalence class map
                for train_fn_tuple, count in tuple_counts.items():
                    equivalence_class_map[test_fn_tuple].setdefault(train_fn_tuple, 0)
                    equivalence_class_map[test_fn_tuple][train_fn_tuple] += count
        print(equivalence_class_map)

        return equivalence_class_map

    def _extract_inputs_from_docs(self, docs):
        """Extract input strings from a test document that uses the target function list"""
        sep_token_idx = self.token_idx[self.sep_token]

        # Find a document with this function list
        string_inputs = []
        for doc in docs:
            string_inputs.append(self._extract_inputs_from_doc(doc, sep_token_idx))
        return string_inputs

    def get_function_list_decoded(self, docs, sep_token, tokens):
        all_function_list = []
        for doc in docs:
            function_list = []
            fn_list = get_function_list(doc, sep_token)
            for fn in fn_list:
                function_list.append(tokens[fn])
            all_function_list.append(function_list)
        return all_function_list

    def _extract_inputs_from_doc(self, doc, sep_token_idx, tokens_to_string=False):
        """Extract input strings from document"""
        sep_positions = np.where(doc == sep_token_idx)[0]

        if len(sep_positions) < 2:
            raise ValueError("Document doesn't have enough SEP tokens")

        # First input: between first and second SEP
        input1_tokens = doc[sep_positions[0] + 1 : sep_positions[1]]
        if tokens_to_string:
            input_str1 = self._tokens_to_string(input1_tokens)
        else:
            input_str1 = input1_tokens

        # Second input: between second and third SEP (if exists)
        input_str2 = ""
        if len(sep_positions) >= 3:
            input2_tokens = doc[sep_positions[1] + 1 : sep_positions[2]]
            if tokens_to_string:
                input_str2 = self._tokens_to_string(input2_tokens)
            else:
                input_str2 = input2_tokens

        return input_str1, input_str2

    def _tokens_to_string(self, tokens):
        """Convert token indices to string"""
        # convert tokens to numpy array
        tokens = np.array(tokens)
        result = ""
        for token_idx in tokens:
            if token_idx in self.token:
                char = self.token[token_idx]
                if len(char) == 1 and char.isalpha():  # Only alphabetic characters
                    result += char
        return result

    def _get_function_output(self, function_list, input_str1, input_str2, net, device):
        """Get model output for a function applied to inputs"""
        # Create document with function and inputs
        doc = self._create_doc_from_function_inputs(
            function_list, input_str1, input_str2
        )

        # Get sequence info
        seq_info = get_seq_info(
            self.token_idx,
            self.sep_token,
            self.end_token,
            doc,
            self.eval_cfg.function_type,
        )

        # Prepare input for model
        inp = doc[: seq_info["prompt_pos_end"]].unsqueeze(0)  # Add batch dimension
        if device == "cuda":
            inp = inp.cuda()

        # Generate prediction
        with torch.no_grad():
            output = self._generate_predictions(net, inp, seq_info["new_len"])

        # Extract relevant output part for comparison
        if self.eval_cfg.prompt_mode == "direct":
            # For direct mode, compare only the final output section
            start_idx = seq_info["last_sep_pos"] + 1
            generated_output = output[0, start_idx:]
        else:
            # For step_by_step mode, compare the entire generated sequence
            generated_output = output[0, seq_info["prompt_pos_end"] :]

        return generated_output.cpu()

    def _create_doc_from_function_inputs(self, function_list, input_str1, input_str2):
        """Create document from function list and input strings"""
        # Convert inputs to token indices
        input1_tokens = np.array([self.token_idx[c] for c in input_str1])
        if input_str2:
            input2_tokens = np.array([self.token_idx[c] for c in input_str2])

        # Get task tokens (function names)
        task_tokens = []
        for fn_name in function_list:
            if fn_name in self.token_idx:
                task_tokens.append(self.token_idx[fn_name])
        task_tokens = np.array(task_tokens)

        # Special tokens
        start_idx = np.array([self.token_idx["<START>"]])
        sep_idx = np.array([self.token_idx["<SEP>"]])

        # Create document based on format
        if self.eval_cfg.function_type in ["sort", "sort_map"] or not input_str2:
            # Single input format: START task SEP input SEP [output]
            document = np.concatenate(
                [
                    start_idx,
                    task_tokens,
                    sep_idx,
                    input1_tokens,
                    sep_idx,
                ]
            )
        else:
            # Two input format: START task SEP input1 SEP input2 SEP [output]
            document = np.concatenate(
                [
                    start_idx,
                    task_tokens,
                    sep_idx,
                    input1_tokens,
                    sep_idx,
                    input2_tokens,
                    sep_idx,
                ]
            )

        return torch.tensor(document, dtype=torch.long)

    def extract_all_representations(self, net, docs, device="cuda", batch_size=100):
        """Extract representations for all documents in the dataset"""
        extractor = RepresentationExtractor(net, device)
        extractor.register_hook()

        all_representations = []
        all_function_lists = []
        input_strings = []
        predicted_output_strings = []
        actual_output_strings = []

        sep_token = self.token_idx[self.sep_token]
        end_token = self.token_idx[self.end_token]

        self.logger.info(f"Processing {len(docs)} documents...")

        num_batches = len(docs) // batch_size + (
            1 if len(docs) % batch_size != 0 else 0
        )
        
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                self.logger.info(f"Processing batch {batch_idx}/{num_batches}")

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(docs))

            batch_docs = docs[start_idx:end_idx]
            batch_seq_infos = [
                get_seq_info(
                    self.token_idx,
                    self.sep_token,
                    self.end_token,
                    doc,
                    self.eval_cfg.function_type,
                )
                for doc in batch_docs
            ]

            # Prepare input batch with random strings or use existing strings
            batch_inputs = []
            

            if self.eval_cfg.replace_strings_repr:
                # Generate random strings
                np.random.seed(2)
                random_string_1, random_string_1_token_idx = self.sample_string()

                if self.eval_cfg.function_type == "diverse" or self.eval_cfg.function_type == "diverse2":
                    random_string_2, random_string_2_token_idx = self.sample_string()
                else:
                    random_string_2 = ""
                    random_string_2_token_idx = []

                # Replace strings in each doc
                for doc, seq_info in zip(batch_docs, batch_seq_infos):
                    new_doc = doc[: seq_info["prompt_pos_end"]]
                    new_doc = self._replace_string(
                        new_doc, random_string_1_token_idx, random_string_2_token_idx
                    )
                    batch_inputs.append(new_doc)

                    # get actual outputs 
                    train_function_list = self.get_function_list_decoded(
                        batch_docs, self.token_idx[self.sep_token], self.token
                    )
                    print(len(train_function_list))
                    actual_output_batch = self.generate_actual_outputs(
                        train_function_list, random_string_1, random_string_2
                    )

                    if self.eval_cfg.function_type == "diverse" or self.eval_cfg.function_type == "diverse2":
                        input_strings.append(random_string_1 + random_string_2)
                    else:
                        input_strings.append(random_string_1)

            else:
                seq_info = batch_seq_infos[0]
                actual_output_batch = batch_docs[:, seq_info["prompt_pos_end"]:]
                # Use existing strings from docs
                for doc, seq_info in zip(batch_docs, batch_seq_infos):
                    
                    new_doc = doc[: seq_info["prompt_pos_end"]]
                    batch_inputs.append(new_doc)
                    input_str1, input_str2 = self._extract_inputs_from_doc(
                        doc, self.token_idx[self.sep_token], tokens_to_string=True
                    )

                    input_strings.append(input_str1 + input_str2)
            max_len = max(len(inp) for inp in batch_inputs)
            padded_inputs = [
                np.pad(inp, (0, max_len - len(inp)), "constant") for inp in batch_inputs
            ]
            inp_batch = torch.tensor(padded_inputs).to(device)

            # Extract representations
            last_token_repr_tensor, predicted_output_batch = extractor.extract_representation(
                inp_batch, batch_seq_infos[0]
            )
            predicted_output_batch = predicted_output_batch.detach().cpu().numpy()
            

            # decode the output batch
            decoded_predicted_output_batch = []
            for i in range(len(predicted_output_batch)):
                decoded_predicted_output_batch.append([self.token[j] for j in predicted_output_batch[i]])
            predicted_output_strings.extend(decoded_predicted_output_batch)

            decoded_actual_output_batch = []
            for i in range(len(actual_output_batch)):
                decoded_actual_output_batch.append([self.token[j] for j in actual_output_batch[i]])
            actual_output_strings.extend(decoded_actual_output_batch)
            # Take the last token representations
            all_representations.extend(last_token_repr_tensor)

            # Extract function lists
            batch_fn_lists = [get_function_list(doc, sep_token) for doc in batch_docs]
            batch_function_lists = [
                [self.token[fn] for fn in fn_list] for fn_list in batch_fn_lists
            ]
            all_function_lists.extend(batch_function_lists)

        extractor.remove_hook()

        return (
            np.array(all_representations),
            all_function_lists,
            input_strings,
            predicted_output_strings,
            actual_output_strings,
        )

    def analyze_representations(self, net, ck):
        """
        Perform representation analysis on train and test data

        Args:
            net: Trained model
            ck: Checkpoint number

        Returns:
            dict: Analysis results including representations and t-SNE embeddings
        """
        self.logger.info("Starting representation analysis...")
        self.logger.info(f"Checkpoint: {ck}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval()
        if device == "cuda":
            net = net.cuda()

        # Load corpus data
        docs_dict = self._load_corpus_data(
            sampled_train_docs=False, cross_eval_flag=True
        )
        train_docs = np.array(docs_dict["train"])
        test_docs = np.array(docs_dict["test"])

        self.logger.info(f"Train data: {len(train_docs)} samples")
        self.logger.info(f"Test data: {len(test_docs)} samples")

        # Extract representations for train data
        self.logger.info("Extracting train representations...")
        (
            train_representations,
            train_function_lists,
            train_input_strings,
            train_predicted_output_strings,
            train_actual_output_strings,
        ) = self.extract_all_representations(net, train_docs, device)

        # Extract representations for test data
        self.logger.info("Extracting test representations...")
        (
            test_representations,
            test_function_lists,
            test_input_strings,
            test_predicted_output_strings,
            test_actual_output_strings,
        ) = self.extract_all_representations(net, test_docs, device)

        self.logger.info(
            f"Extracted {len(train_representations)} train representations"
        )
        self.logger.info(f"Extracted {len(test_representations)} test representations")

        # Combine all data
        all_function_lists = train_function_lists + test_function_lists
        input_strings = train_input_strings + test_input_strings
        predicted_output_strings = train_predicted_output_strings + test_predicted_output_strings
        actual_output_strings = train_actual_output_strings + test_actual_output_strings

        # Prepare results
        results = {
            "train_representations": train_representations,
            "test_representations": test_representations,
            "train_function_lists": train_function_lists,
            "test_function_lists": test_function_lists,
            "train_input_strings": train_input_strings,
            "test_input_strings": test_input_strings,
            "train_predicted_output_strings": train_predicted_output_strings,
            "test_predicted_output_strings": test_predicted_output_strings,
            "train_actual_output_strings": train_actual_output_strings,
            "test_actual_output_strings": test_actual_output_strings,
            "all_function_lists": all_function_lists,
            "input_strings": input_strings,
            "predicted_output_strings": predicted_output_strings,
            "actual_output_strings": actual_output_strings,
        }

        # Perform t-SNE analysis
        self.logger.info("Performing t-SNE analysis...")
        embeddings, perm_labels, dataset_labels, unique_perms, simplified_perms = (
            perform_tsne_and_visualize(
                train_representations,
                test_representations,
                train_function_lists,
                test_function_lists,
            )
        )

        # Add t-SNE results
        results.update(
            {
                "embeddings": embeddings,
                "perm_labels": perm_labels,
                "dataset_labels": dataset_labels,
                "unique_perms": unique_perms,
                "simplified_perms": simplified_perms,
                "original_perms": all_function_lists,
            }
        )

        return results

    def save_representation_results(
        self, cfg, results, model_path="models", run_num=None
    ):
        """Save accuracy results"""
        model_split = self.eval_cfg.model_split
        eval_split = self.eval_cfg.eval_split
        nheads_nlayers = f"nh{self.net_cfg.net.n_head}_nl{self.net_cfg.net.n_layer}"

        results_path = "{}/eval/{}/{}/{}/{}/model_{}/eval_{}/{}/{}/seed_{}".format(
            model_path,
            cfg.function_type,
            self.eval_cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
            cfg.tag,
            cfg.prompt_length,
            model_split,
            eval_split,
            cfg.pos_embedding_type,
            nheads_nlayers,
            cfg.seed,
        )

        os.makedirs(results_path, exist_ok=True)
        pickle.dump(
            results,
            open(
                results_path
                + f"/representation_results_{cfg.replace_strings_repr}.pkl",
                "wb",
            ),
        )
