import importlib.util
import json
import logging
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data_generation.generator import SyntheticData
from src.data_generation.utils import *
from src.evaluation.fixed_evaluator import FixedPromptEvaluator


class VariablePromptEvaluator(FixedPromptEvaluator):
    """Optimized evaluator for variable prompt lengths with vectorized operations"""

    def __init__(self, eval_cfg, net_cfg, nbatch, log_path):
        super().__init__(eval_cfg, net_cfg, nbatch, log_path)
        self.data_path = eval_cfg.data_path
        self.eval_cfg = eval_cfg
        self.net_cfg = net_cfg
        self.nbatch = nbatch

    def _process_variable_prompt_batch(
        self, net, inp, new_len, total_len, batch_size, device
    ):
        """Process variable prompt batch with optimized batching"""
        outputs = []

        for batch_start in range(0, inp.size(0), batch_size):
            batch_end = min(batch_start + batch_size, inp.size(0))
            inp_batch = inp[batch_start:batch_end]

            # Generate predictions
            inp_batch = self._generate_predictions(net, inp_batch, new_len)

            # Vectorized padding with space tokens
            inp_batch = self._pad_to_total_length(inp_batch, total_len, device)
            outputs.append(inp_batch)

        return torch.cat(outputs, dim=0)

    def _pad_to_total_length(self, inp_batch, total_len, device):
        """Vectorized padding to total length"""
        remaining_len = total_len - inp_batch.shape[1]
        if remaining_len > 0:
            padding = torch.full(
                (inp_batch.shape[0], remaining_len),
                self.token_idx[self.space_token],
                device=device,
            )
            inp_batch = torch.cat((inp_batch, padding), dim=1)
        return inp_batch

    def _extract_function_segments(self, dat_batch, sep_token_idx):
        """Vectorized extraction of function segments from batch"""
        batch_size = dat_batch.shape[0]
        dat_np = dat_batch.cpu().numpy()

        all_function_lists = []
        all_decoded_lists = []
        all_segments = []

        for i in range(batch_size):
            # Find separator positions
            sep_positions = np.where(dat_np[i] == sep_token_idx)[0]

            # Extract function segments
            segments = []
            start_pos = 0
            for sep_pos in sep_positions:
                segments.append((start_pos, sep_pos))
                start_pos = sep_pos + 1

            # Extract function list and decode
            function_list = get_function_list(dat_np[i], sep_token_idx)
            decoded_list = self.decode(function_list, return_list=True)

            all_function_lists.append(function_list)
            all_decoded_lists.append(decoded_list)
            all_segments.append(segments)

        return all_function_lists, all_decoded_lists, all_segments

    def _compute_segment_accuracy(self, output_batch, target_batch, segments_batch):
        """Vectorized computation of segment-wise accuracy"""
        batch_size = output_batch.shape[0]
        segment_accuracies = []

        for i in range(batch_size):
            row_accuracies = []
            for start_pos, end_pos in segments_batch[i]:
                # Compare segments
                segment_match = torch.all(
                    output_batch[i, start_pos:end_pos]
                    == target_batch[i, start_pos:end_pos]
                ).item()
                row_accuracies.append(segment_match)
            segment_accuracies.append(row_accuracies)

        return segment_accuracies

    def _aggregate_module_wise_accuracy(self, decoded_lists, segment_accuracies):
        """Aggregate module-wise accuracy across batch"""
        module_wise_acc = defaultdict(lambda: {"acc": [], "total": 0})

        for decoded_list, accuracies in zip(decoded_lists, segment_accuracies):
            for module_name, is_correct in zip(decoded_list, accuracies):
                module_wise_acc[module_name]["acc"].append(is_correct)
                module_wise_acc[module_name]["total"] += 1

        return module_wise_acc

    def _compute_step_by_step_accuracy(self, segment_accuracies):
        """Compute step-by-step accuracy metrics"""
        step_by_step_acc = {
            "individual": defaultdict(list),
            "cumulative": defaultdict(list),
        }

        for accuracies in segment_accuracies:
            for step_idx, is_correct in enumerate(accuracies):
                step_by_step_acc["individual"][step_idx].append(is_correct)

                # Cumulative accuracy (all previous steps must be correct)
                if step_idx == 0:
                    cumulative_correct = is_correct
                else:
                    prev_cumulative = (
                        step_by_step_acc["cumulative"][step_idx - 1][-1]
                        if step_by_step_acc["cumulative"][step_idx - 1]
                        else True
                    )
                    cumulative_correct = prev_cumulative and is_correct

                step_by_step_acc["cumulative"][step_idx].append(cumulative_correct)

        return step_by_step_acc

    def _error_analysis_sbs(self, dat, output_l, targets_l):
        """Optimized step-by-step error analysis using vectorized operations"""
        # Extract function information
        all_function_lists, all_decoded_lists, all_segments = (
            self._extract_function_segments(dat, self.token_idx[self.sep_token])
        )

        # Compute segment-wise accuracy
        segment_accuracies = self._compute_segment_accuracy(
            output_l, targets_l, all_segments
        )

        # Aggregate results
        module_wise_acc = self._aggregate_module_wise_accuracy(
            all_decoded_lists, segment_accuracies
        )
        step_by_step_acc = self._compute_step_by_step_accuracy(segment_accuracies)

        return module_wise_acc, step_by_step_acc

    def _batch_ood_detection(self, output_batch, target_output_batch):
        """Vectorized OOD detection for batch"""
        ood_flags = is_ood_prompt(
            self.token,
            self.token_idx,
            output_batch,
            target_output_batch,
            self.eval_cfg.prompt_length,
        )
        return ood_flags

    def _compute_batch_metrics(
        self, dat_batches, output_batches, prompt_pos_end, new_len
    ):
        """Compute metrics for a single batch with same prompt configuration"""
        # Extract relevant portions
        target_batch = dat_batches[:, prompt_pos_end:]
        output_batch = output_batches[:, prompt_pos_end:]

        # Compute accuracy
        acc_batch = output_batch == target_batch
        sharp_acc = acc_batch.all(dim=-1)

        # Compute OOD flags
        ood_flags = self._batch_ood_detection(output_batch, target_batch)

        # Step-by-step analysis
        module_wise_acc, step_by_step_acc = self._error_analysis_sbs(
            dat_batches, output_batch, target_batch
        )

        return {
            "sharp_acc": sharp_acc,
            "ood_flags": ood_flags,
            "module_wise_acc": module_wise_acc,
            "step_by_step_acc": step_by_step_acc,
        }

    def _merge_module_wise_results(self, all_results):
        """Merge module-wise results from multiple batches"""
        merged_dict = {}

        for result in all_results:
            module_wise_acc = result["module_wise_acc"]
            for key, value in module_wise_acc.items():
                if key not in merged_dict:
                    merged_dict[key] = {"acc": [], "total": 0}
                merged_dict[key]["acc"].extend(value["acc"])
                merged_dict[key]["total"] += value["total"]
                assert len(merged_dict[key]["acc"]) == merged_dict[key]["total"]
                merged_dict[key]["total_acc"] = np.mean(merged_dict[key]["acc"])
        return merged_dict

    def _merge_step_by_step_results(self, all_results):
        """Merge step-by-step results from multiple batches"""
        merged_dict = {"individual": defaultdict(list), "cumulative": defaultdict(list)}

        for result in all_results:
            step_by_step_acc = result["step_by_step_acc"]
            for key, value in step_by_step_acc["individual"].items():
                merged_dict["individual"][key].extend(value)
            for key, value in step_by_step_acc["cumulative"].items():
                merged_dict["cumulative"][key].extend(value)

        return merged_dict

    def _error_metrics_sbs(self, dat, output, grouped_indices, combination_ids):
        """Optimized step-by-step error metrics calculation"""
        all_results = []
        all_sharp_acc = []
        all_ood_flags = []

        for (prompt_pos_end, new_len), indices in grouped_indices.items():
            dat_batch = dat[indices]
            output_batch = output[indices]

            # Compute metrics for this batch
            batch_results = self._compute_batch_metrics(
                dat_batch, output_batch, prompt_pos_end, new_len
            )

            all_results.append(batch_results)
            all_sharp_acc.append(batch_results["sharp_acc"])
            all_ood_flags.append(batch_results["ood_flags"])

        # Merge results
        sharp_acc = torch.cat(all_sharp_acc, dim=0)
        ood_flags = torch.cat(all_ood_flags, dim=0)

        # Calculate overall metrics
        sharp_acc_mean = sharp_acc.float().mean().cpu().numpy()
        ood_mean = ood_flags.float().mean().cpu().numpy()
        # convert ood_flags to list
        ood_flags = ood_flags.cpu().tolist()

        # Merge complex metrics
        merged_module_wise = self._merge_module_wise_results(all_results)
        merged_step_by_step = self._merge_step_by_step_results(all_results)

        # Calculate combination accuracy
        sharp_combination_acc, ood_combination_acc = calculate_combination_accuracy(
            sharp_acc, ood_flags, combination_ids, use_sharp=True
        )

        return {
            "total": {"acc": sharp_acc_mean, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
            "module_wise": merged_module_wise,
            "step_by_step": merged_step_by_step,
            "direct": {},
        }

    def _error_metrics_direct(self, dat, output, grouped_indices, combination_ids):
        """Optimized direct error metrics calculation for variable prompt mode"""
        all_acc = []
        all_ood_flags = []

        for (prompt_pos_end, new_len), indices in grouped_indices.items():
            output_batch = output[indices, prompt_pos_end : prompt_pos_end + new_len]
            target_batch = dat[indices, prompt_pos_end : prompt_pos_end + new_len]

            # Vectorized accuracy calculation
            acc_batch = (output_batch == target_batch).all(dim=-1)

            # Batch OOD detection
            ood_flags = self._batch_ood_detection(output_batch, target_batch)

            all_acc.append(acc_batch)
            all_ood_flags.append(ood_flags)

        # Concatenate results
        acc_combined = torch.cat(all_acc, dim=0)
        ood_combined = torch.cat(all_ood_flags, dim=0)

        # Calculate metrics
        sharp_acc = acc_combined.float().mean().cpu().numpy()
        ood_mean = ood_combined.float().mean().cpu().numpy()

        # Vectorized combination accuracy
        sharp_combination_acc, ood_combination_acc = calculate_combination_accuracy(
            acc_combined, ood_combined, combination_ids, use_sharp=True
        )

        return {
            "total": {"acc": sharp_acc, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
        }

    def _calculate_metrics(
        self, dat, output, grouped_indices, combination_ids, prompt_mode
    ):
        """Optimized metrics calculation dispatcher"""
        if prompt_mode == "step_by_step":
            return self._error_metrics_sbs(
                dat, output, grouped_indices, combination_ids
            )
        elif prompt_mode in ["direct", "curriculum"]:
            return self._error_metrics_direct(
                dat, output, grouped_indices, combination_ids
            )
        else:
            raise ValueError(f"Invalid prompt mode: {prompt_mode}")

    def _group_indices_by_prompt_config(self, seq_info_list):
        """Group indices by prompt configuration"""
        grouped_indices = defaultdict(list)
        for idx, seq_info in enumerate(seq_info_list):
            key = (seq_info["prompt_pos_end"], seq_info["new_len"])
            grouped_indices[key].append(idx)
        return grouped_indices

    def _process_grouped_batches(self, net, dat, grouped_indices, total_len, device):
        """Process all grouped batches and return concatenated results"""
        outputs_list = []
        inputs_list = []
        combination_ids_list = []

        for (prompt_pos_end, new_len), indices in grouped_indices.items():
            self.logger.info(
                f"Processing prompt pos: {prompt_pos_end}, new length: {new_len}, samples: {len(indices)}"
            )

            # Extract batch data
            inp = dat[indices, :prompt_pos_end]
            full_inp = dat[indices, :]

            # Process batch
            outputs = self._process_variable_prompt_batch(
                net, inp, new_len, total_len, self.eval_cfg.nbatch, device
            )

            outputs_list.append(outputs)
            inputs_list.append(full_inp)

        return torch.cat(outputs_list, dim=0), torch.cat(inputs_list, dim=0)

    def evaluate_docs(
        self, ck, split, net, dat, seq_info_list, combination_ids, device
    ):
        """Evaluate documents with variable prompt lengths - main evaluation method"""
        self.logger.info(
            f"Evaluating {split} documents with grouped variable prompts: {len(dat)}"
        )

        # Setup data
        shape = dat.shape
        if device == "cuda":
            dat = dat.cuda(non_blocking=True)
        dat = dat.view(-1, shape[-1])
        total_len = dat.shape[1]

        # Group indices by prompt configuration
        grouped_indices = self._group_indices_by_prompt_config(seq_info_list)

        # Get print indices for logging
        print_indices = get_print_indices(combination_ids)

        # Process all grouped batches
        output, inputs = self._process_grouped_batches(
            net, dat, grouped_indices, total_len, device
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            inputs, output, grouped_indices, combination_ids, self.eval_cfg.prompt_mode
        )

        # Log predictions if available
        if len(inputs) > 0:
            self._log_predictions(print_indices, ck, split, inputs, output)

        return metrics

    def get_acc(self, ck, net, verbose=False):
        """Get accuracy for standard evaluation with optimized processing"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval()
        if device == "cuda":
            net = net.cuda()

        docs_dict = self._load_corpus_data()
        acc_map = {}

        for split, docs in docs_dict.items():
            # Prepare data
            sampled_docs = np.array(docs)
            flatten_docs = torch.Tensor(sampled_docs.astype(int)).long()

            # Generate combination IDs and sequence info
            combination_ids, doc_combination_id_map = map_docs_to_combination_id(
                self.token,
                self.token_idx,
                self.sep_token,
                self.functions_info,
                sampled_docs,
            )

            seq_info = [
                get_seq_info(
                    self.token_idx,
                    self.sep_token,
                    self.end_token,
                    doc,
                    self.eval_cfg.function_type,
                )
                for doc in docs
            ]

            # Evaluate
            metrics = self.evaluate_docs(
                ck, split, net, flatten_docs, seq_info, combination_ids, device
            )

            acc_map[split] = metrics

            if verbose:
                self._log_verbose_results(
                    split, docs, doc_combination_id_map, metrics, ck
                )

        return acc_map

    # Inherit methods from parent class
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)

    def _generate_predictions(self, *args, **kwargs):
        return super()._generate_predictions(*args, **kwargs)

    def _log_predictions(self, *args, **kwargs):
        return super()._log_predictions(*args, **kwargs)

    def _load_corpus_data(self, *args, **kwargs):
        return super()._load_corpus_data(*args, **kwargs)

    def _log_verbose_results(self, *args, **kwargs):
        return super()._log_verbose_results(*args, **kwargs)

    def save_accs(self, cfg, accs, model_path="models", run_num=None):
        """Save accuracy results"""
        model_split = self.eval_cfg.model_split
        eval_split = self.eval_cfg.eval_split
        nheads_nlayers = f"nh{self.net_cfg.net.n_head}_nl{self.net_cfg.net.n_layer}"
        if cfg.num_runs == 1:
            results_path = "{}/eval/{}/{}/{}/model_{}/eval_{}/{}/{}/".format(
                model_path,
                self.eval_cfg.data_n_alphabets_seq_len_fn_len,
                cfg.tag,
                cfg.prompt_length,
                model_split,
                eval_split,
                cfg.pos_embedding_type,
                nheads_nlayers,
            )
        else:
            results_path = "{}/eval/{}/{}/{}/model_{}/eval_{}/{}/{}/runs/run_{}".format(
                model_path,
                self.eval_cfg.data_n_alphabets_seq_len_fn_len,
                cfg.tag,
                cfg.prompt_length,
                model_split,
                eval_split,
                cfg.pos_embedding_type,
                nheads_nlayers,
                run_num,
            )

        os.makedirs(results_path, exist_ok=True)
        pickle.dump(accs, open(results_path + "/accs.pkl", "wb"))
