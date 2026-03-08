import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch


def get_seq_info(token_idx, sep_token, end_token, sample, function_type):
    """Extract sequence information"""
    seq_info = {}
    sep_idx = token_idx[sep_token]
    total_len = len(sample)

    sep_pos = np.where(sample == sep_idx)[0]

    last_sep_pos = sep_pos[-1]
    if function_type == "uniform":
        third_sep_pos = sep_pos[1]
    else:
        third_sep_pos = sep_pos[2]
    end_token_pos = np.where(sample == token_idx[end_token])[0]

    seq_info["last_sep_pos"] = last_sep_pos
    seq_info["prompt_pos_end"] = third_sep_pos + 1
    seq_info["end_pos"] = end_token_pos[0]
    extra_space_tokens = total_len - seq_info["end_pos"] - 1
    seq_info["new_len"] = total_len - seq_info["prompt_pos_end"] - extra_space_tokens

    return seq_info


def get_function_list(doc, sep_idx):
    sep_pos = np.where(doc == sep_idx)[0][0]
    doc_function = doc[1:sep_pos]
    return doc_function


def get_input_string(doc, sep_idx, function_type):
    if function_type == "diverse":
        third_sep_pos = np.where(doc == sep_idx)[0][2]
    else:
        third_sep_pos = np.where(doc == sep_idx)[0][1]
    first_sep_pos = np.where(doc == sep_idx)[0][0]
    input_string = doc[first_sep_pos + 1 : third_sep_pos]
    return input_string


def get_output_string(doc, sep_idx, end_token, function_type):
    if function_type == "diverse":
        third_sep_pos = np.where(doc == sep_idx)[0][2]
    else:
        third_sep_pos = np.where(doc == sep_idx)[0][1]
    end_token_pos = np.where(doc == end_token)[0][0]
    output_string = doc[third_sep_pos + 1 : end_token_pos]
    return output_string


def map_docs_to_combination_id(token, token_idx, sep_token, functions_info, docs):
    """Map documents to their combination IDs"""
    sep_idx = token_idx[sep_token]
    docs_functions = []

    # have vectorized version of get_function_list
    docs_functions = np.array([get_function_list(doc, sep_idx) for doc in docs])

    combination_ids_map = {}
    combination_ids = []

    # vectorized version of the loop
    docs_function_token = [
        tuple([token[fn] for fn in docs_functions[i]])
        for i in range(len(docs_functions))
    ]
    # convert to list of tuples

    combination_ids = np.array(
        [
            functions_info[docs_function_token[i]]
            for i in range(len(docs_function_token))
        ]
    )
    combination_ids_map = {
        cid: docs_function_token[i] for i, cid in enumerate(combination_ids)
    }
    return combination_ids, combination_ids_map


def calculate_combination_accuracy(
    acc_array, ood_flags, combination_ids, use_sharp=True
):
    """Calculate accuracy grouped by combination ID"""
    if combination_ids is None:
        return {}

    print_error_indices = []

    combination_acc = {}
    combination_ood = {}
    combination_indices = {}

    for idx, combination_id in enumerate(combination_ids):
        if combination_id not in combination_acc:
            combination_acc[combination_id] = []
            combination_ood[combination_id] = []
            combination_indices[combination_id] = []
        if use_sharp:
            acc_val = (
                acc_array[idx].all().float().item()
                if hasattr(acc_array[idx], "all")
                else acc_array[idx].all()
            )
            ood_val = ood_flags[idx]
        else:
            acc_val = (
                acc_array[idx].float().mean().item()
                if hasattr(acc_array[idx], "float")
                else acc_array[idx].mean()
            )
            ood_val = ood_flags[idx]

        combination_acc[combination_id].append(acc_val)
        combination_ood[combination_id].append(ood_val)
        combination_indices[combination_id].append(idx)
    # find total number of unique combination ids
    total_unique_combination_ids = len(set(combination_ids))
    # per unique combination id, print 10% of the indices where the accuracy is 0
    for cid in combination_acc:

        if len(combination_acc[cid]) > 0 and cid not in [189, 153]:
            # find top 10 indices where the accuracy is 0
            K = min(10, len(combination_acc[cid]))
            top_K_indices = np.argsort(combination_acc[cid])[:K]
            top_K_indices = [combination_indices[cid][i] for i in top_K_indices]
            print_error_indices.extend(top_K_indices)
        else:
            # print all indices
            print_error_indices = combination_indices[cid]
    print_error_indices = list(set(print_error_indices))
    return (
        {cid: np.mean(accs) for cid, accs in combination_acc.items()},
        {cid: np.mean(oods) for cid, oods in combination_ood.items()},
        total_unique_combination_ids,
        print_error_indices,
    )


def get_print_indices(combination_ids):
    """Get indices for printing examples"""
    print_indices = []
    if combination_ids is not None:
        unique_ids = list(np.unique(np.array(combination_ids)))
        for id in unique_ids:
            indices = np.where(np.array(combination_ids) == id)
            print_indices.append(indices[0][-1])
    return print_indices


def is_ood_prompt_vectorized(
    token, token_idx, output_batch, target_output_batch, prompt_length
):
    """Vectorized OOD detection keeping original logic"""
    task_tokens = ["sort", "map", "filter", "union", "join", "max", "identity"]
    if prompt_length == "variable":
        structure_tokens = ["<SEP>", "<END>", " "]
    else:
        structure_tokens = ["<SEP>", "<END>"]

    # Convert to indices for vectorized operations
    task_indices = torch.tensor(
        [token_idx[t] for t in task_tokens if t in token_idx],
        dtype=output_batch.dtype,
        device=output_batch.device,
    )
    structure_indices = torch.tensor(
        [token_idx[t] for t in structure_tokens if t in token_idx],
        dtype=output_batch.dtype,
        device=output_batch.device,
    )

    # Check task tokens in output (vectorized is_function_in_output)
    has_task_tokens = torch.isin(output_batch, task_indices).any(dim=1)

    # Check structure mismatch (vectorized is_structure_mismatch)
    structure_mask = torch.isin(output_batch, structure_indices)
    structure_mismatch = structure_mask & (output_batch != target_output_batch)
    has_structure_mismatch = structure_mismatch.any(dim=1)

    # Print OOD examples for debugging
    ood_flags = has_task_tokens | has_structure_mismatch
    if ood_flags.any():
        print("\n=== {} OOD Examples Found ===".format(ood_flags.sum()))
        # print the first 5 ood examples
        MAX_PRINT = min(5, len(ood_flags))
        for i in range(MAX_PRINT):
            if ood_flags[i]:
                print(f"Sample {i} (OOD):")
                output_tokens = [token[t.item()] for t in output_batch[i]]
                target_tokens = [token[t.item()] for t in target_output_batch[i]]
                print(f"  Output: {output_tokens}")
                print(f"  Target: {target_tokens}")
                print(f"  Has task tokens: {has_task_tokens[i].item()}")
                print(f"  Has structure mismatch: {has_structure_mismatch[i].item()}")
                print()

    return ood_flags


def is_ood_prompt(token, token_idx, output_batch, target_output_batch, prompt_length):
    """Optimized batch OOD detection"""
    return torch.tensor(
        is_ood_prompt_vectorized(
            token, token_idx, output_batch, target_output_batch, prompt_length
        )
    )
