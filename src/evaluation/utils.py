import numpy as np
import pickle
import logging


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



def log_verbose_results(
    logger, split, docs, doc_combination_id_map, metrics, ck, prompt_mode
):
    """Log verbose results for standard evaluation"""
    logger.info("Evaluating {} documents: {}".format(split, len(docs)))
    combination_acc = metrics["combination"]["acc"]
    combination_ood = metrics["combination"]["ood"]
    total_acc = metrics["total"]["acc"]
    total_ood = metrics["total"]["ood"]
    if prompt_mode == "step_by_step":
        module_wise_acc = metrics["module_wise"]
        step_by_step_acc = metrics["step_by_step"]
        direct_acc = metrics["direct"]

    for cid, acc in combination_acc.items():
        logger.info(
            "Accuracy for combination id {} {} is {:.3f}".format(
                cid, doc_combination_id_map[cid], acc
            )
        )

    for cid, ood in combination_ood.items():
        logger.info(
            "OOD for combination id {} {} is {:.3f}".format(
                cid, doc_combination_id_map[cid], ood
            )
        )

    logger.info(
        "Iter: {} Split: {} Acc: {:.3f} OOD: {:.3f}".format(
            ck, split, total_acc, total_ood
        )
    )
    if prompt_mode == "step_by_step":
        for k, v in module_wise_acc.items():
            logger.info("Module wise acc for {} is {:.3f}".format(k, np.mean(v["acc"])))
        for k, v in step_by_step_acc.items():
            logger.info("Step by step acc for {} is {:.3f}".format(k, np.mean(v)))
        for k, v in direct_acc.items():
            logger.info("Direct acc for {} is {:.3f}".format(k, np.mean(v["acc"])))


# Additional utility function for batch processing
def batch_process_functions(
    dat_batch, token_idx, sep_token, decode, get_function_list, batch_size=None
):
    """Process function extraction for entire batch at once"""
    if batch_size is None:
        batch_size = dat_batch.shape[0]

    dat_np = dat_batch.cpu().numpy()
    sep_token_idx = token_idx[sep_token]

    # Vectorized function list extraction
    all_function_lists = []
    all_decoded_lists = []

    for i in range(batch_size):
        function_list = get_function_list(dat_np[i], sep_token_idx)
        decoded_list = decode(function_list, return_list=True)
        all_function_lists.append(function_list)
        all_decoded_lists.append(decoded_list)

    return all_function_lists, all_decoded_lists
