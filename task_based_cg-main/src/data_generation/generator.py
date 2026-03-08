import argparse
import itertools
import json
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_generation.functions import CreateFunctions, apply_function_composition
from src.data_generation.init import read_config

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"
VARIABLE_MAX_PROMPT_LENGHTS = {"direct": 40, "curriculum": 40, "step_by_step": 102}


class SyntheticData:
    """
    Generates a synthetic sequence of the form
     t, x, t(x)
    """

    def __init__(
        self, cfg, train_functions, test_functions, function_dict, functions_info
    ):
        self.cfg = cfg
        self.n_alphabets = cfg.n_alphabets
        self.seq_len = cfg.seq_len
        self.function_properties = cfg.function
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
        self.train_functions = train_functions
        self.test_functions = test_functions
        self.function_dict = function_dict
        self.functions_info = functions_info
        self.dir_flag = self.cfg.function.split.strategy
        self.n_alphabets_seq_len_fn_len_task_max_length = (
            "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
                self.n_alphabets,
                self.seq_len,
                self.cfg.function.n_functions,
                self.cfg.task_max_length,
            )
        )
        self.step_fdir = "{}/data/{}/{}/{}/step_by_step/{}".format(
            ROOT_DIR,
            self.cfg.function.type,
            self.cfg.prompt_length,
            self.n_alphabets_seq_len_fn_len_task_max_length,
            self.dir_flag,
        )
        self.direct_fdir = "{}/data/{}/{}/{}/direct/{}".format(
            ROOT_DIR,
            self.cfg.function.type,
            self.cfg.prompt_length,
            self.n_alphabets_seq_len_fn_len_task_max_length,
            self.dir_flag,
        )
        self.curriculum_fdir = "{}/data/{}/{}/{}/curriculum/{}".format(
            ROOT_DIR,
            self.cfg.function.type,
            self.cfg.prompt_length,
            self.n_alphabets_seq_len_fn_len_task_max_length,
            self.dir_flag,
        )
        os.makedirs(self.step_fdir, exist_ok=True)
        os.makedirs(self.direct_fdir, exist_ok=True)
        os.makedirs(self.curriculum_fdir, exist_ok=True)
        # Initialize logger
        log_path = "{}/logs/{}".format(ROOT_DIR, self.cfg.function.type)
        os.makedirs(log_path, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        log_file_dir = "{}/{}/{}/{}/model_{}".format(
            log_path,
            self.n_alphabets_seq_len_fn_len_task_max_length,
            "direct",
            self.cfg.prompt_length,
            self.dir_flag,
        )
        print("log_file_dir", log_file_dir)
        print("data_dir", self.direct_fdir)
        os.makedirs(log_file_dir, exist_ok=True)
        # Set up logging configuration
        logging.basicConfig(
            filename="{}/data.log".format(log_file_dir),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )

    def init_tokens(self):
        """
        Initializes the tokens for the sequence.
        """
        print("init_tokens")
        self.token = {}
        self.token_idx = {}
        # create a list of all possible data tokens between a-z and a null token
        for i in range(self.n_alphabets):
            self.token[i] = chr(i + 97)
            self.token_idx[chr(i + 97)] = i

        # special tokens
        sp_token_count = 0
        for i, token in enumerate(self.special_tokens):
            if self.cfg.function.split.strategy == "sort" and token not in [
                self.start_token,
                self.sep_token,
                self.end_token,
                self.null_token,
                self.space_token,
            ]:
                continue
            if self.cfg.function.split.strategy == "sort_map" and token not in [
                self.start_token,
                self.sep_token,
                self.end_token,
                self.null_token,
                self.space_token,
            ]:
                continue
            self.token[self.n_alphabets + sp_token_count] = token
            self.token_idx[token] = self.n_alphabets + sp_token_count
            sp_token_count += 1

        # function tokens
        count = 0
        for i, token in enumerate(self.function_dict.keys()):
            if self.cfg.function.split.strategy == "sort" and token not in ["sort"]:
                continue
            if self.cfg.function.split.strategy == "sort_map" and token not in [
                "sort",
                "map",
            ]:
                continue
            self.token[self.n_alphabets + sp_token_count + count] = token
            self.token_idx[token] = self.n_alphabets + sp_token_count + count
            count += 1

        self.start_idx = np.array([self.token_idx[self.start_token]])
        self.sep_idx = np.array([self.token_idx[self.sep_token]])
        self.end_idx = np.array([self.token_idx[self.end_token]])
        if self.cfg.function.split.strategy != "sort":
            self.space_idx = np.array([self.token_idx[self.space_token]])
            self.null_idx = np.array([self.token_idx[self.null_token]])

        self.logger.info("Tokens: {}".format(self.token))
        self.logger.info("Token indices: {}".format(self.token_idx))

    def all_possible_strings(self):
        """
        Returns all possible strings of the given length.
        """
        alph = [chr(i + 97) for i in range(self.n_alphabets)]
        return ["".join(p) for p in itertools.product(alph, repeat=self.cfg.seq_len)]

    def sample_string(self):
        """
        Samples a string of the given length.
        """
        # sample a string of the given length
        alph = [chr(i + 97) for i in range(self.n_alphabets)]
        tokens = np.random.choice(
            alph, size=self.cfg.seq_len, replace=self.cfg.with_replacement
        )
        token_idx = [self.token_idx[c] for c in tokens]
        tokens = "".join(tokens)
        return tokens, token_idx

    def decode(self, token_indices, return_list=False):
        # txt_list = [self.token[i] for i in token_indices]
        txt_list = []
        for i in token_indices:
            if i in self.token:
                txt_list.append(self.token[i])
            # add a space for readability
            if not return_list:
                txt_list.append(" ")
        # remove the last space
        if txt_list[-1] == " ":
            txt_list = txt_list[:-1]

        return "".join(txt_list) if not return_list else txt_list

    def encode(self, txt):
        # encode the string to token indices
        return [self.token_idx[c] for c in txt]

    def get_function_list_for_curriculum_fixed_prompt(self, function_list):
        function_lists = []

        max_combination_id = max(list(self.functions_info.values()))
        count = max_combination_id + 1
        for i in range(len(function_list) - 1):
            function_list_i = []
            # add atleast 2 functions from the function list
            for j in range(i + 2):
                function_list_i.append(function_list[j])
            N = len(function_list)
            # add identity functions to the function list for remaining functions
            for j in range(N - i - 2):
                function_list_i.append("identity")
            sorted_function_list_i = tuple(function_list_i)

            if sorted_function_list_i not in self.functions_info:
                self.functions_info[sorted_function_list_i] = count
                count += 1
            function_lists.append(function_list_i)
        return function_lists

    def get_function_list_for_curriculum_variable_prompt(self, function_list):
        orig_function_list = function_list
        function_lists = [orig_function_list]

        if len(orig_function_list) > 2:
            max_combination_id = max(list(self.functions_info.values()))
            count = max_combination_id + 1
            for i in range(2, len(function_list)):
                function_list_i = []
                for j in range(i + 1):
                    function_list_i.append(function_list[j])
                sorted_function_list_i = tuple(function_list_i)

                if sorted_function_list_i not in self.functions_info:
                    self.functions_info[sorted_function_list_i] = count
                    count += 1
                function_lists.append(function_list_i)
        return function_lists

    def generate_curriculum_data(
        self, function_list, xstr1, xstr2, filter_func, offset
    ):
        # create function_list in such a way that [f1, f2, f3, f4, f5, f6, f7] is split into
        curriculum_function_list = []
        output_list = []
        currculum_documents = []

        xstr1_tokens = np.array([self.token_idx[c] for c in xstr1])
        xstr2_tokens = np.array([self.token_idx[c] for c in xstr2])

        function_lists = self.get_function_list_for_curriculum_fixed_prompt(
            function_list
        )

        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            # apply the function to the string
            outputs = apply_function_composition(
                self.cfg.n_alphabets,
                function_list_copy,
                self.function_dict,
                xstr1,
                xstr2,
                filter_func,
                offset,
            )

            # add padding to the outputs if the length is less than the 2*sequence length
            pad_length = 2 * self.seq_len
            if self.cfg.function.split.strategy in ["sort_map", "sort"]:
                pad_length = self.seq_len
            for i in range(len(outputs)):
                if self.cfg.prompt_length == "fixed":
                    if len(outputs[i]) < pad_length:
                        outputs[i] = outputs[i] + self.space_token * (
                            pad_length - len(outputs[i])
                        )
                    else:
                        outputs[i] = outputs[i][:pad_length]

            output_list.append(outputs)

        for i in range(len(curriculum_function_list)):
            # get the output tokens
            task_indices = self.get_task_tokens(curriculum_function_list[i])
            output_tokens = self.get_output_tokens(output_list[i])

            # create a document for the curriculum
            curriculum_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    xstr2_tokens,
                    self.sep_idx,
                    output_tokens[-1],
                    self.end_idx,
                ]
            )

            currculum_documents.append(curriculum_document)

        return currculum_documents

    def generate_curriculum_data_variable_prompt(
        self, function_list, xstr1, xstr2, filter_func, offset
    ):
        # create function_list in such a way that [f1, f2, f3, f4, f5, f6, f7] is split into
        curriculum_function_list = []
        output_list = []
        currculum_documents = []

        xstr1_tokens = np.array([self.token_idx[c] for c in xstr1])
        xstr2_tokens = np.array([self.token_idx[c] for c in xstr2])

        max_combination_id = max(list(self.functions_info.values()))
        count = max_combination_id + 1

        function_lists = self.get_function_list_for_curriculum_variable_prompt(
            function_list
        )

        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            # apply the function to the string
            outputs = apply_function_composition(
                self.cfg.n_alphabets,
                function_list_copy,
                self.function_dict,
                xstr1,
                xstr2,
                filter_func,
                offset,
            )

            output_list.append(outputs)

        for i in range(len(curriculum_function_list)):
            # get the output tokens
            task_indices = self.get_task_tokens(curriculum_function_list[i])
            output_tokens = self.get_output_tokens(output_list[i])

            # create a document for the curriculum

            curriculum_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    xstr2_tokens,
                    self.sep_idx,
                    output_tokens[-1],
                    self.end_idx,
                ]
            )

            currculum_documents.append(curriculum_document)

        return currculum_documents

    def get_task_tokens(self, function_list):
        """
        Returns the task tokens for the given function list.
        """
        task_indices = []
        for fn_name in function_list:
            task_indices.append(self.token_idx[fn_name])
        task_indices = np.array(task_indices)
        return task_indices

    def get_output_tokens(self, outputs):
        """
        Returns the output tokens for the given function list.
        """

        output_indices = []
        for output in outputs:
            output_idx = []
            if len(output) == 0:
                output_idx.append(self.token_idx["<NULL>"])
            for i in range(len(output)):
                output_idx.append(self.token_idx[output[i]])
            output_indices.append(output_idx)
        return output_indices

    def generate_different_prompts_data(self, function_list):
        """
        Generates different prompts data for the given split.
        """

        xstr1, xstr1_tokens = self.sample_string()
        xstr2, xstr2_tokens = self.sample_string()

        filter_func = lambda x: x in "aeiou"
        offset = 1

        task_indices = self.get_task_tokens(function_list)

        # apply the function to the string
        outputs = apply_function_composition(
            self.cfg.n_alphabets,
            function_list,
            self.function_dict,
            xstr1,
            xstr2,
            filter_func,
            offset,
        )
        # add padding to the outputs if the length is less than the 2*sequence length
        pad_length = 2 * self.seq_len
        if self.cfg.function.split.strategy in ["sort_map", "sort"]:
            pad_length = self.seq_len
        for i in range(len(outputs)):
            if self.cfg.prompt_length == "fixed":
                if len(outputs[i]) < pad_length:
                    outputs[i] = outputs[i] + self.space_token * (
                        pad_length - len(outputs[i])
                    )
                else:
                    outputs[i] = outputs[i][:pad_length]

        # get the output tokens

        output_tokens = self.get_output_tokens(outputs)

        if self.cfg.function.split.strategy in ["sort", "sort_map"]:
            direct_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    output_tokens[-1],
                    self.end_idx,
                ]
            )
            step_by_step_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    output_tokens[-1],
                    self.end_idx,
                ]
            )
        else:
            direct_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    xstr2_tokens,
                    self.sep_idx,
                    output_tokens[-1],
                    self.end_idx,
                ]
            )
            step_by_step_document = np.concatenate(
                [
                    self.start_idx,
                    task_indices,
                    self.sep_idx,
                    xstr1_tokens,
                    self.sep_idx,
                    xstr2_tokens,
                ]
            )
            for i in range(len(outputs)):
                step_by_step_document = np.concatenate(
                    [step_by_step_document, self.sep_idx, output_tokens[i]]
                )

            # add the end token
            step_by_step_document = np.concatenate(
                [step_by_step_document, self.end_idx]
            )

        if (
            self.cfg.function.split.strategy == "sort"
            or self.cfg.function.split.strategy == "sort_map"
        ):
            curriculum_documents = [step_by_step_document]
        else:
            if self.cfg.prompt_length == "fixed":
                curriculum_documents = self.generate_curriculum_data(
                    function_list, xstr1, xstr2, filter_func, offset
                )
            else:
                curriculum_documents = self.generate_curriculum_data_variable_prompt(
                    function_list, xstr1, xstr2, filter_func, offset
                )

        return direct_document, step_by_step_document, curriculum_documents

    def add_padding(
        self, direct_documents, step_by_step_documents, curriculum_documents_list
    ):
        """
        Adds padding to the documents so that the length of each sample in each document type is the same.
        Padding is applied separately for each document type.
        """

        # Helper function to pad a list of documents
        def pad_documents(documents, prompt_mode):
            if self.cfg.prompt_length == "variable":
                max_len = VARIABLE_MAX_PROMPT_LENGHTS[prompt_mode]
            else:
                if len(documents) == 0:
                    max_len = 0
                else:
                    max_len = max(len(doc) for doc in documents)
            return [
                (
                    np.pad(
                        doc, (0, max_len - len(doc)), constant_values=self.space_idx[0]
                    )
                    if len(doc) < max_len
                    else doc
                )
                for doc in documents
            ]

        # Pad each type of document separately
        direct_documents = pad_documents(direct_documents, "direct")
        step_by_step_documents = pad_documents(step_by_step_documents, "step_by_step")
        curriculum_documents_list = pad_documents(
            curriculum_documents_list, "curriculum"
        )

        # print the max length of each document type
        print(
            "Max length of direct documents: ",
            max(len(doc) for doc in direct_documents),
        )
        print(
            "Max length of step by step documents: ",
            max(len(doc) for doc in step_by_step_documents),
        )
        print(
            "Max length of curriculum documents: ",
            (
                max(len(doc) for doc in curriculum_documents_list)
                if len(curriculum_documents_list) > 0
                else 0
            ),
        )

        return direct_documents, step_by_step_documents, curriculum_documents_list

    def generate_document(self, split="train"):
        direct_documents = []
        step_by_step_documents = []
        curriculum_documents_list = []

        if split == "train":
            ndocuments = self.cfg.ndocuments
            n_train_functions = len(self.train_functions)
            nsamples = int(ndocuments / n_train_functions)
        elif split == "train_heldout":
            ndocuments = self.cfg.neval_documents
            n_train_heldout_functions = len(self.train_functions)
            nsamples = int(ndocuments / n_train_heldout_functions)
            if n_train_heldout_functions > ndocuments:
                nsamples = 1
        elif split == "test":
            ndocuments = self.cfg.neval_documents
            n_test_functions = len(self.test_functions)
            if n_test_functions > ndocuments:
                nsamples = 1
            else:
                nsamples = int(ndocuments / n_test_functions)
        else:
            raise ValueError("Invalid split: {}".format(split))

        if split == "train":
            functions = self.train_functions
        elif split == "train_heldout":
            functions = self.train_functions
        elif split == "test":
            functions = self.test_functions
        self.logger.info(
            "Sampling {} documents for {} functions {} samples/fns for {} split".format(
                ndocuments, len(functions), nsamples, split
            )
        )
        # add tqdm for progress bar
        for function_list in tqdm(functions):
            for i in tqdm(range(nsamples)):
                # generate different prompts data
                direct_document, step_by_step_document, curriculum_documents = (
                    self.generate_different_prompts_data(function_list)
                )
                direct_documents.append(direct_document)
                for curriculum_document in curriculum_documents:
                    curriculum_documents_list.append(curriculum_document)
                step_by_step_documents.append(step_by_step_document)

        direct_documents, step_by_step_documents, curriculum_documents_list = (
            self.add_padding(
                direct_documents, step_by_step_documents, curriculum_documents_list
            )
        )
        return direct_documents, step_by_step_documents, curriculum_documents_list

    def generate_corpus(self):

        train_documents, train_step_by_step_documents, train_curriculum_documents = (
            self.generate_document("train")
        )
        (
            train_heldout_documents,
            train_heldout_step_by_step_documents,
            train_heldout_curriculum_documents,
        ) = self.generate_document("train_heldout")
        test_documents, test_step_by_step_documents, test_curriculum_documents = (
            self.generate_document("test")
        )
        for i in range(1):
            self.logger.info("Direct documents")
            self.logger.info(len(train_documents[i]))
            self.logger.info(train_documents[i])
            self.logger.info(self.decode(train_documents[i]))
        for i in range(1):
            self.logger.info(len(train_step_by_step_documents[i]))
            self.logger.info(train_step_by_step_documents[i])
            self.logger.info("Step by step documents")
            self.logger.info(self.decode(train_step_by_step_documents[i]))
        self.logger.info("Curriculum documents")
        for i in range(10):
            if len(train_curriculum_documents) > 0:
                self.logger.info(len(train_curriculum_documents[i]))
                self.logger.info(train_curriculum_documents[i])
                self.logger.info(self.decode(train_curriculum_documents[i]))
        self.corpus = {}
        self.corpus["train_direct"] = train_documents
        self.corpus["train_step_by_step"] = train_step_by_step_documents
        self.corpus["train_curriculum"] = train_curriculum_documents
        self.corpus["test_direct"] = test_documents
        self.corpus["test_step_by_step"] = test_step_by_step_documents
        self.corpus["test_curriculum"] = test_curriculum_documents
        self.corpus["train_heldout_direct"] = train_heldout_documents
        self.corpus["train_heldout_step_by_step"] = train_heldout_step_by_step_documents
        self.corpus["train_heldout_curriculum"] = train_heldout_curriculum_documents

    def store_data(self):
        """
        Store the tokens into a file
        """

        modes = ["step_by_step", "direct", "curriculum"]
        for mode in modes:
            if mode == "step_by_step":
                mode_dir = self.step_fdir
            elif mode == "direct":
                mode_dir = self.direct_fdir
            elif mode == "curriculum":
                mode_dir = self.curriculum_fdir
            os.makedirs(mode_dir, exist_ok=True)
            pickle.dump(self.token_idx, open(mode_dir + "/token_idx.pkl", "wb"))

            pickle.dump(self.token, open(mode_dir + "/token.pkl", "wb"))

            pickle.dump(
                self.functions_info, open(mode_dir + "/functions_info.pkl", "wb")
            )

            np.save(
                mode_dir + "/train_{}_corpus.npy".format(mode),
                self.corpus["train_" + mode],
            )
            np.save(
                mode_dir + "/test_{}_corpus.npy".format(mode),
                self.corpus["test_" + mode],
            )
            np.save(
                mode_dir + "/train_heldout_{}_corpus.npy".format(mode),
                self.corpus["train_heldout_" + mode],
            )
            self.cfg["tag"] = mode
            if mode == "step":
                self.cfg["direct"] = False
                self.cfg["curriculum"] = False

            if mode == "direct":
                self.cfg["direct"] = True
                self.cfg["use_curriculum"] = False
            if mode == "curriculum":
                self.cfg["direct"] = True
                self.cfg["use_curriculum"] = True
            if mode == "step_by_step":
                self.cfg["direct"] = False
                self.cfg["use_curriculum"] = False
            # save the config file
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)

            json.dump(config_dict, open(mode_dir + "/config.json", "w"), indent=4)


class SyntheticDataset:
    """
    Dataset object to create a dataloader
    """

    def __init__(self, fpath, split="train", mode="step_by_step"):
        datafiles = {
            "train": os.path.join(fpath, "train_{}_corpus.npy".format(mode)),
            "test": os.path.join(fpath, "test_{}_corpus.npy".format(mode)),
            "train_heldout": os.path.join(
                fpath, "train_heldout_{}_corpus.npy".format(mode)
            ),
        }

        self.data = np.load(datafiles[split])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = torch.from_numpy(self.data[idx])
        dat, target = elem[:-1], elem[1:]
        return dat, target


def get_trainLoader(cfg):
    dataset = SyntheticDataset(cfg.data.path, split="train", mode=cfg.tag)
    # print sample data, target
    data, target = dataset[0]
    print("Sample data: ", data)
    print("Sample data shape: ", data.shape)
    print("Sample target: ", target)
    print("Sample target shape: ", target.shape)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
    )
    return dataloader


def get_evalLoaders(cfg):
    loaders = []
    for split in ["train", "test", "train_heldout"]:
        dataset = SyntheticDataset(cfg.data.path, split=split, mode=cfg.tag)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.data.num_workers,
        )
        loaders.append(dataloader)
    return loaders


def get_vocab_len(fpath):
    token = np.load(os.path.join(fpath, "token.pkl"), allow_pickle=True)
    return len(token)


def get_sep_pos(fpath, loader):
    token_idx = np.load(os.path.join(fpath, "token_idx.pkl"), allow_pickle=True)
    sep_idx = token_idx["<SEP>"]
    sep_pos = np.where(loader.dataset.data[0] == sep_idx)[0][-1]
    return sep_pos


def get_seq_info(fpath, loader, function_type):
    token_idx = np.load(os.path.join(fpath, "token_idx.pkl"), allow_pickle=True)

    seq_info = {}
    data_cfg = json.load(open(os.path.join(fpath, "config.json"), "r"))
    sep_idx = token_idx["<SEP>"]
    sample = loader.dataset.data[0]
    total_len = len(sample)

    if (not data_cfg["direct"]) and (not data_cfg["use_curriculum"]):
        # find all the positions of the sep token
        sep_pos = np.where(sample == sep_idx)[0]
        # find the position of last sep token
        last_sep_pos = sep_pos[-1]
        # find the position of the third sep token
        if function_type == "uniform":
            third_sep_pos = sep_pos[1]
        else:
            third_sep_pos = sep_pos[2]
        print("Sep positions: ", sep_pos)
        # find the position of the first sep token
        print("Last sep position: ", last_sep_pos)
        print("Third sep position: ", third_sep_pos)
        seq_info["last_sep_pos"] = last_sep_pos
        seq_info["prompt_pos_end"] = third_sep_pos + 1
        seq_info["new_len"] = total_len - seq_info["prompt_pos_end"]
    return seq_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_length", type=str, default="fixed", help="fixed or variable"
    )
    args = parser.parse_args()
    # Read config file
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    cfg = read_config(cfg_path)
    cfg.prompt_length = args.prompt_length

    # Create function object
    function_obj = CreateFunctions(cfg)

    # Create functions
    train_functions, test_functions, functions_info = function_obj.get_train_functions()
    # Create function dictionary
    function_dict = function_obj.function_dict
    # Create synthetic data object
    synthetic_data = SyntheticData(
        cfg,
        train_functions,
        test_functions,
        function_dict,
        functions_info,
    )
    # Initialize tokens
    synthetic_data.init_tokens()

    synthetic_data.generate_corpus()
