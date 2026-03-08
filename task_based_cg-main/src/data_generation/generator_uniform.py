import json
import logging
import os

import numpy as np

from src.data_generation.functions import apply_function_composition_uniform
from src.data_generation.generator import SyntheticData

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"


class SyntheticDataUniform(SyntheticData):
    """
    SyntheticData subclass for the 'uniform' function type (unary functions only).
    Overrides data generation to use only one input string and apply unary function compositions.
    """

    def __init__(
        self,
        cfg,
        train_functions,
        test_functions,
        function_dict,
        functions_info,
        function_mappings,
    ):
        # Call parent constructor properly
        super().__init__(
            cfg, train_functions, test_functions, function_dict, functions_info
        )
        self.functions_mappings = function_mappings
        """Override parent's init_dirs to use uniform-specific paths"""
        print("init_dirs - uniform version")
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

        # Store function mappings as json file
        with open(self.direct_fdir + "/function_mappings.json", "w") as f:
            json.dump(self.functions_mappings, f, indent=4)
        """Override parent's init_logging to use uniform-specific paths"""
        print("init_logging - uniform version")
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

        # Clear any existing handlers to prevent conflicts
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        logging.basicConfig(
            filename="{}/data.log".format(log_file_dir),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
            force=True,  # Force reconfiguration
        )

    def generate_curriculum_data(self, function_list, xstr1):
        """
        Generate curriculum data for uniform (unary) functions.
        Uses parent's method but with uniform-specific logic.
        """
        curriculum_function_list = []
        output_list = []
        curriculum_documents = []
        xstr1_tokens = np.array([self.token_idx[c] for c in xstr1])

        # Use parent's method for getting function lists
        function_lists = super().get_function_list_for_curriculum_fixed_prompt(
            function_list
        )

        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            # Apply the uniform function composition
            outputs = apply_function_composition_uniform(
                function_list_copy, self.function_dict, xstr1
            )

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

        for i in range(len(output_list)):
            task_indices = self.get_task_tokens(curriculum_function_list[i])
            output_tokens = self.get_output_tokens(output_list[i])
            curriculum_documents.append(
                np.concatenate(
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
            )

        return curriculum_documents

    def generate_curriculum_data_variable_prompt(self, function_list, xstr1):
        """
        Generate curriculum data for variable prompt length.
        Uses parent's method but with uniform-specific logic.
        """
        curriculum_function_list = []
        output_list = []
        curriculum_documents = []
        xstr1_tokens = np.array([self.token_idx[c] for c in xstr1])

        # Use parent's method for getting function lists
        function_lists = super().get_function_list_for_curriculum_variable_prompt(
            function_list
        )

        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            # Apply the uniform function composition
            outputs = apply_function_composition_uniform(
                function_list_copy, self.function_dict, xstr1
            )
            output_list.append(outputs)

        for i in range(len(output_list)):
            task_indices = self.get_task_tokens(curriculum_function_list[i])
            output_tokens = self.get_output_tokens(output_list[i])
            curriculum_documents.append(
                np.concatenate(
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
            )

        return curriculum_documents

    def generate_different_prompts_data(self, function_list):
        """
        Generate different prompt data for uniform (unary) functions.
        Only uses one input string instead of two.
        """
        xstr1, xstr1_tokens = self.sample_string()
        task_indices = self.get_task_tokens(function_list)

        # Apply uniform function composition (unary functions only)
        outputs = apply_function_composition_uniform(
            function_list, self.function_dict, xstr1
        )

        pad_length = self.seq_len
        for i in range(len(outputs)):
            if self.cfg.prompt_length == "fixed":
                if len(outputs[i]) < pad_length:
                    outputs[i] = outputs[i] + self.space_token * (
                        pad_length - len(outputs[i])
                    )
                else:
                    outputs[i] = outputs[i][:pad_length]

        output_tokens = self.get_output_tokens(outputs)

        # Direct document: task + input + final output
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

        # Step-by-step document: task + input + all intermediate outputs
        step_by_step_document = np.concatenate(
            [self.start_idx, task_indices, self.sep_idx, xstr1_tokens]
        )

        for i in range(len(outputs)):
            step_by_step_document = np.concatenate(
                [step_by_step_document, self.sep_idx, output_tokens[i]]
            )

        step_by_step_document = np.concatenate([step_by_step_document, self.end_idx])

        # Generate curriculum documents
        if self.cfg.prompt_length == "fixed":
            curriculum_documents = self.generate_curriculum_data(function_list, xstr1)
        else:
            curriculum_documents = self.generate_curriculum_data_variable_prompt(
                function_list, xstr1
            )

        return direct_document, step_by_step_document, curriculum_documents
