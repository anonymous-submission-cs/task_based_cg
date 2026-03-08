import functools
import itertools
import random
import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations

from src.data_generation.init import read_config
from src.data_generation.utils import *

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"
TRAIN_TEST_RATIO = 0.8
import json


class BaseFunction:
    @staticmethod
    def identity(xstr):
        # max length of the string is 12
        return xstr

    @staticmethod
    def map(xstr, offset=1, n_alphabets=26):
        """
        Maps a string to a new string by replacing each character with its corresponding character in the alphabet.
        """
        # make sure charaacter is in a-chr(n_alphabets)
        return "".join(
            chr((ord(c) - ord("a") + offset) % n_alphabets + ord("a")) for c in xstr
        )

    @staticmethod
    def sort(xstr):
        """
        Sorts the characters in a string.
        """
        # max length of the string is 12
        return "".join(sorted(xstr))

    @staticmethod
    def aggregate(xstr):
        # returns the sum of the ASCII values of the characters in the string and converts it to a string by modulo 256
        # makes sure the result is a string in a-z
        return "".join(chr((sum(ord(c) for c in xstr) % 26) + ord("a")))

    @staticmethod
    def join(xstr1, xstr2):
        """
        Joins two strings together.
        """
        # max length of the string is 12
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """
        Returns the intersection of two strings.
        """
        # maintain the order of the first and second string
        # keeps the first string and adds the characters from the second string that are not in the first string

        ans = []
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def max(xstr1):
        """
        Finds the letter with the maximum occurrences in the string.
        If multiple characters have the same occurrence, chooses the alphabetically smallest one.
        """
        if len(xstr1) == 0:
            return ""
        from collections import Counter

        # Count occurrences of each character
        char_count = Counter(xstr1)

        # Find the maximum occurrence count
        max_count = max(char_count.values())

        # Filter characters with the maximum count and return the smallest alphabetically
        max_chars = [char for char, count in char_count.items() if count == max_count]
        return min(max_chars)

    def filter(xstr, filter_func):
        """
        Filters a string based on a filter function.
        """
        # max length of the string is 12
        return "".join(filter(filter_func, xstr))

class Diverse_v2_Function:
    @staticmethod
    def identity(xstr):
        # max length of the string is 12
        return xstr

    def map(xstr, offset=1, n_alphabets=26):
        mapping_json_file = f"{ROOT_DIR}/data/jsons/mappings.json"
        with open(mapping_json_file, "r") as f:
            mapping = json.load(f)
        return "".join(mapping["map1"][c] for c in xstr)

    @staticmethod
    def sort(xstr):
        """
        Sorts the characters in a string.
        """
        # max length of the string is 12
        return "".join(sorted(xstr))

    @staticmethod
    def join(xstr1, xstr2):
        """
        Joins two strings together.
        """
        # max length of the string is 12
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """
        Returns the intersection of two strings.
        """
        # maintain the order of the first and second string
        # keeps the first string and adds the characters from the second string that are not in the first string

        ans = []
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def max(xstr1):
        """
        Finds the letter with the maximum occurrences in the string.
        If multiple characters have the same occurrence, chooses the alphabetically smallest one.
        """
        # implement reverse sort
        return "".join(sorted(xstr1, reverse=True))

    def filter(xstr, filter_func):
        """
        Filters a string based on a filter function.
        """
        # filter consonants
        filter_func = lambda x: x not in "aeiou"
        return "".join(filter(filter_func, xstr))


class MapRandom:
    @staticmethod
    def map_random(seed, n_alphabets=26):
        """
        Returns a function that maps a string using a random mapping generated with the given seed.
        Also provides a way to get the mapping dictionary for inspection/storage.
        """
        np.random.seed(seed)
        random.seed(seed)
        # mapping is without replacement
        # sample n_alphabets characters from a-z
        sampled_characters = random.sample(list(string.ascii_lowercase), n_alphabets)
        mapping = {chr(i + ord("a")): sampled_characters[i] for i in range(n_alphabets)}

        def map_func(xstr):
            return "".join(mapping[c] for c in xstr)

        map_func.mapping = mapping  # Attach mapping dict for external access
        
        return map_func


DIVERSE_FUNCTIONS = {
    "sort": BaseFunction.sort,
    "join": BaseFunction.join,
    "filter": BaseFunction.filter,
    "map": BaseFunction.map,
    "union": BaseFunction.union,
    "max": BaseFunction.max,
    "identity": BaseFunction.identity,
}

DIVERSE_2_FUNCTIONS = {
    "sort": Diverse_v2_Function.sort,
    "join": Diverse_v2_Function.join,
    "filter": Diverse_v2_Function.filter,
    "map": Diverse_v2_Function.map,
    "union": Diverse_v2_Function.union,
    "max": Diverse_v2_Function.max,
    "identity": Diverse_v2_Function.identity,
}


class CreateFunctions:
    def __init__(self, cfg):
        self.n_alphabets = cfg.n_alphabets
        self.seq_len = cfg.seq_len
        self.function_properties = cfg.function
        self.n_functions = cfg.function.n_functions
        self.seed = cfg.seed
        self.function_type = cfg.function.type
        # create a function dictionary with the function name as the key and the function as the value
        # include all possible functions and function selection/combination strategies will be implemented in the create_function
        if cfg.function.type == "diverse":
            self.function_dict = DIVERSE_FUNCTIONS
            self.mappings = None
        elif cfg.function.type == "uniform":
            # create functon dict with 10 random maps and function names as map1, map2, ..., map10, pass 1,2, 10 as seed
            self.function_dict = {
                f"map{i}": MapRandom.map_random(seed=i)
                for i in range(1, self.n_functions + 1)
            }
            # add identity function to the function dict
            self.function_dict["identity"] = BaseFunction.identity
            # store 1 to 10 mappings in a list (just the mapping dicts)
            self.mappings = {
                f"map{i}": MapRandom.map_random(seed=i).mapping
                for i in range(1, self.n_functions + 1)
            }
        elif cfg.function.type == "diverse2":
            self.function_dict = DIVERSE_2_FUNCTIONS

        self.cfg = cfg
        self.K = int(self.function_properties.split.strategy.split("_")[1])
        self.task_max_length = self.cfg.task_max_length
        self.function_names = list(self.function_dict.keys())
        self.strategy = self.function_properties.split.strategy
        split_list = self.function_properties.split.strategy.split("_")
        if len(split_list) == 4:
            self.multiple_relative_order = True
            self.num_relative_order = int(split_list[2])
            self.equivalence_class_leakage = int(split_list[3])
        if len(split_list) == 3:
            self.multiple_relative_order = True
            self.num_relative_order = int(split_list[2])
        elif len(split_list) == 2:
            self.multiple_relative_order = False
            self.num_relative_order = 1

    def create_functions(self) -> Tuple[List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Create function combinations based on the specified strategy.

        Returns:
            Tuple of (all_functions, combination_ids)
        """

        if self.strategy == "random":
            unique_function_combinations_permutations = (
                self._create_random_combinations_permutations()
            )
        elif self.strategy.startswith("combination") or self.strategy.startswith(
            "permutation"
        ):
            unique_function_combinations_permutations = (
                self._create_systematic_combinations_permutations()
            )
        elif self.strategy.startswith("equivalence"):
            unique_function_combinations_permutations = (
                self._create_systematic_combinations_permutations()
            )
        elif self.strategy.startswith("uniequivalence"):
            unique_function_combinations_permutations = (
                self._create_systematic_combinations_permutations()
            )
        elif self.strategy.startswith("custom"):
            unique_function_combinations_permutations = (
                self._create_systematic_combinations_permutations()
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Create combination IDs mapping
        combination_ids = {
            tuple(combo): i
            for i, combo in enumerate(unique_function_combinations_permutations)
        }
        # Convert to list format for consistency
        all_functions = [
            list(combo) for combo in unique_function_combinations_permutations
        ]

        self.combination_ids = combination_ids
        return all_functions, combination_ids

    def _generate_multiple_relative_order(self) -> List[str]:
        """Generate the relative order of the functions."""
        all_relative_order = self._generate_all_relative_order()
        # select the first self.num_relative_order relative order permutations
        if self.strategy.startswith("permutationrandom"):
            # set random seed
            np.random.seed(self.seed)
            # shuffle the relative order
            if self.strategy == "permutationrandom_6_1":
                np.random.shuffle(all_relative_order)
                # write to the file
                with open(
                    f"{ROOT_DIR}/data/jsons/relative_order_{self.strategy}_{self.function_type}.json",
                    "w",
                ) as f:
                    json.dump(all_relative_order, f)
            else:
                # read from the file
                with open(
                    f"{ROOT_DIR}/data/jsons/relative_order_permutationrandom_6_1_{self.function_type}.json",
                    "r",
                ) as f:
                    all_relative_order = json.load(f)
            return all_relative_order[: self.num_relative_order]
        elif self.strategy.startswith("permutationrotated"):
            with open(
                f"{ROOT_DIR}/data/jsons/relative_order_permutationrandom_6_1_{self.function_type}.json",
                "r",
            ) as f:
                all_relative_order = json.load(f)
            # pick first six
            first_six = all_relative_order[:6]
            # create rotations
            rotations = first_six.copy()
            for j in range(len(all_relative_order[0]) - 1):
                for i in range(len(first_six)):

                    new_rotation = first_six[i][j + 1 :] + first_six[i][: j + 1]
                    rotations.append(new_rotation)
            return rotations[: self.num_relative_order]
        elif self.strategy.startswith("permutationcoveragev1"):
            function_names_without_identity = [
                function for function in self.function_names if function != "identity"
            ]
            num_orderings = int(self.strategy.split("_")[2])
            print("num_orderings", num_orderings)
            with open(
                f"{ROOT_DIR}/data/csvs/absolute_coverage_permutations_30.csv", "r"
            ) as f:
                relative_order_index_list = pd.read_csv(f)
            relative_order_index_list = relative_order_index_list.values.tolist()
            relative_order_index_list = relative_order_index_list[:num_orderings]
            function_lists = []
            for i in range(len(relative_order_index_list)):
                # take previous relative order and add a new function to the end
                function_list = [
                    function_names_without_identity[i - 1]
                    for i in relative_order_index_list[i]
                ]
                function_lists.append(function_list)
            return function_lists
        elif (
            self.strategy.startswith("permutation")
            or self.strategy.startswith("equivalence")
            or self.strategy.startswith("uniequivalence")
        ):
            return all_relative_order[: self.num_relative_order]
        else:
            print("Unknown strategy: ", self.strategy)
            raise ValueError("Unknown strategy: ", self.strategy)

    def _generate_all_relative_order(self) -> List[str]:
        """Generate all possible relative order permutations of the functions."""
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        all_relative_order = list(
            itertools.permutations(
                function_names_without_identity, len(function_names_without_identity)
            )
        )
        return all_relative_order

    def _get_relative_order(self) -> List[str]:
        """Get the relative order of the functions."""
        if self.multiple_relative_order:
            return self._generate_multiple_relative_order()
        else:
            return [self._generate_all_relative_order()[0]]

    def _create_random_combinations_permutations(self) -> List[List[str]]:
        """Create random permutations of function names."""
        if self.n_functions < len(self.function_names):
            self.function_names = self.function_names[: self.n_functions]
        return list(itertools.permutations(self.function_names, self.n_functions))

    def _create_systematic_combinations_permutations(self) -> List[List[str]]:
        """Create systematic combinations of size K with identity functions."""
        if self.n_functions < len(self.function_names):
            self.function_names = self.function_names[: self.n_functions]

        # Create combinations with identity functions
        if self.cfg.prompt_length == "variable":
            return (
                self._create_systematic_combinations_permutations_variable_prompt_length()
            )
        else:
            return (
                self._create_systematic_combinations_permutations_fixed_prompt_length()
            )

    def _create_systematic_combinations_permutations_fixed_prompt_length(
        self,
    ) -> List[List[str]]:
        """Create fixed-length systematic combinations with identity functions."""
        all_combinations = []
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        for size in range(1, len(function_names_without_identity) + 1):
            combinations = list(
                itertools.combinations(function_names_without_identity, size)
            )

            # Pad each combination with identity functions
            for combo in combinations:
                padded_combo = list(combo) + ["identity"] * (
                    self.cfg.task_max_length - len(combo)
                )
                # Generate all permutations of the padded combination
                # In this way, we can get a random train/test split for each combination where different orderings of the same combination
                # are randomly sampled with no explicit bias.
                all_combinations.extend(multiset_permutations(padded_combo))

        return all_combinations

    def _create_systematic_combinations_permutations_variable_prompt_length(
        self,
    ) -> List[List[str]]:
        """Create variable-length systematic combinations."""
        all_combinations = []
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        for i in range(1, len(function_names_without_identity) + 1):

            all_combinations.extend(
                itertools.permutations(function_names_without_identity, i)
            )
        return all_combinations

    def get_train_functions(
        self,
    ) -> Tuple[List[List[str]], List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Returns training and test function splits based on the configured strategy.

        Returns:
            Tuple of (train_functions, test_functions, functions_info)
        """
        all_functions, functions_info = self.create_functions()

        # Split functions based on strategy
        if self.strategy == "random":
            train_functions, test_functions = self._split_random(all_functions)
        elif self.strategy.startswith("combination"):
            # This chooses K-sized combinations but splits them into train and test sets randomly
            train_functions, test_functions = self._split_random_permutations(
                all_functions
            )
        elif self.strategy.startswith("permutation"):
            # This chooses K-sized combinations but splits them into train and test sets systematically based on the relative order of the functions
            train_functions, test_functions = self._split_systematic_permutations(
                all_functions
            )
        elif self.strategy.startswith("equivalence"):
            train_functions, test_functions = self._split_equivalence(all_functions)
        elif self.strategy.startswith("uniequivalence"):
            train_functions, test_functions = self._split_unique_equivalence(
                all_functions
            )
        elif self.strategy.startswith("custom"):
            # select functions that only have sort max filter and map
            selected_functions = []
            test_functions = []
            selected = True
            for fn_list in all_functions:
                selected = True
                for fn in fn_list:
                    if fn not in ["sort", "max", "filter", "map"] or len(fn_list) - fn_list.count("identity") != 3:
                        selected = False
                        break
                if selected:
                    selected_functions.append(fn_list)
                    
                        
            train_functions = [["sort", "max", "filter"], ["filter", "sort", "max"], ["max", "sort", "filter"], ["sort", "filter", "max"]]
            test_functions = [fn for fn in selected_functions if fn not in train_functions]
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")

        self._print_split_info(train_functions, test_functions)
        return train_functions, test_functions, functions_info

    def _split_random_permutations(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if self.cfg.prompt_length == "fixed":
            return self._split_random_permutations_fixed_prompt_length(all_functions)
        else:
            return self._split_random_permutations_variable_prompt_length(all_functions)

    def _split_systematic_permutations(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if self.cfg.prompt_length == "fixed":
            return self._split_systematic_permutations_fixed_prompt_length(
                all_functions
            )
        else:
            return self._split_systematic_permutations_variable_prompt_length(
                all_functions
            )

    def _split_equivalence(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if self.cfg.prompt_length == "fixed":
            return self._split_equivalence_fixed_prompt_length(all_functions)
        else:
            return self._split_equivalence_variable_prompt_length(all_functions)

    def _split_unique_equivalence(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if self.cfg.prompt_length == "fixed":
            return self._split_unique_equivalence_fixed_prompt_length(all_functions)
        else:
            return None, None

    def _split_random_permutations_fixed_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        K_max = 6
        if self.K <= K_max:
            num_i_functions = [
                fn_list
                for fn_list in all_functions
                if len(fn_list) - fn_list.count("identity") == self.K
            ]
        else:
            num_i_functions = [
                fn_list
                for fn_list in all_functions
                if len(fn_list) - fn_list.count("identity") == K_max
            ]

        train_functions = random.sample(
            num_i_functions, int(len(num_i_functions) * TRAIN_TEST_RATIO)
        )
        test_functions = [fn for fn in num_i_functions if fn not in train_functions]
        print("Train functions: ", len(train_functions))
        print("Test functions: ", len(test_functions))

        return train_functions, test_functions

    def _split_random_permutations_variable_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        # train_functions include all the functions which have number of non-identity functions less than or equal to num_train_i
        # test_functions include all the functions which have number of non-identity functions greater than or equal to num_test_i
        num_i_functions = [
            fn_list
            for fn_list in all_functions
            if len(fn_list) - fn_list.count("identity") == self.K
        ]
        if self.K == 1:
            train_functions = num_i_functions
            test_functions = num_i_functions
        else:
            train_functions = random.sample(
                num_i_functions, int(len(num_i_functions) * TRAIN_TEST_RATIO)
            )
            test_functions = [fn for fn in num_i_functions if fn not in train_functions]
        print("Train functions: ", len(train_functions))
        print("Test functions: ", len(test_functions))
        return train_functions, test_functions

    def _split_systematic_permutations_fixed_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        relative_order_list = self._get_relative_order()
        print("relative_order_list", relative_order_list)

        if not self.multiple_relative_order:
            print("relative_order", relative_order_list)
        else:
            print("Number of relative order permutations: ", len(relative_order_list))
        K_max = 6
        if self.K <= K_max:
            i_functions = [
                fn_list
                for fn_list in all_functions
                if len(fn_list) - fn_list.count("identity") == self.K
            ]
        else:
            i_functions = [
                fn_list
                for fn_list in all_functions
                if len(fn_list) - fn_list.count("identity") == K_max
            ]
        selected_combinations = []
        unseen_combinations = []
        if self.K != 1:
            for i in range(len(i_functions)):
                combo = i_functions[i]
                # remove the identity functions from the combo
                combo = [function for function in combo if function != "identity"]
                given_order_combo = []
                for fn in combo:
                    given_order_combo.append(fn)
                select_combo = False
                for relative_order in relative_order_list:
                    # filter functions in the combo
                    filtered_combo = [fn for fn in relative_order if fn in combo]
                    if filtered_combo == given_order_combo:
                        selected_combinations.append(i_functions[i])
                        select_combo = True
                        break
                if not select_combo:
                    unseen_combinations.append(i_functions[i])
        else:
            for i in range(len(i_functions)):
                combo = i_functions[i]
                # find index of the non-identity function
                non_identity_index = combo.index(
                    next(function for function in combo if function != "identity")
                )
                non_identity_function = combo[non_identity_index]

                # check if the function's position matches its position in any of the relative orders
                select_combo = False
                for relative_order in relative_order_list:
                    # find index of the same function in the relative order
                    same_function_index = relative_order.index(non_identity_function)
                    # find index of the same function in the combo
                    same_function_index_in_combo = combo.index(
                        relative_order[same_function_index]
                    )
                    # if the index of the same function in the combo is the same as the index of the same function in the relative order, then it is a selected combination
                    if same_function_index_in_combo == same_function_index:
                        selected_combinations.append(i_functions[i])
                        select_combo = True
                        break

                if not select_combo:
                    unseen_combinations.append(i_functions[i])
        if len(selected_combinations) == 0 or len(unseen_combinations) == 0:
            print("No selected or unseen combinations found for size {}".format(self.K))
            unseen_combinations = i_functions
            selected_combinations = i_functions
        print("Selected combinations: ", len(selected_combinations))
        print("Unseen combinations: ", len(unseen_combinations))
        return selected_combinations, unseen_combinations

    def _learn_equivalence_class_map(
        self, function_lists: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        # learn equivalence class map from function lists
        equivalence_class_map = {}
        unique_function_lists = []
        for i, function_list in enumerate(function_lists):
            # get the function list without identity
            function_list = [
                function for function in function_list if function != "identity"
            ]
            # get the equivalence class of the function list
            equivalence_class = tuple(function_list)
            if equivalence_class not in equivalence_class_map:
                equivalence_class_map[equivalence_class] = []
                unique_function_lists.append(equivalence_class)
            equivalence_class_map[equivalence_class].append(i)
        return equivalence_class_map, unique_function_lists

    def _split_equivalence_fixed_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:

        train_functions, test_functions = (
            self._split_systematic_permutations_fixed_prompt_length(all_functions)
        )
        train_equivalence_class_map, train_unique_function_lists = (
            self._learn_equivalence_class_map(train_functions)
        )
        test_equivalence_class_map, test_unique_function_lists = (
            self._learn_equivalence_class_map(test_functions)
        )
        all_train_indices = [i for i in range(len(train_functions))]
        all_test_indices = [i for i in range(len(test_functions))]
        # learn equivalence class map from function lists
        if self.equivalence_class_leakage == 0:
            return train_functions, test_functions
        else:
            final_train_functions = []
            final_test_functions = []
            sampled_test_indices = []
            sampled_train_indices = []
            for i in range(self.equivalence_class_leakage):
                # get first relative order
                candidate_train_equivalence_class = train_unique_function_lists[i]
                candidate_test_equivalence_class = test_unique_function_lists[i]
                # find equivalence class that is not in train_equivalence_class_map
                # find indices of train functions that satisfy the equivalence class
                train_functions_indices = train_equivalence_class_map[
                    candidate_train_equivalence_class
                ]
                print(train_functions_indices)
                # find indices of test functions that satisfy the equivalence class
                test_functions_indices = test_equivalence_class_map[
                    candidate_test_equivalence_class
                ]
                print(test_functions_indices)
                # find indices of test functions that are not in train functions
                # randomly sample 0.5 * len(test_functions_indices) indices from test_functions_indices
                test_functions_indices_sampled = random.sample(
                    test_functions_indices, int(0.5 * len(test_functions_indices))
                )
                train_functions_indices_sampled = random.sample(
                    test_functions_indices, int(0.5 * len(test_functions_indices))
                )
                sampled_test_indices.extend(test_functions_indices_sampled)
                sampled_train_indices.extend(train_functions_indices_sampled)
            remaining_test_functions_indices = [
                i for i in all_test_indices if i not in sampled_test_indices
            ]
            remaining_train_functions_indices = [
                i for i in all_train_indices if i not in sampled_train_indices
            ]
            print(len(test_functions_indices_sampled))
            print(len(train_functions_indices_sampled))
            print(len(remaining_train_functions_indices))
            print(len(remaining_test_functions_indices))
            # have train functions as remaining train functions and test functions sampled
            for i in range(len(sampled_test_indices)):
                final_train_functions.append(test_functions[sampled_test_indices[i]])
            for i in range(len(sampled_train_indices)):
                final_test_functions.append(train_functions[sampled_train_indices[i]])
            for i in range(len(remaining_train_functions_indices)):
                final_train_functions.append(
                    train_functions[remaining_train_functions_indices[i]]
                )
            for i in range(len(remaining_test_functions_indices)):
                final_test_functions.append(
                    test_functions[remaining_test_functions_indices[i]]
                )
                # add test functions sampled to final_test_functions

            return final_train_functions, final_test_functions

    def _split_unique_equivalence_fixed_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:

        train_functions, test_functions = (
            self._split_systematic_permutations_fixed_prompt_length(all_functions)
        )
        all_functions = train_functions + test_functions
        equivalence_class_map, unique_function_lists = (
            self._learn_equivalence_class_map(all_functions)
        )
        all_indices = [i for i in range(len(all_functions))]

        new_train_function_indices = []
        new_test_function_indices = []
        new_test_functions = []
        for unique_function_list in unique_function_lists:
            equivalence_class_functions_indices = equivalence_class_map[
                unique_function_list
            ]
            # ransomly sample 1 index from train_functions_indices
            train_function_index = random.sample(
                equivalence_class_functions_indices, self.equivalence_class_leakage + 1
            )
            new_train_function_indices.extend(train_function_index)
            remaining_test_functions_indices = [
                i
                for i in equivalence_class_functions_indices
                if i not in train_function_index
            ]
            new_test_function_indices.extend(remaining_test_functions_indices)
        print(len(new_train_function_indices))
        print(len(new_test_function_indices))
        # have unique indices for train and test functions
        new_train_function_indices = list(set(new_train_function_indices))
        new_test_function_indices = list(set(new_test_function_indices))
        new_train_functions = [all_functions[i] for i in new_train_function_indices]
        new_test_functions = [all_functions[i] for i in new_test_function_indices]
        return new_train_functions, new_test_functions

    def _split_systematic_permutations_variable_prompt_length(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        relative_order_list = self._get_relative_order()

        K_max = 6
        if self.K <= K_max:
            i_functions = [
                fn_list for fn_list in all_functions if len(fn_list) == self.K
            ]
        else:
            i_functions = [
                fn_list for fn_list in all_functions if len(fn_list) == K_max
            ]

        selected_combinations = []
        unseen_combinations = []
        for i in range(len(i_functions)):
            combo = i_functions[i]
            select_combo = False
            for relative_order in relative_order_list:
                selected_relative_order = []
                for j in relative_order:
                    if j in combo:
                        selected_relative_order.append(j)
                if selected_relative_order == combo:
                    selected_combinations.append(combo)
                    select_combo = True
                    break

            if not select_combo:
                unseen_combinations.append(combo)
        if len(selected_combinations) == 0 or len(unseen_combinations) == 0:
            print("No selected or unseen combinations found for size {}".format(self.K))
            unseen_combinations = i_functions
            selected_combinations = i_functions
        return selected_combinations, unseen_combinations

    def _split_random(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split functions randomly with 90/10 train/test ratio."""
        import random

        shuffled_functions = all_functions.copy()
        random.shuffle(shuffled_functions)

        split_index = int(len(shuffled_functions) * TRAIN_TEST_RATIO)
        train_functions = shuffled_functions[:split_index]
        test_functions = shuffled_functions[split_index:]

        return train_functions, test_functions

    def _print_split_info(
        self, train_functions: List[List[str]], test_functions: List[List[str]]
    ) -> None:
        """Print information about the train/test split."""
        print(f"Total number of training functions: {len(train_functions)}")
        print(f"Total number of test functions: {len(test_functions)}")


def apply_function_composition(
    n_alphabets, function_list, function_dict, xstr1, xstr2, filter_func, offset
):
    """
    Applies a function to a string.
    """
    outputs = []
    # apply the function to the string
    for function in function_list:
        if function == "join" or function == "union":
            # if the function is join or intersect, apply it to both strings
            xstr1 = function_dict[function](xstr1, xstr2)
        elif function == "filter":
            xstr1 = function_dict[function](xstr1, filter_func)
        elif function == "map":
            xstr1 = function_dict[function](xstr1, offset, n_alphabets)
        else:
            xstr1 = function_dict[function](xstr1)

        outputs.append(xstr1)
    return outputs


def apply_function_composition_uniform(function_list, function_dict, xstr1):
    """
    Applies a function to a string.
    """
    outputs = []
    # apply the function to the string
    for function in function_list:
        xstr1 = function_dict[function](xstr1)
        outputs.append(xstr1)
    return outputs


# test the functions
def main():
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    # read the config file
    cfg = read_config(cfg_path)
    cfg.prompt_length = "fixed"
    cfg.function.split.strategy = "custom_3"
    cfg.function.type = "diverse"
    cfg.task_max_length = 3
    # create the functions
    create_functions = CreateFunctions(cfg)
    all_functions, combination_ids = create_functions.create_functions()
    # function_lists = random.sample(all_functions, 20)
    train_functions, test_functions, functions_info = (
        create_functions.get_train_functions()
    )
    # print first 5 train functions
    for i in range(len(train_functions)):
        print(train_functions[i])
    # print first 5 test functions
    for i in range(5):
        print(test_functions[i])


    print("train_functions", len(train_functions))
    print("test_functions", len(test_functions))


if __name__ == "__main__":
    main()
