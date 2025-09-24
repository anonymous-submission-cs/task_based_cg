# plotting_utils.py
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"


class DataLoader:
    """Handles loading evaluation data."""

    def __init__(self, root_dir: str = ROOT_DIR):
        self.root_dir = root_dir

    def load_data(
        self,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        trains: List[str],
        evals: List[str],
        seeds: List[int],
        nh_nl: str,
        nalph: str,
        plot_direct_sbs: bool = False,
        step_wise_mode: str = "individual",
        function_type: str = "uniform",
        task_max_length_flag: str = "fixed",
        self_eval: bool = False,
    ) -> Dict:
        """Load all evaluation data."""
        data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            )
        )
        print(modes, pos_types, lengths, trains, evals, seeds)
        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    for train in trains:
                        for eval_split in evals:
                            for seed in seeds:
                                if self_eval and train != eval_split:
                                    continue
                                nalph_modified = self._get_modified_nalph(
                                    nalph, train, task_max_length_flag
                                )
                                path = self._build_path(
                                    mode,
                                    pos,
                                    length,
                                    train,
                                    eval_split,
                                    seed,
                                    nh_nl,
                                    nalph_modified,
                                    function_type,
                                )

                                if os.path.exists(path):
                                    print(path)
                                    acc_data = self._load_pickle_file(path)

                                    if acc_data:
                                        print(
                                            mode, pos, length, train, eval_split, seed
                                        )
                                        processed_data = self._process_acc_data(
                                            acc_data, plot_direct_sbs, step_wise_mode
                                        )
                                        if processed_data:
                                            data[mode][pos][length][train][eval_split][
                                                f"seed_{seed}"
                                            ] = processed_data
                                else:
                                    if eval_split == train:
                                        print(f"Path {path} does not exist")
        return data

    def _get_modified_nalph(
        self, nalph: str, train: str, task_max_length_flag: str
    ) -> str:
        if task_max_length_flag == "variable":
            task_max_len = train.split("_")[-1]
            return nalph.replace("taskmaxlen_6", f"taskmaxlen_{task_max_len}")
        return nalph

    def _build_path(
        self,
        mode: str,
        pos: str,
        length: str,
        train: str,
        eval_split: str,
        seed: int,
        nh_nl: str,
        nalph: str,
        function_type: str,
    ) -> str:
        return (
            f"{self.root_dir}/models/eval/{function_type}/{nalph}/{mode}/{length}/"
            f"model_{train}/eval_{eval_split}/{pos}/{nh_nl}/seed_{seed}/accs.pkl"
        )

    def _load_pickle_file(self, path: str) -> Optional[List]:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _process_acc_data(
        self, acc_data: List, plot_direct_sbs: bool, step_wise_mode: str
    ) -> Optional[Dict]:
        result = {
            "train": None,
            "test": None,
            "direct_train": None,
            "direct_test": None,
            "module_wise_train": {},
            "module_wise_test": {},
            "step_wise_train": {},
            "step_wise_test": {},
            "train_ood_ratio": None,
            "test_ood_ratio": None,
        }

        for checkpoint, metrics in acc_data:
            for key, value in metrics.items():
                acc = value["total"]["acc"] if isinstance(value, dict) else value[0]
                ood_ratio = (
                    value["total"]["ood"] if isinstance(value, dict) else value[1]
                )

                if key == "train":
                    result["train"] = acc
                    result["train_ood_ratio"] = ood_ratio
                elif key == "test":
                    result["test"] = acc
                    result["test_ood_ratio"] = ood_ratio
                    print(acc)

                if plot_direct_sbs and "direct" in value:
                    if key == "train":
                        result["direct_train"] = value["direct"]["total"]["acc"]
                    elif key == "test":
                        result["direct_test"] = value["direct"]["total"]["acc"]

                # Process module-wise data
                if "module_wise" in value:
                    for fn, fn_stats in value["module_wise"].items():
                        if key == "train":
                            result["module_wise_train"][fn] = (
                                sum(fn_stats["acc"]) / fn_stats["total"]
                            )
                        elif key == "test":
                            result["module_wise_test"][fn] = (
                                sum(fn_stats["acc"]) / fn_stats["total"]
                            )

                # Process step-wise data
                if "step_by_step" in value:
                    step_wise_data = value["step_by_step"][step_wise_mode]
                    for fn, fn_stats in step_wise_data.items():
                        if key == "train":
                            result["step_wise_train"][fn] = sum(fn_stats) / len(
                                fn_stats
                            )
                        elif key == "test":
                            result["step_wise_test"][fn] = sum(fn_stats) / len(fn_stats)

        return (
            result
            if result["train"] is not None and result["test"] is not None
            else None
        )


class DataProcessor:
    """Processes loaded data for plotting."""

    @staticmethod
    def extract_points(
        data: Dict,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        trains: List[str],
        evals: List[str],
        seeds: List[int],
        self_eval: bool = False,
        plot_direct_sbs: bool = False,
    ) -> Tuple:
        """Extract plot points from data."""
        # Initialize result dictionaries
        train_mean_pts = {
            m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
        }
        test_mean_pts = {
            m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
        }
        train_std_pts = {
            m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
        }
        test_std_pts = {
            m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
        }

        direct_train_mean_pts = direct_test_mean_pts = None
        direct_train_std_pts = direct_test_std_pts = None

        if plot_direct_sbs:
            direct_train_mean_pts = {
                m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
            }
            direct_test_mean_pts = {
                m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
            }
            direct_train_std_pts = {
                m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
            }
            direct_test_std_pts = {
                m: {p: {l: [] for l in lengths} for p in pos_types} for m in modes
            }

        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    for train in trains:
                        eval_list = [train] if self_eval else evals
                        for eval_split in eval_list:
                            try:
                                d_train = [
                                    data[mode][pos][length][train][eval_split][
                                        f"seed_{seed}" if f"seed_{seed}" in data[mode][pos][length][train][eval_split] else "seed_0"
                                    ]["train"]
                                    for seed in seeds
                                ]
                                d_test = [
                                    data[mode][pos][length][train][eval_split][
                                        f"seed_{seed}" if f"seed_{seed}" in data[mode][pos][length][train][eval_split] else "seed_0"
                                    ]["test"]
                                    for seed in seeds
                                ]

                                train_mean_pts[mode][pos][length].append(
                                    np.mean(d_train)
                                )
                                test_mean_pts[mode][pos][length].append(np.mean(d_test))
                                train_std_pts[mode][pos][length].append(np.std(d_train))
                                test_std_pts[mode][pos][length].append(np.std(d_test))

                                if plot_direct_sbs:
                                    d_direct_train = [
                                        data[mode][pos][length][train][eval_split][
                                            f"seed_{seed}" if f"seed_{seed}" in data[mode][pos][length][train][eval_split] else "seed_0"
                                        ]["direct_train"]
                                        for seed in seeds
                                    ]
                                    d_direct_test = [
                                        data[mode][pos][length][train][eval_split][
                                            f"seed_{seed}" if f"seed_{seed}" in data[mode][pos][length][train][eval_split] else "seed_0"
                                        ]["direct_test"]
                                        for seed in seeds
                                    ]

                                    direct_train_mean_pts[mode][pos][length].append(
                                        np.mean(d_direct_train)
                                    )
                                    direct_test_mean_pts[mode][pos][length].append(
                                        np.mean(d_direct_test)
                                    )
                                    direct_train_std_pts[mode][pos][length].append(
                                        np.std(d_direct_train)
                                    )
                                    direct_test_std_pts[mode][pos][length].append(
                                        np.std(d_direct_test)
                                    )

                            except KeyError:
                                continue

        return (
            train_mean_pts,
            train_std_pts,
            test_mean_pts,
            test_std_pts,
            direct_train_mean_pts,
            direct_test_mean_pts,
            direct_train_std_pts,
            direct_test_std_pts,
        )

    @staticmethod
    def extract_step_wise_points(
        data: Dict,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        trains: List[str],
        evals: List[str],
    ) -> Tuple[Dict, Dict]:
        """Extract step-wise plot points from data."""
        mean_step_wise_train_pts = {
            m: {p: {l: {} for l in lengths} for p in pos_types} for m in modes
        }
        mean_step_wise_test_pts = {
            m: {p: {l: {} for l in lengths} for p in pos_types} for m in modes
        }
        std_step_wise_train_pts = {
            m: {p: {l: {} for l in lengths} for p in pos_types} for m in modes
        }
        std_step_wise_test_pts = {
            m: {p: {l: {} for l in lengths} for p in pos_types} for m in modes
        }

        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    for train in trains:
                        for eval_split in evals:
                            try:
                                d = data[mode][pos][length][train][eval_split]

                                for seed_key, seed_data in d.items():
                                    print(seed_data.keys())
                                    if (
                                        "step_wise_train" in seed_data
                                        and "step_wise_test" in seed_data
                                    ):
                                        d_train = seed_data["step_wise_train"]
                                        d_test = seed_data["step_wise_test"]

                                        mean_step_wise_train_pts[mode][pos][
                                            length
                                        ].append(np.mean(d_train))
                                        mean_step_wise_test_pts[mode][pos][
                                            length
                                        ].append(np.mean(d_test))
                                        std_step_wise_train_pts[mode][pos][
                                            length
                                        ].append(np.std(d_train))
                                        std_step_wise_test_pts[mode][pos][
                                            length
                                        ].append(np.std(d_test))

                            except KeyError:
                                continue

        return (
            mean_step_wise_train_pts,
            mean_step_wise_test_pts,
            std_step_wise_train_pts,
            std_step_wise_test_pts,
        )


class Plotter:
    """Creates plots from processed data."""

    def __init__(
        self, font_size: int = 30, figure_size: Tuple = (20, 10), dpi: int = 300
    ):
        self.font_size = font_size
        self.figure_size = figure_size
        self.dpi = dpi
        self.colors = {
            "direct_fixed": {"abs": "tab:blue", "rel_global": "tab:red"},
            "step_by_step_fixed": {"abs": "tab:green", "rel_global": "tab:orange"},
            "step_by_step_direct_fixed": {
                "abs": "tab:purple",
                "rel_global": "tab:brown",
            },
            "direct_variable": {"abs": "tab:cyan", "rel_global": "tab:pink"},
            "step_by_step_variable": {"abs": "tab:olive", "rel_global": "tab:gray"},
            "step_by_step_direct_variable": {
                "abs": "tab:gray",
                "rel_global": "tab:olive",
            },
        }
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        plt.rcParams.update(
            {
                "font.size": self.font_size,
                "legend.fontsize": self.font_size,
                "axes.labelsize": self.font_size,
                "axes.titlesize": self.font_size,
            }
        )

    def plot_accuracy_comparison(
        self,
        train_mean_pts: Dict,
        train_std_pts: Dict,
        test_mean_pts: Dict,
        test_std_pts: Dict,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        trains: List[str],
        evals: List[str],
        save_path: str,
        direct_train_mean_pts: Optional[Dict] = None,
        direct_test_mean_pts: Optional[Dict] = None,
        direct_train_std_pts: Optional[Dict] = None,
        direct_test_std_pts: Optional[Dict] = None,
        self_eval: bool = False,
        function_type: str = "uniform",
        plot_std: bool = False,
        plot_train: bool = False,
    ):
        """Create and save accuracy comparison plot."""

        plt.figure(figsize=self.figure_size)

        # Plot main data
        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    color = self.colors[f"{mode}_{length}"][pos]

                    if plot_train:
                        plt.plot(
                            train_mean_pts[mode][pos][length],
                            marker="o",
                            linestyle="-",
                            color=color,
                            markersize=self.font_size / 2,
                            linewidth=2,
                        )
                    plt.plot(
                        test_mean_pts[mode][pos][length],
                        marker="o",
                        linestyle="--",
                        color=color,
                        markersize=self.font_size / 2,
                        linewidth=2,
                    )

                    if plot_std:
                        if plot_train:
                            self._add_error_bands(
                                train_mean_pts[mode][pos][length],
                                train_std_pts[mode][pos][length],
                                color,
                            )

                        self._add_error_bands(
                            test_mean_pts[mode][pos][length],
                            test_std_pts[mode][pos][length],
                            color,
                        )

        # Plot direct SBS data if available
        if direct_train_mean_pts:
            for mode in modes:
                for pos in pos_types:
                    for length in lengths:
                        color = self.colors[f"{mode}_direct_{length}"][pos]
                        if plot_train:
                            plt.plot(
                                direct_train_mean_pts[mode][pos][length],
                                marker="o",
                                linestyle="-",
                                color=color,
                                markersize=self.font_size / 2,
                                linewidth=2,
                            )
                            if plot_std:
                                self._add_error_bands(
                                    direct_train_mean_pts[mode][pos][length],
                                    direct_train_std_pts[mode][pos][length],
                                    color,
                                )

                        plt.plot(
                            direct_test_mean_pts[mode][pos][length],
                            marker="o",
                            linestyle="--",
                            color=color,
                            markersize=self.font_size / 2,
                            linewidth=2,
                        )
                        if plot_std:
                            self._add_error_bands(
                                direct_test_mean_pts[mode][pos][length],
                                direct_test_std_pts[mode][pos][length],
                                color,
                            )

        self._setup_plot_formatting(trains, evals, self_eval, function_type)
        print(trains[0].startswith("equivalence"))
        self._create_legend(
            modes,
            pos_types,
            lengths,
            direct_train_mean_pts is not None,
            plot_train,
            trains[0].startswith("equivalence"),
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()
        print(f"Plot saved to: {save_path}")

    def _add_error_bands(self, mean_vals: List, std_vals: List, color: str):
        plt.fill_between(
            range(len(mean_vals)),
            np.array(mean_vals) - np.array(std_vals),
            np.array(mean_vals) + np.array(std_vals),
            color=color,
            alpha=0.3,
        )

    def _setup_plot_formatting(
        self, trains: List[str], evals: List[str], self_eval: bool, function_type: str
    ):
        if trains and "combination" in trains[0]:
            if len(evals) == 1:
                sizes = [int(t.split("_")[1]) for t in trains]
                plt.xticks(range(len(sizes)), sizes, fontsize=self.font_size)
            elif len(evals) > 1 and not self_eval:
                sizes = [int(e.split("_")[1]) for e in evals]
                plt.xticks(range(len(sizes)), sizes, fontsize=self.font_size)
                plt.xlabel("Evaluation Combination Size", fontsize=self.font_size)
            else:
                sizes = [int(t.split("_")[1]) for t in trains]
                plt.xticks(range(len(sizes)), sizes, fontsize=self.font_size)
                plt.xlabel("Combination Size (K)", fontsize=self.font_size)

        elif "permutation" in str(trains):
            num_permutations_num = [int(train.split("_")[2]) for train in trains]
            labels = num_permutations_num
            plt.xticks(range(len(labels)), labels, fontsize=self.font_size, rotation=45)
            plt.xlabel("Training Permutation Size", fontsize=self.font_size)
            # 45 degree rotation

        elif str(trains[0]).startswith("equivalence"):
            sizes = [int(t.split("_")[3]) for t in trains]
            train_perm_size = int(trains[0].split("_")[2])
            unique_test_perm_size = 720 - train_perm_size
            print(f"Unique test perm size: {unique_test_perm_size}")
            print(f"Sizes: {sizes}")
            predicted_accuracy = [
                round((size * 7) / (unique_test_perm_size * 7), 2) for size in sizes
            ]
            sizes_normalized = [
                int(round(size / unique_test_perm_size, 2) * 100) for size in sizes
            ]

            plt.plot(
                predicted_accuracy,
                marker="o",
                linestyle="-",
                color="black",
                label="Predicted Accuracy",
                markersize=self.font_size / 2,
            )

            sizes_normalized = [f"{size}%" for size in sizes_normalized]
            plt.xticks(
                range(len(sizes_normalized)), sizes_normalized, fontsize=self.font_size
            )
            plt.xlabel("Shared equivalence class size (%)", fontsize=self.font_size)

        elif str(trains[0]).startswith("uniequivalence"):
            sizes = [int(t.split("_")[3]) + 1 for t in trains]
            plt.xticks(range(len(sizes)), sizes, fontsize=self.font_size)
            plt.xlabel(
                "Number of equivalent task per class present in training (out of 7)",
                fontsize=self.font_size,
            )
            plt.ylabel("Accuracy on test set", fontsize=self.font_size)

        else:
            if len(trains) > 1:
                plt.xticks(
                    range(len(trains)), trains, rotation=90, fontsize=self.font_size
                )
                plt.xlabel("Training Split Strategy", fontsize=self.font_size)
            else:
                plt.xticks(
                    range(len(evals)), evals, rotation=90, fontsize=self.font_size
                )
                plt.xlabel("Evaluation Split Strategy", fontsize=self.font_size)

        plt.ylabel("Accuracy", fontsize=self.font_size)
        plt.ylim(-0.05, 1.05)

    def _create_legend(
        self,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        has_direct_sbs: bool,
        plot_train: bool,
        equivalence_class: bool,
    ):
        mode_pos_legend = []
        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    color = self.colors[f"{mode}_{length}"][pos]
                    pos_label = "Relative Global" if pos == "rel_global" else "Absolute"
                    mode_label = "Direct" if mode == "direct" else "Step-by-Step"

                    train_label = "Hold-out (Unseen data)"
                    test_label = "Test (Unseen tasks)"
                    if has_direct_sbs:
                        label = f"({pos_label}) - Step-by-step accuracy"
                    else:
                        label = f"({mode_label} ({pos_label}))"
                    mode_pos_legend.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            lw=2,
                            label=label,
                            markersize=self.font_size / 2,
                            marker="o",
                        )
                    )

                    if has_direct_sbs:
                        direct_label = f"({pos_label}) - Final output accuracy"
                        direct_color = self.colors[f"{mode}_direct_{length}"][pos]
                        mode_pos_legend.append(
                            Line2D(
                                [0],
                                [0],
                                color=direct_color,
                                lw=2,
                                label=direct_label,
                                markersize=self.font_size / 2,
                                marker="o",
                            )
                        )
        if plot_train:
            train_legend = [
                Line2D([0], [0], color="black", linestyle="-", lw=2, label=train_label)
            ]
            test_legend = [
                Line2D([0], [0], color="black", linestyle="--", lw=2, label=test_label)
            ]
        else:
            train_legend = []
            test_legend = []
        if equivalence_class:
            predicted_accuracy_legend = [Line2D([0], [0], color="black", linestyle="-", marker="o",
                                        label="Predicted Accuracy", markersize=self.font_size/2, linewidth=2)]
        else:
            predicted_accuracy_legend = []
        separator = Line2D([0], [0], color="none", label="")
        combined_handles = (
            mode_pos_legend
            + [separator]
            + train_legend
            + test_legend
            + predicted_accuracy_legend
        )
        combined_labels = [h.get_label() for h in combined_handles]
        # have legend in order of how plot lines are plotted from top to bottom 3, 4, 1, 2
        combined_handles = [combined_handles[1]] + [combined_handles[3]] + [combined_handles[2]] + [combined_handles[0]]
        # have legend in order of how plot lines are plotted from top to bottom 3, 4, 1, 2
        combined_labels = [combined_labels[1]] + [combined_labels[0]] + [combined_labels[2]] + [combined_labels[3]]

        plt.legend(
            combined_handles,
            combined_labels,
            loc="best",
            fontsize=self.font_size,
            frameon=False,
        )
        # add grid lines
        plt.grid(True)

    def plot_step_wise_accuracy(
        self,
        step_wise_train_pts: Dict,
        step_wise_test_pts: Dict,
        modes: List[str],
        pos_types: List[str],
        lengths: List[str],
        save_path_prefix: str,
        step_wise_mode: str,
    ):
        """Plot step-wise accuracy."""
        for mode in modes:
            for pos in pos_types:
                for length in lengths:
                    plt.figure(figsize=(12, 6))

                    train_data = step_wise_train_pts[mode][pos][length]
                    test_data = step_wise_test_pts[mode][pos][length]

                    plt.plot(
                        list(train_data.keys()),
                        list(train_data.values()),
                        label="Train",
                    )
                    plt.plot(
                        list(test_data.keys()), list(test_data.values()), label="Test"
                    )

                    plt.legend(loc="best")
                    plt.title(f"Step-wise Accuracies ({mode} {pos}, {step_wise_mode})")
                    plt.xlabel("Step")
                    plt.ylabel("Accuracy")
                    plt.tight_layout()

                    save_path = f"{save_path_prefix}_{mode}_{pos}_{length}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    plt.close()
