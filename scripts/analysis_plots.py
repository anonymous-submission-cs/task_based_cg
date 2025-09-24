import argparse
import sys



sys.path.append("FILL/IN/PATH/TO/SRC/DIRECTORY")

from src.evaluation.plots import DataLoader, DataProcessor, Plotter


class PlottingPipeline:
    """Main pipeline for plotting operations."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.plotter = Plotter()

    def run(self, args):
        """Run the complete plotting pipeline."""
        # Configuration
        nh_nl = args.nh_nl
        nalph = f"nalph_26_seqlen_6_fnlen_6_taskmaxlen_{args.task_max_length}"
        evals = args.train_splits if args.self_eval else args.eval_splits

        # Load data
        data = self.data_loader.load_data(
            modes=args.modes,
            pos_types=args.pos_types,
            lengths=args.prompt_length,
            trains=args.train_splits,
            evals=evals,
            seeds=args.seeds,
            nh_nl=nh_nl,
            nalph=nalph,
            plot_direct_sbs=args.plot_direct_sbs,
            step_wise_mode=args.step_wise_mode,
            function_type=args.function_type,
            task_max_length_flag=args.task_max_length_flag,
            self_eval=args.self_eval,
        )

        # Extract points for plotting
        (
            train_mean_pts,
            train_std_pts,
            test_mean_pts,
            test_std_pts,
            direct_train_mean_pts,
            direct_test_mean_pts,
            direct_train_std_pts,
            direct_test_std_pts,
        ) = self.data_processor.extract_points(
            data=data,
            modes=args.modes,
            pos_types=args.pos_types,
            lengths=args.prompt_length,
            trains=args.train_splits,
            evals=evals,
            seeds=args.seeds,
            self_eval=args.self_eval,
            plot_direct_sbs=args.plot_direct_sbs,
        )

        # Create main plot
        save_path = self._generate_save_path(args)
        self.plotter.plot_accuracy_comparison(
            train_mean_pts=train_mean_pts,
            test_mean_pts=test_mean_pts,
            train_std_pts=train_std_pts,
            test_std_pts=test_std_pts,
            modes=args.modes,
            pos_types=args.pos_types,
            lengths=args.prompt_length,
            trains=args.train_splits,
            evals=evals,
            save_path=save_path,
            direct_train_mean_pts=direct_train_mean_pts,
            direct_test_mean_pts=direct_test_mean_pts,
            direct_train_std_pts=direct_train_std_pts,
            direct_test_std_pts=direct_test_std_pts,
            self_eval=args.self_eval,
            function_type=args.function_type,
            plot_std=args.plot_std,
            plot_train=args.plot_train,
        )

        # Create step-wise plots if requested
        if args.plot_step_wise:
            (
                step_wise_train_mean_pts,
                step_wise_test_mean_pts,
                step_wise_train_std_pts,
                step_wise_test_std_pts,
            ) = self.data_processor.extract_step_wise_points(
                data,
                args.modes,
                args.pos_types,
                args.prompt_length,
                args.train_splits,
                evals,
            )
            save_path_prefix = f"{args.results_dir}/step_wise_accs_{args.prompt_length[0]}_{args.step_wise_mode}"
            self.plotter.plot_step_wise_accuracy(
                step_wise_train_mean_pts,
                step_wise_test_mean_pts,
                step_wise_train_std_pts,
                step_wise_test_std_pts,
                args.modes,
                args.pos_types,
                args.prompt_length,
                save_path_prefix,
                args.step_wise_mode,
            )

    def _generate_save_path(self, args) -> str:
        """Generate save path for main plot."""
        if args.self_eval:
            filename = f"self_eval_combined_prompt_{args.prompt_length[0]}"
        elif len(args.train_splits) == 1:
            filename = f"train_{args.train_splits[0]}_prompt_{args.prompt_length[0]}"
        else:
            filename = (
                f"eval_{'-'.join(args.eval_splits)}_prompt_{args.prompt_length[0]}"
            )

        if args.plot_direct_sbs:
            filename += "_direct"

        return f"{args.results_dir}/{filename}.pdf"


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--prompt_length", type=str, nargs="+", default=["fixed", "variable"]
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        default=[
            "combination_1",
            "combination_2",
            "combination_3",
            "combination_4",
            "combination_5",
            "combination_6",
        ],
    )
    parser.add_argument(
        "--train_splits",
        nargs="+",
        default=[
            "combination_1",
            "combination_2",
            "combination_3",
            "combination_4",
            "combination_5",
            "combination_6",
        ],
    )
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--self_eval", action="store_true")
    parser.add_argument("--plot_direct_sbs", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["step_by_step"])
    parser.add_argument("--pos_types", nargs="+", default=["rel_global"])
    parser.add_argument("--plot_module_wise", action="store_true")
    parser.add_argument("--plot_step_wise", action="store_true")
    parser.add_argument(
        "--step_wise_mode",
        type=str,
        default="individual",
        choices=["individual", "cumulative"],
    )
    parser.add_argument("--plot_ood_ratio", action="store_true")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--function_type", type=str, default="uniform")
    parser.add_argument("--task_max_length_flag", type=str, default="fixed")
    parser.add_argument("--task_max_length", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", default=[0])
    parser.add_argument("--plot_std", action="store_true")
    parser.add_argument("--plot_train", action="store_true")
    parser.add_argument("--nh_nl", type=str, default="nh6_nl3")

    args = parser.parse_args()
    pipeline = PlottingPipeline()
    pipeline.run(args)


if __name__ == "__main__":
    main()
