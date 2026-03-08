import argparse
import logging
import os
import shutil

import torch
import torch.nn.functional as F

from src.data_generation.generator import get_evalLoaders, get_sep_pos, get_trainLoader
from src.data_generation.init import read_config, set_seed
from src.models.nanogpt import nanoGPT
from src.training.trainer import (
    configure_optimizers,
    evaluate,
    log_eval,
    log_train,
    move_to_device,
    sanity_checks,
    save_model,
    update_cosine_warmup_lr,
)

ROOT_DIR = "FILL/IN/PATH/TO/SRC/DIRECTORY"


def main(cfg, logger):
    set_seed(cfg.seed)

    # Get data
    trainLoader = get_trainLoader(cfg)
    evalLoaders = get_evalLoaders(cfg)

    # Check if network is compatible with data
    sanity_checks(cfg, trainLoader)

    # Load network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, optimizer = initialize_network_and_optimizer(cfg, device)
    logger.info("number of parameters: %.2fM" % (net.get_num_params() / 1e6,))
    # print the config
    logger.info(cfg)

    train(cfg, net, (trainLoader, evalLoaders), optimizer, device, logger)


def initialize_network_and_optimizer(cfg, device):
    net = nanoGPT(cfg.net)
    net.to(device)
    if cfg.net.compile:
        net = torch.compile(net)
    optimizer = configure_optimizers(net, cfg.optimizer)
    return net, optimizer


def train(cfg, net, loaders, optimizer, device, logger):

    net.train()
    trainLoader, evalLoaders = loaders

    dt = torch.bfloat16 if cfg.bf16 else torch.float32
    device_info = (device, dt)

    # space_pos is the position of the seperator token. After this token, the
    # transformer should predict the output of the function. We use this to
    # compute the loss only after the seperator code (only during evaluation)
    pad_pos = get_sep_pos(cfg.data.path, trainLoader)
    nheads_nlayers = f"nh{cfg.net.n_head}_nl{cfg.net.n_layer}"
    n_alphabets_seq_len_fn_len_task_max_length = (
        cfg.data.n_alphabets_seq_len_fn_len_task_max_length
    )

    fdir_original = "{}/models/ckpts/{}/{}/{}/{}/{}/{}/{}/seed_{}".format(
        ROOT_DIR,
        cfg.function_type,
        cfg.prompt_length,
        n_alphabets_seq_len_fn_len_task_max_length,
        cfg.tag,
        cfg.train_split,
        cfg.net.pos_embedding_type,
        nheads_nlayers,
        cfg.seed,
    )

    for run in range(cfg.num_runs):
        # set the seed for each run
        set_seed(cfg.seed + run)
        lr, it = 0.0, 0
        total_steps = len(trainLoader) * cfg.epochs
        train_loss = []
        if cfg.num_runs > 1:
            fdir = fdir_original + f"/run_{run}"
            net, optimizer = initialize_network_and_optimizer(cfg, device)
        else:
            fdir = fdir_original
        if os.path.exists(fdir):
            shutil.rmtree(fdir)
        os.makedirs(fdir, exist_ok=True)

        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Learning rate warmup steps: {cfg.optimizer.warmup_iters}")

        for _ in range(cfg.epochs):
            for dat, targets in trainLoader:
                if it % cfg.log.eval_interval == 0:
                    eval_info = evaluate(net, evalLoaders, pad_pos, device_info)
                    log_eval(it, lr, eval_info, logger=logger)

                elif it % cfg.log.log_interval == 0:
                    train_loss = log_train(it, lr, train_loss)

                # Update LR
                it, lr = update_cosine_warmup_lr(
                    it, cfg.optimizer, optimizer, total_steps
                )

                optimizer.zero_grad(set_to_none=True)
                dat, targets = move_to_device(dat, targets, device)

                # Compute loss
                with torch.amp.autocast(device_type=device, dtype=dt):
                    logits = net(dat)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )

                    train_loss.append(loss.item())

                # Update model
                loss.backward()
                if cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), cfg.optimizer.grad_clip
                    )

                optimizer.step()

        # Log one final time
        eval_info = evaluate(net, evalLoaders, pad_pos, device_info)
        log_eval(it, lr, eval_info, logger=logger)
        save_model(cfg, net, optimizer, it, fdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--prompt_length", type=str, default="fixed", help="fixed or variable"
    )
    parser.add_argument(
        "--train_split", type=str, default="combination_6", help="Model training split"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--n_alphabets", type=int, default=10, help="number of alphabets"
    )
    parser.add_argument("--seq_len", type=int, default=6, help="sequence length")
    parser.add_argument(
        "--n_functions", type=int, default=6, help="number of functions"
    )
    parser.add_argument(
        "--pos_embedding_type", type=str, default="abs", help="abs or rel_global"
    )
    parser.add_argument(
        "--n_heads_nlayers",
        type=str,
        default="nh6_nl3",
        help="number of heads and layers",
    )
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=6, help="task max length"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )

    args = parser.parse_args()
    cfg = read_config(f"{ROOT_DIR}/config/train/conf.yaml")
    cfg.tag = args.prompt_mode
    cfg.prompt_length = args.prompt_length
    cfg.net.prompt_length = args.prompt_length
    cfg.train_split = args.train_split
    cfg.epochs = args.epochs
    cfg.task_max_length = args.task_max_length
    cfg.data.n_alphabets_seq_len_fn_len_task_max_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            args.n_alphabets, args.seq_len, args.n_functions, args.task_max_length
        )
    )
    cfg.data.path = "{}/data/{}/{}/{}/{}/{}".format(
        ROOT_DIR,
        args.function_type,
        cfg.prompt_length,
        cfg.data.n_alphabets_seq_len_fn_len_task_max_length,
        cfg.tag,
        cfg.train_split,
    )
    cfg.net.pos_embedding_type = args.pos_embedding_type
    split_nhk_nlj = args.n_heads_nlayers.split("_")
    # extract k and j from split_nhk_nlj, number of heads and layers given as nhk_nlj
    n_heads = int(split_nhk_nlj[0].split("h")[1])
    n_layers = int(split_nhk_nlj[1].split("l")[1])
    cfg.net.n_head = n_heads
    cfg.net.n_layer = n_layers
    cfg.num_runs = args.num_runs
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    # Initialize logger
    log_path = "{}/logs/".format(ROOT_DIR)
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger(__name__)

    log_file_dir = "{}/{}/{}/{}/{}/model_{}/{}".format(
        log_path,
        cfg.function_type,
        cfg.data.n_alphabets_seq_len_fn_len_task_max_length,
        cfg.tag,
        cfg.prompt_length,
        cfg.train_split,
        cfg.net.pos_embedding_type,
    )
    print("log_file_dir", log_file_dir)

    os.makedirs(log_file_dir, exist_ok=True)
    # Set up logging configuration
    logging.basicConfig(
        filename="{}/train.log".format(log_file_dir),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    logger.info("Initializing Trainer...")
    main(cfg, logger)
