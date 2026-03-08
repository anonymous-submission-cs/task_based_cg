import logging

from init import read_config, set_seed, ROOT_DIR
from src.training.lora_finetuning import LiveCodeBenchLoRAFineTuner


def main():
    cfg = read_config(f"{ROOT_DIR}/config/finetune/livecodebench_lora.yaml")
    set_seed(cfg.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("finetune_livecodebench_lora")
    finetuner = LiveCodeBenchLoRAFineTuner(cfg=cfg, logger=logger)
    finetuner.training_loop()


if __name__ == "__main__":
    main()
