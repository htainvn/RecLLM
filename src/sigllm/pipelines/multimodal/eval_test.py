"""Evaluate a trained Stage-3 model on the TEST splits (test / test_warm / test_cold).

The training scripts only report validation metrics; test metrics are produced
only in eval-only mode (`run.evaluate=True`). For a Stage-3 model the trained
weights are split across two checkpoints:
  - LoRA: Step 1's checkpoint_best.pth (loaded via model.ckpt at construction).
  - Q-Former + projection + CoRA injector: Step 2's runner checkpoint_best.pth
    (the runner saves only trainable params, so LoRA/base-LLM are absent there).
This script builds the model exactly as the chosen step does, overlays the
trained Step-2 checkpoint (strict=False), flips on eval-only mode, and runs the
runner so it evaluates every split in `run.test_splits`.

Usage (from /content/SigLLM, with PYTHONPATH=src):
  # Step 2 (full soft-token / both model):
  python -m sigllm.pipelines.multimodal.eval_test --cfg-path configs/config.yaml \
      --step 2 \
      --model-ckpt ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-instruct/checkpoint_best.pth \
      --overlay-ckpt ckpt/qformer_stage3_step2_cie_qwen2/qwen2-7b-instruct/checkpoint_best.pth

  # Step 1 (text-only LLM baseline):
  python -m sigllm.pipelines.multimodal.eval_test --cfg-path configs/config.yaml \
      --step 1 \
      --model-ckpt ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-instruct/checkpoint_best.pth

The exact checkpoint paths are printed in your Step 1 / Step 2 training logs
("Load QRecLLM Checkpoint: ..." and "Saving checkpoint ... to ...").
Make sure `model.cf_injection_mode` and `model.prompt_path` in the config match
what you trained with (Step 1 = soft_token + text-only; Step 2 = whatever you ran).
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from sigllm import tasks
from sigllm.common.config import Config
from sigllm.common.dist_utils import get_rank, init_distributed_mode
from sigllm.common.utils import derive_job_id_from_llm
from sigllm.runners.runner_base_rec import RecRunnerBase  # noqa: F401  (registry side-effect)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Stage-3 model on the test splits")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--step", type=int, default=2, choices=[1, 2, 3],
                        help="Which Stage-3 step's model to evaluate "
                             "(1=text-only LoRA, 2=full, 3=joint MF+Q-Former+proj+LoRA).")
    parser.add_argument("--model-ckpt", type=str, default=None,
                        help="Checkpoint loaded as model.ckpt. Defaults to Step 1's best for "
                             "steps 1-2, and to Step 3's best for step 3 (which carries every "
                             "trained module, including MF).")
    parser.add_argument("--overlay-ckpt", type=str, default=None,
                        help="Step 2 runner checkpoint to overlay (Q-Former/proj/injector). "
                             "Step 2 only; defaults to Step 2's best. Unused for step 3, whose "
                             "single checkpoint already holds all trained weights.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config")
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def _default_ckpt(out_dir, slug, name="checkpoint_best.pth"):
    return os.path.join(out_dir, slug, name)


def main():
    args = parse_args()
    cfg = Config(args)
    slug = derive_job_id_from_llm(cfg)

    step1 = cfg.run_cfg.qformer_stage3_step1
    overlay_path = None

    if args.step == 1:
        cfg.model_cfg.tuning_step = 1
        cfg.model_cfg.prompt_path = step1.prompt_path
        model_ckpt = args.model_ckpt or _default_ckpt(
            step1.output_dir, slug, step1.best_ckpt_name)
        cfg.model_cfg.ckpt = model_ckpt
    elif args.step == 3:
        # Joint tuning trained LoRA + Q-Former + projection + MF, so all of
        # them are in the ONE runner checkpoint (the saver keeps every
        # trainable tensor) — no overlay needed. MF in particular must come
        # from here: from_config detects the rec_encoder.* keys and skips the
        # pretrained-MF restore that would otherwise overwrite the jointly
        # trained embeddings with mf_model.pth.
        step3 = cfg.run_cfg.qformer_stage3_step3
        cfg.model_cfg.tuning_step = 3
        cfg.model_cfg.prompt_path = step3.prompt_path
        cfg.model_cfg.freeze_rec = False
        model_ckpt = args.model_ckpt or _default_ckpt(
            step3.output_dir, slug, step3.get("best_ckpt_name", "checkpoint_best.pth"))
        cfg.model_cfg.ckpt = model_ckpt
        if args.overlay_ckpt:
            raise ValueError(
                "--overlay-ckpt is not applicable to step 3: its single checkpoint "
                "already contains every trained module. Pass --model-ckpt instead."
            )
    else:
        step2 = cfg.run_cfg.qformer_stage3_step2
        cfg.model_cfg.tuning_step = 2
        cfg.model_cfg.prompt_path = step2.prompt_path
        # LoRA comes from Step 1's checkpoint via model.ckpt at construction.
        model_ckpt = args.model_ckpt or _default_ckpt(
            step1.output_dir, slug, step1.best_ckpt_name)
        cfg.model_cfg.ckpt = model_ckpt
        # Trained Q-Former/projection/injector come from Step 2's runner save.
        overlay_path = args.overlay_ckpt or _default_ckpt(step2.output_dir, slug)

    # Eval-only: runner skips training and evaluates run.test_splits.
    cfg.run_cfg.evaluate = True

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    task = tasks.setup_task(cfg=cfg)
    datasets = task.build_datasets(cfg=cfg)

    first_key = list(cfg.datasets_cfg.keys())[0]
    data_dir = cfg.datasets_cfg[first_key].path
    train_ = pd.read_pickle(os.path.join(data_dir, "train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))
    test_ = pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))
    cfg.model_cfg.rec_config.user_num = int(max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1)
    cfg.pretty_print()

    model = task.build_model(cfg=cfg)

    # Overlay the trained Step-2 weights (Q-Former + projection + injector).
    if overlay_path:
        if not os.path.isfile(overlay_path):
            raise FileNotFoundError(
                f"Step-2 checkpoint not found: {overlay_path}. Pass --overlay-ckpt "
                f"with the path printed in your Step 2 log ('Saving checkpoint ... to ...')."
            )
        sd = torch.load(overlay_path, map_location="cpu")
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        msg = model.load_state_dict(sd, strict=False)
        print(f"[eval_test] overlaid Step-2 weights from {overlay_path}")
        print(f"[eval_test] overlay missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    job_id = derive_job_id_from_llm(cfg)
    runner = task.build_runner(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    # Evaluate each test split explicitly with a clear label. skip_reload=True so
    # it uses the constructed + overlaid weights (not a re-loaded checkpoint).
    eval_model = runner.unwrap_dist_model(runner.model)
    eval_model.eval()
    print("\n" + "=" * 60)
    print("TEST-SET EVALUATION")
    print("=" * 60)
    for split in cfg.run_cfg.test_splits:
        loader = runner.dataloaders.get(split)
        if loader is None:
            print(f"[{split}] no dataloader (did preprocess_test_cold_warm.py run?) — skipped")
            continue
        print(f"\n----- {split} -----")
        log = runner.eval_epoch(split_name=split, cur_epoch="best", skip_reload=True)
        print(f"[{split}] {log}")


if __name__ == "__main__":
    main()
