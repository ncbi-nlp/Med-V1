#!/usr/bin/env python3
# encoding: utf-8

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from collections import defaultdict
import os
import shutil
import re
from glob import glob


def _torch_load(path: str):
    """Load on CPU; support newer PyTorch 'weights_only' if available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_world_size_from_actor_dir(actor_dir: str, world_size=None):
    """
    Actor dir contains files like:
      model_world_size_{ws}_rank_{rank}.pt
    If world_size is None, auto-detect (pick ws with most ranks).
    Returns: (world_size, ranks)
    """
    pattern = os.path.join(actor_dir, "model_world_size_*_rank_*.pt")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard files found in actor_dir. Expected pattern: {pattern}")

    rx = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    ws_to_ranks = defaultdict(set)
    for f in files:
        m = rx.search(f)
        if not m:
            continue
        ws = int(m.group(1))
        rk = int(m.group(2))
        ws_to_ranks[ws].add(rk)

    if not ws_to_ranks:
        raise RuntimeError(
            "Found shard files but couldn't parse world_size/rank. "
            "Expected suffix like model_world_size_8_rank_0.pt"
        )

    if world_size is None:
        world_size = max(ws_to_ranks.keys(), key=lambda ws: len(ws_to_ranks[ws]))

    ranks = sorted(ws_to_ranks[world_size])
    return world_size, ranks


def convert_checkpoint_dir(actor_dir, huggingface_model_path, output_dir, world_size=None):
    print(f"🚀 Converting: {actor_dir}")

    world_size, ranks = _infer_world_size_from_actor_dir(actor_dir, world_size)
    print(f"[INFO] Detected world_size={world_size}, ranks={ranks}")

    state_dict = defaultdict(list)
    for rank in ranks:
        filepath = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print("🔄 Loading", filepath)
        this_state_dict = _torch_load(filepath)
        for key, value in this_state_dict.items():
            if hasattr(value, "to_local"):
                value = value.to_local()
            state_dict[key].append(value)

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ HuggingFace model saved to: {output_dir}")


def remove_actor_dir(step_dir, keep=False):
    actor_path = os.path.join(step_dir, "actor")
    if os.path.exists(actor_path):
        if keep:
            print(f"✅ Keeping actor dir: {actor_path}")
        else:
            print(f"🗑️ Removing actor dir: {actor_path}")
            shutil.rmtree(actor_path)


def _require_nonempty(arg_name: str, value: str):
    if value is None or str(value).strip() == "":
        raise ValueError(f"[ERROR] Missing required argument: --{arg_name}")
    return value


def main(
    root_dir=None,                 # REQUIRED
    huggingface_model_path=None,    # REQUIRED
    step=None,                      # optional; if None uses latest_checkpointed_iteration.txt
    keep_actor=False,
    world_size=None,                # optional; auto-detect if None
):
    """
    Convert exactly one checkpoint step:
      <root_dir>/global_step_{step}/actor  ->  <root_dir>/global_step_{step}/huggingface

    Required:
      --root_dir
      --huggingface_model_path

    Optional:
      --step (default: latest step from latest_checkpointed_iteration.txt)
      --keep_actor (default: False)
      --world_size (default: auto-detect from filenames)
    """
    root_dir = _require_nonempty("root_dir", root_dir)
    huggingface_model_path = _require_nonempty("huggingface_model_path", huggingface_model_path)

    root_dir = os.path.expanduser(root_dir)
    huggingface_model_path = os.path.expanduser(huggingface_model_path)

    if step is None:
        latest_step_file = os.path.join(root_dir, "latest_checkpointed_iteration.txt")
        if not os.path.exists(latest_step_file):
            raise FileNotFoundError(
                f"[ERROR] step not provided and file not found: {latest_step_file}\n"
                "Please pass --step explicitly."
            )
        with open(latest_step_file, "r") as f:
            step = int(f.read().strip())
        print(f"[INFO] step not provided; using latest step from file: {step}")
    else:
        step = int(step)

    step_name = f"global_step_{step}"
    step_path = os.path.join(root_dir, step_name)
    actor_path = os.path.join(step_path, "actor")
    hf_output_dir = os.path.join(step_path, "huggingface")

    if not os.path.isdir(step_path):
        raise FileNotFoundError(f"[ERROR] Step directory not found: {step_path}")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"[ERROR] No actor dir found at: {actor_path}")

    convert_checkpoint_dir(actor_path, huggingface_model_path, hf_output_dir, world_size=world_size)
    remove_actor_dir(step_path, keep=keep_actor)


if __name__ == "__main__":
    fire.Fire(main)
