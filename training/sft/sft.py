# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
OMP_NUM_THREADS=8 nohup  \
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file ../default_config.yaml sft.py \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_name ./data_0314.jsonl \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --output_dir ./models/llama33_70BI_0314_v2\
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --dataset_num_proc 20 \
    --save_strategy epoch > training.out 2>error.log &

"""

import argparse


from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import os
import json
import torch
import datasets
import pandas as pd

import pickle
os.environ['TF_TOKEN'] = '*****************'


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True, num_proc = 10,
    )


    ################
    # Dataset
    ################

    train_dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'log.json'), 'w') as f:
        json.dump(trainer.state.log_history, f)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)




