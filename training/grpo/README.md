<br>
<h2 id="1-1">🎯 Reinforcement Learning (GRPO)</h2>

**Installation**

```
conda create -n grpo python=3.9
conda activate grpo

# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "https://files.pythonhosted.org/packages/6e/75/b424aebc9f2fc5db319d5df5fff62fa19254c8ef974c254588d48c480df2/pyairports-2.1.1-py3-none-any.whl"
pip install "numpy<2.0" "outlines==0.0.45"
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
pip install transformers==4.47.0
pip install trl==0.17.0

# verl
cd verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```
## 📑 Contents
- [1.  Preprocess Training Data](#2-1)
- [2. Run GRPO Training](#2-2)
- [3. Convert RL Checkpoint to Hugging Face Format](#2-3)

<h3 id="2-1">🧱 1.  Preprocess Training Data </h3>

This repo does **not** include the GRPO training data. Please provide your own JSON files for:
- `train_file`
- `test_file`
- `val_file`

Convert the raw JSON data into Parquet format:

```bash
python examples/data_preprocess/process_to_parquet.py \
  --train_file /path/to/grpo_train.json \
  --test_file  /path/to/grpo_test.json \
  --val_file   /path/to/grpo_val.json \
  --local_dir  ./parquet_output
```

This creates `.parquet` files in `parquet_output/`, used for training and validation.

<h3 id="2-2">🏋️‍♂️ 2.  Run GRPO Training </h3>

This repo does **not** ship training data or model checkpoints. You must provide:

- `TRAIN_DATA`: path to `train.parquet`
- `VAL_DATA`: path to `val.parquet`
- `SFT_MERGED_CKPT`: HuggingFace repo id or local path to the SFT-merged checkpoint
- `REWARD_FN`: path to the custom reward function Python file

Run GRPO:

```bash
TRAIN_DATA=/path/to/train.parquet \
VAL_DATA=/path/to/val.parquet \
SFT_MERGED_CKPT=Qwen/Qwen2.5-3B-Instruct \
REWARD_FN=/path/to/compute_score.py \
PROJECT_NAME=your_project \
EXPERIMENT_NAME=your_experiment \
bash examples/grpo_trainer/med_v1_grpo.sh
```

Logs will be written to: 
- `${EXPERIMENT_NAME}.log`


<h3 id="2-3">📤 3. Convert RL Checkpoint to Hugging Face Format </h3>

After RL training, checkpoints are often saved in FSDP shard format under:

- `<root_dir>/global_step_<STEP>/actor/`

Use `scripts/convert_fsdp_to_hf.py` to convert a single step to Hugging Face format:

```bash
python scripts/convert_fsdp_to_hf.py \
  --root_dir /path/to/rl_checkpoints/your_experiment \
  --huggingface_model_path Qwen/Qwen2.5-3B-Instruct \
  --step 2500 \
  --keep_actor True
```

This will write the merged Hugging Face checkpoint to:
- `/path/to/rl_checkpoints/your_experiment/global_step_2500/huggingface/`

**Arguments**:
- `--root_dir`: the RL checkpoint root directory containing `global_step_*` folders and (optionally) `latest_checkpointed_iteration.txt`
- `--huggingface_model_path`: Hugging Face repo id or local model path used to load the config + tokenizer
- `--step`: (optional) which checkpoint step to convert. If omitted, the script will try to read the latest step from `latest_checkpointed_iteration.txt`
- `--world_size`: (optional) if omitted, auto-detected from shard filenames in the `actor/` directory
- `--keep_actor`: (optional) keep the `actor/` directory after conversion (default: remove it)
