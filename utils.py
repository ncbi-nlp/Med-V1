import json
import re
import sys
import tqdm
import glob
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any
import torch

med_v1_system_prompt = """You are a fact-checking expert trained in evidence-based medicine. Your task is to evaluate how strongly an *article* agrees or disagrees with a *claim*. The *article* is retrieved from a search engine using the *claim* as the query.

Use the following five-point scale:
   - **-2 Strong Contradiction**  – The article clearly and directly refutes the claim.
   - **-1 Partial Contradiction** – The article provides mixed or indirect evidence against the claim.
   - ** 0 Neutral / Unrelated**   – The article does not address the claim, offers insufficient information, or is irrelevant to the claim.
   - ** 1 Partial Agreement**	 – The article offers some indirect or tentative support for the claim.
   - ** 2 Strong Agreement**	 – The article explicitly and strongly supports the claim.

Note that the *article* might not describe the exact same subjects, interventions, or measurements as the *claim*. In this case, please note the difference and assign a score of 0. 

Output in two parts only and do not output anything else:
<think>[your detailed, step‐by‐step explanation for scoring]</think>
<score>[the integer score only, i.e., -2, -1, 0, 1, or 2]</score>""" 

def get_med_v1_messages(source, claim) -> List[Dict]: 

	user_msg = f"Article:\n{source}\n\nClaim:\n{claim}".strip()
	
	return med_v1_system_prompt, user_msg 

def parse_llm_output(output: str) -> Optional[Dict[str, Any]]:
	"""Extracts rationale and score from the LLM's output."""
	if not output:
		return None

	think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
	score_match = re.search(r"<score>(.*?)</score>", output, re.DOTALL)

	rationale = think_match.group(1).strip() if think_match else ""
	score_str = score_match.group(1).strip() if score_match else ""

	try:
		score = int(score_str)
	except (ValueError, TypeError):
		score = None

	return {"rationale": rationale, "score": score}


def run_med_v1(model_path: str, input_path: str, output_path: str) -> None:
	"""Running Med-V1 locally from a pre-trained model"""
	if input_path.endswith(".jsonl"):
		dataset = load_jsonl(input_path)
	elif input_path.endswith(".json"):
		dataset = load_json(input_path)

	# check if there is cached output
	if os.path.exists(output_path):
		output = load_jsonl(output_path)
		finished_inds = set(entry["file_idx"] for entry in output)
	else:
		finished_inds = set()

	tasks = []
	for idx, entry in enumerate(dataset):
		if str(idx) not in finished_inds:
			tasks.append((str(idx), entry))

	if not tasks:
		print("All tasks have been completed.")
		return

	tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

	if not tokenizer.pad_token:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	# llm utils code
	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		device_map="auto",
		cache_dir="./med_v1_model",
	)

	model.eval()

	generator = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
	)

	all_messages = []
	for _, entry in tasks:

		if "system_prompt" not in entry:
			entry["system_prompt"] = med_v1_system_prompt
		if "user_prompt" not in entry:
			entry["user_prompt"] = f"Article:\n{entry['source']}\n\nClaim:\n{entry['claim']}"

		messages = [
			{"role": "system", "content": entry["system_prompt"]},
			{"role": "user", "content": entry["user_prompt"]},
		]
		all_messages.append(messages)
	
	with torch.no_grad():
		completions = generator(
			all_messages,
			batch_size=8,
			do_sample=False,
			max_new_tokens=1024,
			temperature=None,
			top_p=None,
		)

	for (idx, entry), completion in zip(tasks, completions):
		raw_output = completion[0]["generated_text"][-1]["content"]
		extracted_output = parse_llm_output(raw_output)

		if not extracted_output:
			continue

		output_entry = {
			"_file_idx": idx,
			"raw_output": raw_output,
			"extracted_rationale": extracted_output["rationale"],
			"extracted_score": extracted_output["score"],
			**entry,
		}

		with open(output_path, "a") as f:
			f.write(json.dumps(output_entry) + "\n")

def load_json(file_path: str) -> Any:
	"""Loads a JSON file."""
	with open(file_path, 'r', encoding='utf-8') as f:
		return json.load(f)

def save_json(data: Any, file_path: str, indent: int = 4) -> None:
	"""Saves data to a JSON file."""
	with open(file_path, 'w', encoding='utf-8') as f:
		json.dump(data, f, indent=indent, ensure_ascii=False)

def load_jsonl(file_path: str) -> List[Dict]:
	"""Loads a JSONL file."""
	data = []
	with open(file_path, 'r', encoding='utf-8') as f:
		for line in f:
			data.append(json.loads(line))
	return data

def save_jsonl(data: List[Dict], file_path: str) -> None:
	"""Saves data to a JSONL file."""
	with open(file_path, 'w', encoding='utf-8') as f:
		for item in data:
			f.write(json.dumps(item, ensure_ascii=False) + '\n')
