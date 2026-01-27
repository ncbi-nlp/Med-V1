# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def get_final_answer(answer_str):
	return answer_str


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_dir', default='~/data/mydata')
	parser.add_argument('--hdfs_dir', default=None)
	
	parser.add_argument('--train_file', required=True, help='Path to train JSON file')
	parser.add_argument('--test_file', required=True, help='Path to test JSON file')
	parser.add_argument('--val_file', required=True, help='Path to val JSON file')

	args = parser.parse_args()

	data_files = {
	  "train": args.train_file,
	  "test": args.test_file,
	  "val": args.val_file,
	}
	dataset = datasets.load_dataset("json", data_files=data_files)

	train_dataset = dataset['train']
	test_dataset = dataset['test']
	val_dataset = dataset['val']
	
	instruction_following = (
		"You are a fact-checking expert trained in evidence-based medicine. Your task is to evaluate how strongly an *article* agrees or disagrees with a *claim*. The *article* is retrieved from a search engine using the *claim* as the query.\n\nUse the following five-point scale:\n   - **-2 Strong Contradiction**  \u2013 The article clearly and directly refutes the claim.\n	- **-1 Partial Contradiction** \u2013 The article provides mixed or indirect evidence against the claim.\n	 - ** 0 Neutral / Unrelated**	\u2013 The article does not address the claim, offers insufficient information, or is irrelevant to the claim.\n   - ** 1 Partial Agreement**\t \u2013 The article offers some indirect or tentative support for the claim.\n	- ** 2 Strong Agreement**\t \u2013 The article explicitly and strongly supports the claim.\n\nNote that the *article* might not describe the exact same subjects, interventions, or measurements as the *claim*. In this case, please note the difference and assign a score of 0. \n\nOutput in two parts only and do not output anything else:\n<think>[your detailed, step\u2010by\u2010step explanation for scoring]</think>\n<score>[the integer score only, i.e., -2, -1, 0, 1, or 2]</score>"
	)
	
	# add a row to each data item that represents a unique id
	def make_map_fn(split):

		def process_fn(example, idx):
			question_raw = example.pop('user_msg')

			# question = question_raw + ' ' + instruction_following
			#question = instruction_following + '\n' + question_raw

			answer_raw = example.pop('ground_truth_score')
			final_answer = get_final_answer(answer_raw)
			data = {
				"data_source": "cell_data",
				"prompt": [
					{
						"role": "system",
						"content": instruction_following,
					},
					{
						"role": "user",
						"content": question_raw,
					},
				],
				"ability": "claim_verification",
				"reward_model": {
					"style": "rule",
					"ground_truth": final_answer
				},
				"extra_info": {
					'split': split,
					'index': idx,
					'answer': answer_raw,
					"question": question_raw,
				}
			}
			return data

		return process_fn

	train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
	test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
	val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)

	local_dir = args.local_dir
	hdfs_dir = args.hdfs_dir

	train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
	test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
	val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))

	if hdfs_dir is not None:
		makedirs(hdfs_dir)

		copy(src=local_dir, dst=hdfs_dir)
