import argparse
from utils import run_med_v1

def main():
	parser = argparse.ArgumentParser(description="Run Med-V1 on a dataset.")
	parser.add_argument("--input_path", type=str, required=True, help="Path to the input data in JSONL format.")
	parser.add_argument("--output_path", type=str, required=True, help="Path to save the output in JSONL format.")
	parser.add_argument("--model_path", type=str, help="Path to a local pre-trained model.")

	args = parser.parse_args()
	run_med_v1(args.model_path, args.input_path, args.output_path)

if __name__ == "__main__":
	main()
