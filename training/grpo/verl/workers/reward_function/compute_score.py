import re

def is_legitimate_format(prediction):
	# Check for exactly one occurrence of each tag
	if (prediction.count('<think>') != 1 or
		prediction.count('</think>') != 1 or
		prediction.count('<score>') != 1 or
		prediction.count('</score>') != 1):
		return False
	
	# Ensure the entire string is just <think>...</think>\n<score>...</score>
	# with no extra text outside the tags
	pattern = re.compile(r'^<think>.*?</think>\s*<score>.*?</score>$', re.DOTALL)
	if not pattern.match(prediction):
		return False
	
	return True


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
	# step 1 check the format
	if not is_legitimate_format(solution_str):
		return -1

	# step 2 compute the reward
	m = re.search(r'<score>(.*?)</score>', solution_str, re.DOTALL)

	try:
		score = int(m.group(1).strip())
	except:
		# score is not an integer, essentially also format issue
		return -1
	
	reward = 0.5 * (2 - abs(score - ground_truth))

	return reward
