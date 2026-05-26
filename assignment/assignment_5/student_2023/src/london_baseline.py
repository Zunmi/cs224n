# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils


if __name__ == '__main__':
	num_examples = sum(1 for _ in open('birth_dev.tsv', encoding='utf-8'))
	total, correct = utils.evaluate_places('birth_dev.tsv', ['London'] * num_examples)
	print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
