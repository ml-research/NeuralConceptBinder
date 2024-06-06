import numpy as np
import os
import pickle
import random
import argparse

from send_more_money_sol import solve

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_puzzle_per_digit', type=int, default=30)


def get_args():
	args = parser.parse_args()

	random.seed(args.seed)
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	np.random.seed(args.seed)
	return args


def int_digit_to_digits(digit):
	num_str = str(digit)
	digits = []
	for num in num_str:
		digits.append(int(num))
	return digits


def main():
	args = get_args()

	puzzles = []
	for n_digits in range(3, 7):
		n_puzzle_per_digit = 0
		while n_puzzle_per_digit != args.n_puzzle_per_digit:
			digit1 = np.random.randint(low=10**(n_digits-1), high=10**(n_digits)-1, size=1)[0]
			digit2 = np.random.randint(low=10**(n_digits-1), high=10**(n_digits)-1, size=1)[0]
			res = digit1 + digit2

			eq_str = f'{digit1} + {digit2} = {res}'

			if solve(eq_str) is not None:
				puzzles.append([int_digit_to_digits(digit1),
				                int_digit_to_digits(digit2),
				                int_digit_to_digits(res)])
				n_puzzle_per_digit += 1

	with open('puzzles_list.pickle', 'wb') as handle:
		pickle.dump(puzzles, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print('Saved generated puzzles to puzzles_list.pickle')
	print('Number of generated puzzles:')
	print(len(puzzles))
	# print('These are the generated puzzles')
	for puzzle in puzzles:
		print(puzzle)


if __name__ == "__main__":
	main()
