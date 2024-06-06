"""
Based on https://colab.research.google.com/github/norvig/pytudes/blob/main/ipynb/Cryptarithmetic.ipynb#scrollTo=ymY68KEZUYbX
"""
from typing import Iterable, Callable, Tuple
from itertools import permutations
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eq', type=str, default='NUM + BER = PLAY')


Formula  = str # A formula in Math notation, like "NUM ^ BER = PLAY"
Pformula = str # A formula in Python notation, like "NUM ** BER == PLAY"
Solution = str # A formula with letters relaced by digits, like "356 + 742 = 1098"


def solve(formula) -> Iterable[Solution]:
	"""Given a formula like 'NUM + BER == PLAY', fill in digits to solve it."""
	letters = all_letters(formula)
	pformula = to_python(formula)
	for digits in permutations('1234567890', len(letters)):
		if valid(subst(digits, letters, pformula)):
			yield subst(digits, letters, formula)


def subst(digits, letters, formula) -> Formula:
	"""Substitute digits for letters in formula."""
	return formula.translate(str.maketrans(letters, cat(digits)))


def valid(pformula) -> bool:
	"""A pformula is valid iff it has no leading zero, and evaluates to True."""
	try:
		return (not leading_zero(pformula)) and (eval(pformula) is True)
	except ArithmeticError:
		return False


def to_python(formula: Formula) -> Pformula:
	"""Convert ' = ' and '^' to ' == ' and '**'."""
	return formula.replace(' = ', ' == ').replace('^', '**')


def all_letters(formula: str) -> str:
	"""The set of letters in formula, in the form of an alphabetized string."""
	return cat(sorted(set(re.findall('[A-Z]', formula))))


def first(iterable): "First element"; return next(iter(iterable), None)


def convert_digits_in_eq(eq_str):
	"""In case the equation is already written in digits convert them to characters for compatability."""
	letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	s = eq_str
	for i in range(9):
		s = re.sub(str(i), letters[i], s)
	return s

cat = ''.join  # Function to concatenate strings

leading_zero = re.compile(r'\b0[0-9]').search  # Function to check for illegal number


def main():
	args = parser.parse_args()

	eq = convert_digits_in_eq(args.eq)

	print(eq)
	print(first(solve(eq)))
	for i in solve(eq):
		print(i)


if __name__ == "__main__":
	main()

