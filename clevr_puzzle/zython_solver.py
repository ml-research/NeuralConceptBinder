# import zython as zn
# import numpy as np
#
#
# # class MoneyModel(zn.Model):
# #     def __init__(self):
# #         self.S = zn.var(range(1, 10))
# #         self.E = zn.var(range(0, 10))
# #         self.N = zn.var(range(0, 10))
# #         self.D = zn.var(range(0, 10))
# #         self.M = zn.var(range(1, 10))
# #         self.O = zn.var(range(0, 10))
# #         self.R = zn.var(range(0, 10))
# #         self.Y = zn.var(range(0, 10))
# #
# #         self.constraints = [(self.S * 1000 + self.E * 100 + self.N * 10 + self.D +
# #                              self.M * 1000 + self.O * 100 + self.R * 10 + self.E ==
# #                              self.M * 10000 + self.O * 1000 + self.N * 100 + self.E * 10 + self.Y),
# #                              zn.alldifferent((self.S, self.E, self.N, self.D, self.M, self.O, self.R, self.Y))]
# #
# # model = MoneyModel()
# # result = model.solve_satisfy()
# # print(" ", result["S"], result["E"], result["N"], result["D"])
# # print(" ", result["M"], result["O"], result["R"], result["E"])
# # print(result["M"], result["O"], result["N"], result["E"], result["Y"])
#
#
# class ConstraintModel(zn.Model):
# 	def __init__(self, digits):
# 		self.chars = []
# 		for digit in np.unique(np.concatenate(digits)):
# 			self.chars.append(zn.var(range(1, 10)))
# 		# for
# 		self.S = zn.var(range(1, 10))
# 		self.E = zn.var(range(0, 10))
# 		self.N = zn.var(range(0, 10))
# 		self.D = zn.var(range(0, 10))
# 		self.M = zn.var(range(1, 10))
# 		self.O = zn.var(range(0, 10))
# 		self.R = zn.var(range(0, 10))
# 		self.Y = zn.var(range(0, 10))
#
# 		self.constraints = [(self.S * 1000 + self.E * 100 + self.N * 10 + self.D +
# 							 self.M * 1000 + self.O * 100 + self.R * 10 + self.E ==
# 							 self.M * 10000 + self.O * 1000 + self.N * 100 + self.E * 10 + self.Y),
# 							 zn.alldifferent((self.S, self.E, self.N, self.D, self.M, self.O, self.R, self.Y))]
#
#
# def convert_digits_in_eq(eq_str):
# 	"""In case the equation is already written in digits convert them to characters for compatability."""
# 	letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# 	s = eq_str
# 	for i in range(9):
# 		s = re.sub(str(i), letters[i], s)
# 	return s
#
#
# def state_and_solve(digits):
# 	model = ConstraintModel(digits)
# 	result = model.solve_satisfy()
# 	print(" ", result["S"], result["E"], result["N"], result["D"])
# 	print(" ", result["M"], result["O"], result["R"], result["E"])
# 	print(result["M"], result["O"], result["N"], result["E"], result["Y"])
#
#
#
# def main():
# 	digits = [[8, 9, 1, 2, 8], [4, 3, 5, 3, 8], [1, 3, 2, 6, 6, 6]]
# 	state_and_solve(digits)
#
# if __name__ == "__main__":
# 	main()
