import itertools
import numpy as np
import random

class Mastermind():
	def __init__(self):
		# Scores
		self.a = 2 # Correct color and correct position
		self.b = 1 # Correct color but incorrect position
		self.c = 0 # Incorrect color and incorrect position
		self.colors = ["red", "green", "blue", "yellow", "pink"]  
		self.win = False
		self.reset()

	def reset(self):
		self.win = False
		self.attempts = 0
		self.color_code = random.sample(self.colors,4)

		self.colorPairs = [list(pair) for pair in itertools.permutations(self.colors, 4)]

	def guess(self,guessed_code):
		self.attempts += 1
		return_code = []
		for i in range(len(self.color_code)):
			if guessed_code[i] == self.color_code[i]:
				return_code.append(self.a)
			elif guessed_code[i] in self.color_code:
				return_code.append(self.b)
		return_code = return_code + [self.c for i in range(len(self.color_code)-len(return_code))]

		if np.sum(return_code) == 4*self.a:
			self.win = True
		return return_code

	def code2state(self,guessed_code,return_code):
		state = []
		for color in guessed_code:
			array = [0 for i in self.colors]
			array[self.colors.index(color)] = 1
			state.extend(array)
		for num in return_code:
			array = [0,0,0]
			array[num] = 1
			state.extend(array)
		return state

if __name__ == "__main__":
	mind = Mastermind()
