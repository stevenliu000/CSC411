import tictactoe
import sys

if __name__ == "__main__":
	game = tictactoe.Environment()
	print "Game starts!"
	game.render()
	while game.done == False:
		print "It is player%i's turn..."%game.turn
		inp = raw_input("Please enter your action:")
		game.step(int(inp))
		game.render()
		print "\n"

	print "Game finished!"
	