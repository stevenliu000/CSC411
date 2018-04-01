from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

#random seed
np.random.seed(1)
torch.manual_seed(1)

class Environment(object):
	"""
	The Tic-Tac-Toe Environment
	"""
	# possible ways to win
	win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
						 (0,3,6), (1,4,7), (2,5,8), # vertical
						 (0,4,8), (2,4,6)])         # diagonal
	# statuses
	STATUS_VALID_MOVE = 'valid'
	STATUS_INVALID_MOVE = 'inv'
	STATUS_WIN = 'win'
	STATUS_TIE = 'tie'
	STATUS_LOSE = 'lose'
	STATUS_DONE = 'done'

	def __init__(self):
		self.reset()

	def reset(self):
		"""Reset the game to an empty board."""
		self.grid = np.array([0] * 9) # grid
		self.turn = 1                 # whose turn it is
		self.done = False             # whether game is done

		#modify
		self.numStep = 0


		return self.grid

	def render(self):
		"""Print what is on the board."""
		map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
		print(''.join(map[i] for i in self.grid[0:3]))
		print(''.join(map[i] for i in self.grid[3:6]))
		print(''.join(map[i] for i in self.grid[6:9]))
		print('====')

	def check_win(self):
		"""Check if someone has won the game."""
		for pos in self.win_set:
			s = set([self.grid[p] for p in pos])
			if len(s) == 1 and (0 not in s):
				return True
		return False

	def step(self, action):
		"""Mark a point on position action."""
		assert type(action) == int and action >= 0 and action < 9
		# done = already finished the game

		#modify
		self.numStep += 1
		if self.numStep >= 10000:
			self.done = True
			
		if self.done:
			return self.grid, self.STATUS_DONE, self.done
		# action already have something on it
		if self.grid[action] != 0:
			return self.grid, self.STATUS_INVALID_MOVE, self.done
		# play move
		self.grid[action] = self.turn
		if self.turn == 1:
			self.turn = 2
		else:
			self.turn = 1
		# check win
		if self.check_win():
			self.done = True
			return self.grid, self.STATUS_WIN, self.done
		# check tie
		if all([p != 0 for p in self.grid]):
			self.done = True
			return self.grid, self.STATUS_TIE, self.done
		return self.grid, self.STATUS_VALID_MOVE, self.done

	def random_step(self):
		"""Choose a random, unoccupied move on the board to play."""
		pos = [i for i in range(9) if self.grid[i] == 0]
		move = random.choice(pos)
		return self.step(move)

	def play_against_random(self, action):
		"""Play a move, and then have a random agent play the next move."""
		state, status, done = self.step(action)
		if not done and self.turn == 2:
			state, s2, done = self.random_step()
			if done:
				if s2 == self.STATUS_WIN:
					status = self.STATUS_LOSE
				elif s2 == self.STATUS_TIE:
					status = self.STATUS_TIE
				else:
					raise ValueError("???")
		return state, status, done

class Policy(nn.Module):
	"""
	The Tic-Tac-Toe Policy
	"""
	def __init__(self, input_size=27, hidden_size=64, output_size=9):
		super(Policy, self).__init__()
		# TODO
		self.features = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size),
            nn.ReLU(),
			nn.Softmax()
			)

	def forward(self, x):
		# TODO
		return self.features(x)

def select_action(policy, state):
	"""Samples an action from the policy at the state."""
	state = torch.from_numpy(state).long().unsqueeze(0)
	state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
	pr = policy(Variable(state))
	m = torch.distributions.Categorical(pr)
	action = m.sample()
	log_prob = torch.sum(m.log_prob(action))
	return action.data[0], log_prob

def compute_returns(rewards, gamma=1):
	"""
	Compute returns for each time step, given the rewards
	  @param rewards: list of floats, where rewards[t] is the reward
					  obtained at time step t
	  @param gamma: the discount factor
	  @returns list of floats representing the episode's returns
		  G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

	>>> compute_returns([0,0,0,1], 1.0)
	[1.0, 1.0, 1.0, 1.0]
	>>> compute_returns([0,0,0,1], 0.9)
	[0.7290000000000001, 0.81, 0.9, 1.0]
	>>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
	[-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
	"""
	# TODO
	result = []
	for i in range(len(rewards)):
		temp = 0.0
		gammaAccumulated = 1/gamma
		for j in range(i, len(rewards)):
			gammaAccumulated = gammaAccumulated * gamma
			temp += gammaAccumulated * rewards[j]
		result.append(temp)
	return result

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
	"""Samples an action from the policy at the state."""
	policy_loss = []
	returns = compute_returns(saved_rewards, gamma)
	returns = torch.Tensor(returns)
	# subtract mean and std for faster training
	returns = (returns - returns.mean()) / (returns.std() +
											np.finfo(np.float32).eps)
	for log_prob, reward in zip(saved_logprobs, returns):
		policy_loss.append(-log_prob * reward)
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward(retain_graph=True)
	# note: retain_graph=True allows for multiple calls to .backward()
	# in a single step

def get_reward(status):
	"""Returns a numeric given an environment status."""
	return {
			Environment.STATUS_VALID_MOVE  : 0,
			Environment.STATUS_INVALID_MOVE: -1000,
			Environment.STATUS_WIN         : 100,
			Environment.STATUS_TIE         : 0,
			Environment.STATUS_LOSE        : -100
	}[status]

def train(policy, env, gamma=1.0, log_interval=1000, ifSave=False):
	"""Train policy gradient."""
	optimizer = optim.Adam(policy.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer, step_size=10000, gamma=1.0)
	running_reward = 0

	avgReturn_ = []
	i_episode_ = []

	for i_episode in range(100000):
		saved_rewards = []
		saved_logprobs = []
		state = env.reset()
		done = False
		while not done:
			action, logprob = select_action(policy, state)
			state, status, done = env.play_against_random(action)
			reward = get_reward(status)
			saved_logprobs.append(logprob)
			saved_rewards.append(reward)

		R = compute_returns(saved_rewards)[0]
		running_reward += R

		finish_episode(saved_rewards, saved_logprobs, gamma)

		if i_episode % log_interval == 0:
			print('Episode {}\tAverage return: {:.2f}'.format(
				i_episode,
				running_reward / log_interval))
			avgReturn_.append(running_reward / log_interval)
			i_episode_.append(i_episode)
			running_reward = 0

		if ifSave:
			if i_episode % (log_interval) == 0:
				torch.save(policy.state_dict(),
						   "ttt/policy-%d.pkl" % i_episode)
		
		if i_episode % 1 == 0: # batch_size
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

	return i_episode_, avgReturn_




def first_move_distr(policy, env):
	"""Display the distribution of first moves."""
	state = env.reset()
	state = torch.from_numpy(state).long().unsqueeze(0)
	state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
	pr = policy(Variable(state))
	return pr.data


def load_weights(policy, episode):
	"""Load saved weights"""
	weights = torch.load("ttt/policy-%d.pkl" % episode)
	policy.load_state_dict(weights)

#----------------------------Helper functions----------------------------------

def Part5a():
	policy = Policy(hidden_size=64)
	env = Environment()
	gamma = 1.0
	i_episode_, avgReturn_ = train(policy, env, gamma=gamma,ifSave = True)
	fig = plt.figure(0)
	plt.plot(i_episode_, avgReturn_)
	plt.xlabel("i_episode")
	plt.ylabel("Average Return")
	fig.savefig("part5a.png") 
	plt.show()

def Part5_find_gamma():
	import sys
	orig_stdout = sys.stdout
	f = open("Part5_find_gamma_output", "w")
	sys.stdout = f
	policy = Policy()

	env = Environment()
	for gamma in [0.1,0.3,0.5,0.7,0.9,1]:
		print("gamma = %02.2f Begin! \n"%gamma)
		i_episode_, avgReturn_ = train(policy, env, gamma=gamma,ifSave = False)
		fig = plt.figure(0)
		plt.plot(i_episode_, avgReturn_)
		plt.xlabel("i_episode")
		plt.ylabel("Average Return")
		fig.savefig("part5_gamma%02.2f.png"%gamma) 
		plt.show()

	sys.stdout = orig_stdout
	f.close()

def part5b():
	for hidden_size in [5,10,30,50,70,100,150,200]:
		policy = Policy(hidden_size=hidden_size)
		env = Environment()
		i_episode_, avgReturn_ = train(policy, env, ifSave = False)
		fig = plt.figure(0)
		plt.plot(i_episode_, avgReturn_)
		plt.xlabel("i_episode")
		plt.ylabel("Average Return")
		fig.savefig("part5b_%i"%hidden_size) 
		plt.show()

def part5d():
	'''
	Game render result is stored in file part5d_output
	'''
	import sys
	orig_stdout = sys.stdout
	f = open("part5d_output", "w")
	sys.stdout = f

	policy = Policy(hidden_size=64)
	env = Environment()
	win = 0
	lose = 0
	tie = 0
	load_weights(policy, 99000)
	for i in range(100):
		print("Game No.%i begin!"%(i+1))
		state = env.reset()
		done = False
		while not done:
			action, logprob = select_action(policy, state)
			state, status, done = env.play_against_random(action)
			env.render()
		if status == env.STATUS_WIN:
			win += 1
		elif status == env.STATUS_LOSE:
			lose += 1
		elif status == env.STATUS_TIE:
			tie += 1
		print("Game No.%i finished! Result: %s \n"%(i+1, status))

	sys.stdout = orig_stdout
	f.close()

	print("win:%i ; lose:%i; tie:%i"%(win,lose,tie))

def part6():
	policy = Policy(hidden_size=64)
	env = Environment()
	win_ = []
	lose_ = []
	tie_ = []
	for i_episode in range(0,100000,1000):
		policy = Policy(hidden_size = 64)
		win = 0
		lose = 0
		tie = 0
		load_weights(policy, i_episode)
		for i in range(100):
			state = env.reset()
			done = False
			while not done:
				action, logprob = select_action(policy, state)
				state, status, done = env.play_against_random(action)
				# env.render()
			if status == env.STATUS_WIN:
				win += 1
			elif status == env.STATUS_LOSE:
				lose += 1
			elif status == env.STATUS_TIE:
				tie += 1
		win_.append(win/100.0)
		lose_.append(lose/100.0)
		tie_.append(tie/100.0)

	i_episode_ = range(0,100000,1000)
	fig = plt.figure(6)
	plt.plot(i_episode_, win_, label="win ratio")
	plt.plot(i_episode_, lose_, label="lose ratio")
	plt.plot(i_episode_, tie_, label="tie ratio")
	plt.legend(loc = "center right")
	plt.xlabel("i_episode")
	plt.ylabel("ratio")
	fig.savefig("part6") 
	plt.show()

def part7():
	policy = Policy(hidden_size=64)
	env = Environment()
	moveProb_ = np.empty((0,9))
	for i_episode_ in range(0,100000,1000):
		load_weights(policy,i_episode_)
		move = first_move_distr(policy, env)
		move = np.array(move.tolist())
		moveProb_ = np.vstack((moveProb_,move))
		
	fig = plt.figure(71)
	plt.title("Learned distribution over the first move")
	plt.imshow(moveProb_[-1,:].reshape(3,3))
	fig.savefig("part7_fullyTrained")
	plt.show()
	
	i_episode_ = range(0,100000,1000)
	for i in range(1,10):
		row = (i-1)/3+1
		col = i-3*((i-1)/3)
		fig = plt.figure(71+i)
		plt.title("First move probability on position (%i, %i)" %(row,col))
		plt.plot(i_episode_, moveProb_[:,i-1])
		plt.legend(loc = "best")
		plt.ylabel("probability")
		fig.savefig("part7_position_%i,%i"%(row,col))

'''
if __name__ == '__main__':
	import sys
	policy = Policy()
	env = Environment()

	if len(sys.argv) == 1:
		# `python tictactoe.py` to train the agent
		train(policy, env)
	else:
		# `python tictactoe.py <ep>` to print the first move distribution
		# using weightt checkpoint at episode int(<ep>)
		ep = int(sys.argv[1])
		load_weights(policy, ep)
		print(first_move_distr(policy, env))
'''