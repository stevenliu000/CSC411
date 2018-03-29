from __future__ import print_function
from collections import defaultdict
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable

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
    def __init__(self, input_size=27, hidden_size=50, output_size=9):
        super(Policy, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,output_size),
            torch.nn.Softmax(),
            )

    def forward(self, x):
        return self.model(x)

def select_action(policy, state): 
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0) #augment dimension from 9 to 1x9
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27) #see doc for tensor.scetter_
    pr = policy(Variable(state))  #the policy returns the probabilities of selecting
                                  #each position as the next move
    m = torch.distributions.Categorical(pr) 
                                  #create a categorical distribution described by pr
    action = m.sample()           #sample from the distribution -> select an action
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob 

def compute_returns(rewards, gamma=1.0):
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
    returns = []
    for i in range (len(rewards)):
        toAppend = 0
        for j in range (i,len(rewards)):
            toAppend += (gamma**(j-i)) * rewards[j]
        returns.append(toAppend)
    return returns

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
    policy_loss = torch.cat(policy_loss).sum()  #148 - 150 doing sampling (forming the sum)
    policy_loss.backward(retain_graph=True)     #this line computes the grad_theta
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step
    '''
    policy grad: grad_theta (J(theta)) = E(discount^t*G_t*grad_theta log(P(a_t|s_t)))   How to estimate this? ---> sampling
    through sampling, get returns [r0, r1, ....]
    through sampling, get log probabilities of [log P(a0|s0), ....] 
    these are the inputs to finish_episode
    '''

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0,
            Environment.STATUS_INVALID_MOVE: -10000,
            Environment.STATUS_WIN         : 1,
            Environment.STATUS_TIE         : 0,
            Environment.STATUS_LOSE        : -1
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000,save = True):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    episodes = []
    avgReturns = []

    # for i_episode in count(1):
    for i_episode in range (1000000):
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
            episodes.append(i_episode)
            avgReturns.append(running_reward / log_interval)
            running_reward = 0
        if save:  
            if i_episode % (log_interval) == 0:
                torch.save(policy.state_dict(),
                           "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return episodes, avgReturns

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

#************************************************************************

def part5a (size):
    policy = Policy (hidden_size = size)
    episodes,avgReturns = train(policy, env)
    plt.figure()
    plt.plot(episodes,avgReturns,color ='b')
    plt.xlabel('episodes')
    plt.ylabel('avgReturns')
    plt.legend()
    plt.title('The relationship between episodes and the average return')
    plt.savefig('pictures/part5a.jpg')

def part5b():
    plt.figure()
    hidden_sizes = [30,50,100]
    legends = ['hSize = 30','hSize = 50','hSize = 100']
    colors = ['r','b','g']
    for i in range(len(hidden_sizes)):
        print ('hidden sizes:', hidden_sizes[i])
        policy = Policy (hidden_size = hidden_sizes[i],save = False)
        episodes, avgReturns = train(policy, env)
        plt.plot(episodes,avgReturns, label = legends[i], color = colors[i])
    plt.xlabel('episodes')
    plt.ylabel('avgReturns')
    plt.ylim((-1 ,1))
    plt.legend()
    plt.title('The relationship between episodes and the average return')
    plt.savefig('pictures/part5b.jpg')

def part5d(episode, fiveGames = True):
    ep = episode
    policy = Policy (hidden_size = size)
    load_weights(policy, ep)
    print(first_move_distr(policy, env))

    winCount = loseCount = 0
    games = [int(np.random.random()*100) for i in range (5)] #display 5 games

    for i in range (100):
        state = env.reset()
        done = False
        while not done:
            if i in games and fiveGames:
                print ('game', i)
                env.render()
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
        if status == 'win':
            winCount += 1
        if status == 'lose':
            loseCount += 1

    print ('\nOut of 100 games')
    print ('win:', winCount)
    print ('lose:', loseCount)
    print ('tie:', 100-winCount-loseCount)
    return float(winCount)/100

def part6():
    winRates = []
    eps = []
    for ep in range (1000,500000,10000):
        eps.append(ep)
        winRate = part5d(ep,fiveGames = False)
        winRates.append(winRate)
    plt.plot(eps,winRates,color = 'b')
    plt.xlabel('episodes')
    plt.ylabel('win rate')
    plt.ylim((0.5,1))
    plt.title('The relationship between episodes and win rate')
    plt.savefig('pictures/part6.jpg')

def part7(episode):
    load_weights(policy, episode)
    dist = first_move_distr(policy,env)
    print (np.array(dist[0]).reshape(3,3))

#************************************************************************

if __name__ == '__main__':
    import sys
    global size

    policy = Policy()
    env = Environment()

    reTrain = False

    size = 50
    if reTrain:
        part5a(size)
        #part5b() 
    else:
        # part5d(100000) 
        # part6()
        part7(500000)
