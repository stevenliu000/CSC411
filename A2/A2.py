from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat


##############################################################################
################################# Import Data ################################

'''
matrix format description
Captial letter indicates that they are matrix
X: n*784
Y: 10*n
output: 10*n
O: 10*n
b0: 10*1
W0: 784*10
'''

def importData():
	M = loadmat("mnist_all.mat")
	X_train = np.empty((0,784))
	Y_train = np.empty((10,0))
	X_test = np.empty((0,784))
	Y_test = np.empty((10,0))

	for i in range(0,10):
		name = "train" + str(i)
		X_train = np.vstack((X_train, M[name]/255.0))
		Y = np.zeros((10,M[name].shape[0]))
		Y[i,:] = 1
		Y_train = np.hstack((Y_train, Y))

		name = "test" + str(i)
		X_test = np.vstack((X_test, M[name]/255.0))
		Y = np.zeros((10,M[name].shape[0]))
		Y[i,:] = 1
		Y_test = np.hstack((Y_test, Y))

	return X_train, Y_train, X_test, Y_train

##############################################################################
#################################### Part1 ###################################

def part1():
	return

##############################################################################
#################################### Part2 ###################################

def softmax(Y):
	'''Return the output of the softmax function for the matrix of output y. y
	is an NxM matrix where N is the number of outputs for a single case, and M
	is the number of cases'''
	return exp(Y)/tile(sum(exp(Y),0), (len(Y),1))

def forward(X, W0b0):
	W0 = W0b0[:-1,:]
	b0 = W0b0[-1:,:].T
	O = dot(W0.T, X.T) + b0
	output = softmax(O)
	return O, output

##############################################################################
#################################### Part3 ###################################

def CostFunction(X, Y, W0b0):
	O, output = forward(X,W0b0)
	return -sum(Y*log(output))

def Gradient(X, Y, W0b0):
	'''
	Return Gradient of W0, Gradient of b0
	'''
	O, output = forward(X, W0b0)
	dy = output - Y
	return dot(dy, X).T, dot(dy, ones((X.shape[0],1)))

def finite_diff(CostFunction, X, Y, W0b0, row, col, h):
	'''
	function for calculating one component of gradient 
	using finite-difference approximation
	'''
	W0b0_h = np.copy(W0b0)
	W0b0_h[row,col] = W0b0_h[row,col] + h
	return (CostFunction(X, Y, W0b0_h) - CostFunction(X, Y, W0b0))/h

def Check_diff(X, Y, W0b0, row, col ,h, gradW0, gradb0):
	finiteDiff = finite_diff(CostFunction, X, Y, W0b0, row, col ,h)
	if row == 785:
		print('The difference on Gradient_of_b0[%i, 1] is %010.10f' %(col, \
			abs(finiteDiff - gradb0[col,1])))
	else:
		print('The difference on Gradient_of_W0[%i,%i] is %010.10f' %(row, \
			col, abs(finiteDiff - gradW0[row,col])))


def part3b():
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	row_ = np.random.randint(0,785,7) 
	#row = 785 indicates that we are checking for the gradient for b
	col_ = np.random.randint(0,10,7)
	h = 10**(-7)
	gradW0, gradb0 = Gradient(X_train, Y_train, W0b0)
	for i in range(7):
		Check_diff(X_train, Y_train, W0b0, row_[i], col_[i] ,h, \
			gradW0, gradb0)

##############################################################################
#################################### Part4 ###################################

def grad_descent(f, df, x, y, init_t, alpha, EPS=1e-5, max_iter=80000 \
	Plot = False):
	prev_t = init_t - 10 * EPS
	t = init_t.copy()
	iter = 0
	cost_func_ = []
	performanceTrain = []
	performanceTest = []
	while norm(t - prev_t) > EPS and iter < max_iter:
		costFunc = f(x, y, t)
		gradW0, gradb0 = df(x, y, t)
		grad = np.vstack((gradW0,gradb0.T))

		cost_func_.append(costFunc)
		performanceTrain.append(performance(x, y, W0b0))
		performanceTest.append(performance(X_test,Y_test, W0b0))
		prev_t = t.copy()
		t -= alpha * grad
		if iter % 500 == 0:
			print "Iter", iter
			print "Cost", costFunc
			print "Gradient: ", grad, "\n"
		iter += 1
		
	if Plot:
		fig = plt.figure(4)
		plt.title("Part4: Learning Rate")
		plt.xlabel('epoch')
		plt.ylabel('performance on datasets')
		plt.plot(range(iter),performanceTrain,'-1', label = 'train')
		plt.plot(range(iter),performanceTest,'-2', label = 'test')
		plt.legend(loc='lower right')
		fig.savefig(dirpath + '/part4_LearningRate.jpg')
		plt.show()

	return t, cost_func_

def performance(X, Y, W0b0):
	O, output = forward(X, Y, W0b0)
	cor = 0.
	for i in range(y.shape[1]):
		if y[np.argmax(output[:,i]),i] == 1:
			cor += 1.
	return cor/y.shape[1]

def part4():
	alpha = 1e-5
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	W0b0, cost_func_ = grad_descent(CostFunction, Gradient, X_train, Y_train, \
		W0b0, alpha)
	return W0b0, cost_func_

##############################################################################
#################################### Main ####################################

if __name__ == "__main__":
	os.chdir(os.path.dirname(__file__))
	dirpath = os.getcwd()
	X_train, Y_train, X_test, Y_train = importData()
	part3b()
	W0b0, cost_func_ = part4()