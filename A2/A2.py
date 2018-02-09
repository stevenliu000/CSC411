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

def forward(X, W0, b0):
    O = dot(W0.T, X.T) + b0
    output = softmax(O)
    return O, output

##############################################################################
#################################### Part3 ###################################

def CostFunction(X, Y, W0, b0):
	result = 0
	output = forward(X, W0, b0)
	output = log(output)
	Y = Y.T
	for i in range(Y.shape[0]):
		result += -dot(Y[i,:], P[:,i])
	return result

def Gradient(X, Y, W0, b0):
	O, output = forward(X, W0, b0)
	return dot((output - Y), X)

def finite_diff(CostFunction, X, Y, W0, b0, row, col, h):
    #function for calculating one component of gradient 
    #using finite-difference approximation
    W0_h = np.copy(W0)
    W0_h[row,col] = W0_h[row,col] + h
    return (CostFunction(Y, X, W0_h) - CostFunction(X, Y, W0))/h

def part3b():
    np.random.seed(1)
    W0 = np.random.normal(scale = 0.0001, size = (784,10))
    b0 = np.random.normal(scale = 0.0001, size = (10,1))
    W0_h = np.copy(W0)
    row_ = np.random.randint(0,10,7)
    col_ = np.random.randint(0,784,7)
    h = 10**(-6)
    print row_
    print col_
    gradient = Gradient(X_train, Y_train, W0, b0)
    for i in range(7):
    	print row_[i], col_[i]
        print(abs(finite_diff(CostFunction, X_train, Y_train, W0, b0, row_[i],\
        	col_[i],h) - gradient[row_[i], col_[i]]))

##############################################################################
#################################### Part4 ###################################

def part4():
	return

##############################################################################
#################################### Main ####################################

if __name__ == "__main__":
	X_train, Y_train, X_test, Y_train = importData()
	part3b()
