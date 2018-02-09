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
def importData():
	M = loadmat("mnist_all.mat")
	X_train = np.empty((0,784))
	Y_train = np.empty((10,0))
	X_test = np.empty((0,784))
	Y_test = np.empty((10,0))

	for i in range(0,10):
		name = "train" + str(i)
		X_train = np.hstack(X_train, M[name]/255.0)
		Y = np.zeros((10,M[name].shape[0]))
		Y[i,:] = 1
		Y_train = np.vstack(Y_train, Y)

		name = "test" + str(i)
		X_test = np.vstack(X_test, M[name]/255.0)
		Y = np.zeros((10,M[name].shape[0]))
		Y[i,:] = 1
		Y_test = np.vstack(Y_test, Y)

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
    return exp(Y)/tile(sum(exp(y),0), (len(y),1))

def forward(X, W0, b0):
	O = dot(W0.T, X) + b0
    output = softmax(O)
    return O, output

##############################################################################
#################################### Part3 ###################################

def CostFunction(X, Y, W0, b0):
	'''
	X is the matrix that contains data (in the form of [X_1, X_2, ....]),
	Y is a matrix that contains the corresponding labels 
	(in the form of [Y^1, Y^2, ....])
	'''
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

def finite_diff(f, x, y, theta, row, col, h):
    #function for calculating one component of gradient 
    #using finite-difference approximation
    theta_h = np.copy(theta)
    theta_h[row,col] = theta_h[row,col] + h
    return (f(x, y ,theta_h) - f(x, y, theta))/h
##############################################################################
#################################### Part4 ###################################



if __name__ == "__main__":
	X_train, Y_train, X_test, Y_train = importData()
