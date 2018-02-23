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

	return X_train, Y_train, X_test, Y_test

##############################################################################
#################################### Part1 ###################################

def part1():
	'''
	Visulazie 10 images of each of the digits, images are selected randomly
	'''
	np.random.seed(0)
	M = loadmat("mnist_all.mat")
	f, axarr = plt.subplots(10,10)
	
	for i in range(0,10): # Loop over 10 digits
		name_train = "train"+str(i)
		name_test = "test"+str(i)
		temp = np.vstack((M[name_train]/255.0,M[name_test]/255.0))
		np.random.shuffle(temp) # Random the pictures

		for j in range(0,10): #Produce all 10 pictures of each digits
			axarr[i,j].imshow(temp[j,:].reshape(28,28),cmap=cm.gray)
			axarr[i,j].axes.get_xaxis().set_visible(False)
			axarr[i,j].axes.get_yaxis().set_visible(False)
			
	plt.savefig('Part1.jpg')
	plt.show()
	return

##############################################################################
#################################### Part2 ###################################

def softmax(Y):
	'''
	Return the output of the softmax function for the matrix of output Y. Y
	is an NxM matrix where N is the number of outputs for a single case, and M
	is the number of cases
	'''
	return exp(Y)/tile(sum(exp(Y),0), (len(Y),1))

def forward(X, W0b0):
	'''
	Function takes in X, the input, which is n*784 matrix, n being the 
	and W0b0, the weights matrix, which is a 785*10 matrix. The first 784 rows 
	corresponds to the neuron weights, the last row is the bias.
	Function returns the Output neurons O and the output after taking the 
	softmax function. 
	Both of the outputs have dimension 10*n.
	'''
	W0 = W0b0[:-1,:]
	b0 = W0b0[-1:,:].T
	O = dot(W0.T, X.T) + b0
	output = softmax(O)
	return O, output

##############################################################################
#################################### Part3 ###################################

def CostFunction(X, Y, W0b0):
	'''
	Use Negative log-probabilities of all training cases as the cost function
	'''
	O, output = forward(X,W0b0)
	return -sum(Y*log(output))

def Gradient(X, Y, W0b0):
	'''
	Return Gradient of the cost function
	The first return matrix is the gradient w.r.t. the W0 matrix (weights), it has
	dimension 784 * 10
	The second return matrix is the gradient w.r.t. the b0 matrix (bias), it has
	dimension 10*1
	'''
	O, output = forward(X, W0b0)
	dy = output - Y
	return dot(dy, X).T, dot(dy, ones((X.shape[0],1)))

def finite_diff(CostFunction, X, Y, W0b0, row, col, h):
	'''
	Function for calculating one component of gradient using finite-difference 
	approximation. 
	h is the "small step" to take for the finite-difference
	row, col is the coordinate which that we wish to take the finite-differnce at
	'''
	W0b0_h = np.copy(W0b0)
	W0b0_h[row,col] = W0b0_h[row,col] + h
	return (CostFunction(X, Y, W0b0_h) - CostFunction(X, Y, W0b0))/h

def Check_diff(X, Y, W0b0, row, col ,h, gradW0, gradb0):
	'''
	This function prints the difference between gradient calculated using the
	finite difference method and vectorized gradient function
	'''
	finiteDiff = finite_diff(CostFunction, X, Y, W0b0, row, col ,h)
	if row == 784:
		print('The difference on Gradient_of_b0[%i, 0] is %010.10f' %(col, \
			abs(finiteDiff - gradb0[col,0])))
	else:
		print('The difference on Gradient_of_W0[%i,%i] is %010.10f' %(row, \
			col, abs(finiteDiff - gradW0[row,col])))

def part3b():
	'''
	Main function for part3b, in order to check the accuracy of the vectorized 
	gradient function by comparing to finite method
	7 Random points for each of the weights and the bias are selected from a 
	normal distribution of scale 0.0001.
	By setting row = 784, we are checking the gradients of the bias
	'''
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	row_ = np.random.randint(0,784,7) 
	#row = 784 # This indicates that we are checking for the gradient for b
	col_ = np.random.randint(0,10,7)
	h = 10**(-7)
	gradW0, gradb0 = Gradient(X_train, Y_train, W0b0)
	for i in range(7):
		Check_diff(X_train, Y_train, W0b0, row_[i], col_[i] ,h, \
			gradW0, gradb0)
		#Check_diff(X_train, Y_train, W0b0, row, col_[i] ,h, \
		#	gradW0, gradb0) #This indicates that we are checking for the gradint for b

##############################################################################
#################################### Part4 ###################################

def grad_descent(f, df, x, y, init_t, alpha, EPS=1e-5, max_iter=80000, \
 plotLR = False):
	'''
	This function uses gradient descent to update the weights on the neurons
	and the bias.
	t is the updated weights matrix of size 785x10
	cost_func_ is a list storing the cost for each iteration
	'''
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
		performanceTrain.append(performance(x, y, t))
		performanceTest.append(performance(X_test,Y_test, t))
		prev_t = t.copy()
		t -= alpha * grad
		#if iter % 500 == 0:
		if iter % 50 == 0:
			print "Iter", iter
			print "Cost", costFunc
			print "Gradient: ", grad, "\n"
		iter += 1
		
	if plotLR:
		fig = plt.figure(40)
		plt.title("Part4: Learning Rate")
		plt.xlabel('epoch')
		plt.ylabel('performance on datasets')
		plt.plot(range(iter),performanceTrain,'-1', label = 'train')
		plt.plot(range(iter),performanceTest,'-2', label = 'test')
		plt.legend(loc='lower right')
		fig.savefig(dirpath + '/part4_LearningRate.jpg')
		plt.show()

	return t, cost_func_, performanceTrain, performanceTest

def performance(X, Y, W0b0):
	'''
	This function returns the performance of the neural network by comparing its
	prediction with true label.
	'''
	O, output = forward(X, W0b0)
	cor = 0.
	for i in range(Y.shape[1]):
		if Y[np.argmax(output[:,i]),i] == 1:
			cor += 1.
	return cor/Y.shape[1]

def part4_alpha():
	'''
	This function plots cost function with different alpha, and thus we can choose
	the alpha that gives us the lowest cost
	'''
	# alpha below 1e-4 makes the cost function unbounded, blows up
	alpha_list = [1e-5, 1e-6, 5e-6, 1e-7, 1e-8]
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	cost_function = []
	max_iter = 100
	for alpha in alpha_list:
		W0b0_part4, cost_func_ind = grad_descent(CostFunction, Gradient, X_train, Y_train, \
			W0b0, alpha, plotLR = False, max_iter = 100)
		cost_function.append(cost_func_ind)
	
	# Plot
	fig = plt.figure(41)
	plt.title("Part4: Picking Alpha")
	plt.xlabel('epoch')
	plt.ylabel('cost')
	for index in range(0, len(alpha_list)):
		plt.plot(range(max_iter),cost_function[index], \
			label = 'alpha = ' + str(alpha_list[index]))
	plt.legend(loc='lower right')
	fig.savefig(dirpath + '/part4_PickAlpha.jpg')
	plt.show()
	return
	

def part4():
	'''
	Train the neural networking using gradient desecent.
	The initial weights is set to be a scaled standard normal with scale 0.0001
	The alpha is set to be 1e-5
	Maximum iteration is set to be 1500
	'''
	alpha = 1e-5
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	W0b0_part4, cost_func_part4, performanceTrain_part4, performanceTest_part4 = \
		grad_descent(CostFunction, Gradient, X_train, Y_train, W0b0, alpha, plotLR \
		= True, max_iter = 1500)

	for i in range(10):
		fig = plt.figure(i)
		plt.title("Part4: Image of %i" %(i))
		plt.imshow(W0b0_part4[:-1,i].reshape(28,28))
		fig.savefig(dirpath + '/part4_' + str(i) + '.jpg')
		plt.show()

	return W0b0_part4, cost_func_part4, performanceTrain_part4, performanceTest_part4

##############################################################################
#################################### Part5 ###################################

def grad_descent_part5(f, df, x, y, init_t, alpha, EPS=1e-5, max_iter=80000, \
 plotLR = False, gamma = 0.9):
	'''
	This function is same as the grad_descent function in part 4, except we update
	the weights with momentum term gamma.
	'''
	prev_t = init_t - 10 * EPS
	t = init_t.copy()
	iter = 0
	v = 0
	cost_func_ = []
	performanceTrain = []
	performanceTest = []
	while norm(t - prev_t) > EPS and iter < max_iter:
		costFunc = f(x, y, t)
		gradW0, gradb0 = df(x, y, t)
		grad = np.vstack((gradW0,gradb0.T))

		cost_func_.append(costFunc)
		performanceTrain.append(performance(x, y, t))
		performanceTest.append(performance(X_test,Y_test, t))
		prev_t = t.copy()
		v = gamma*v + alpha*grad
		t -= v
		if iter % 100 == 0:
			print "Iter", iter
			print "Cost", costFunc
			print "Gradient: ", grad, "\n"
		iter += 1
		
	if plotLR:
		fig = plt.figure(50)
		plt.title("Part5: Learning Rate")
		plt.xlabel('epoch')
		plt.ylabel('performance on datasets')
		plt.plot(range(iter),performanceTrain,'-1', label = 'train')
		plt.plot(range(iter),performanceTest,'-2', label = 'test')
		plt.legend(loc='lower right')
		fig.savefig(dirpath + '/part5_LearningRate.jpg')
		plt.show()

	return t, cost_func_, performanceTrain, performanceTest

def part5_compare(train, test, train_m, test_m):
	'''
	Compare the performance of neural network with and without momentum
	'''
	iter = 200
	fig = plt.figure(51)
	plt.title("Part5: Compare learning rate")
	plt.xlabel('epoch')
	plt.ylabel('performance on datasets')
	plt.plot(range(iter),train[0:200], '-1', label = 'train')
	plt.plot(range(iter),test[0:200], '-2', label = 'test')
	plt.plot(range(iter),train_m[0:200], '-3', label = 'train with momentum')
	plt.plot(range(iter),test_m[0:200], '-4', label = 'test with momentum')
	plt.legend(loc='lower right')
	fig.savefig(dirpath + '/part5_CompareLearn.jpg')
	plt.show()
	
def part5():
	'''
	Train the neural networking using gradient desecent.
	The initial weights is set to be a scaled standard normal with scale 0.0001
	The alpha is set to be 1e-5
	Maximum iteration is set to be 1500
	'''
	alpha = 1e-5
	np.random.seed(1)
	W0 = np.random.normal(scale = 0.0001, size = (784,10))
	b0 = np.random.normal(scale = 0.0001, size = (10,1))
	W0b0 = np.vstack((W0, b0.T))
	W0b0_part5, cost_func_part5, performanceTrain_part5, performanceTest_part5 = \
		grad_descent_part5(CostFunction, Gradient, X_train, Y_train, W0b0, alpha, \
		plotLR = True, max_iter = 1500)

	return W0b0_part5, cost_func_part5,	performanceTrain_part5, performanceTest_part5

##############################################################################
#################################### Part6 ###################################

def Gradient_part6(X, Y, W0b0, w1r, w1c, w2r, w2c):
	'''
	Return Gradient on w1 and w2, all other weights are kept constant
	'''
	O, output = forward(X, W0b0)
	dy = output - Y
	return dot(dy[w1c,:], X[:,w1r]), dot(dy[w2c,:], X[:,w2r])

def CostFunction_part6(X, Y, W0b0, w1, w2, w1r, w1c, w2r, w2c):
	'''
	Compute the new cost of the neural networking after sub in the new w1 and w2
	'''
	W0b0Modified = W0b0.copy()
	W0b0Modified[w1r,w1c] = w1
	W0b0Modified[w2r,w2c] = w2
	O, output = forward(X,W0b0Modified)
	return -sum(Y*log(output))

def grad_descent_part6(f, df, x, y, init_t, alpha, w1r, w1c, w2r, w2c, \
	EPS=1e-5, max_iter=80000, ifmomentum = False, gamma = 0.9):
	'''
	Perform gradient descent on the weights w1 and w2 only
	traj_ is the list of turples which stores w1 and w2 value for each iteration
	'''
	prev_t = init_t - 10 * EPS
	t = init_t.copy()
	iter = 0
	v = 0
	cost_func_ = []
	traj_ = []
	while norm(t - prev_t) > EPS and iter < max_iter:
		traj_.append((t[w1r,w1c],t[w2r,w2c]))
		costFunc = f(x, y, t)
		gradw1, gradw2 = df(x, y, t, w1r, w1c, w2r, w2c)
		grad = np.array([gradw1, gradw2])
		cost_func_.append(costFunc)
		prev_t = t.copy()
		if ifmomentum:
			v = gamma*v + alpha*grad
		else:
			v = alpha * grad
		t[w1r,w1c] -= v[0]
		t[w2r,w2c] -= v[1]
		if iter % 500 == 0:
			print "Iter", iter
			print "Cost", costFunc
			print "Gradient: ", grad, "\n"
		iter += 1

	return traj_

def part6(w1r,w1c,w2r,w2c,w1i,w2i,alpha1,alpha2):
	'''
	w1r: the row index of the changing weight 1
	w1c: the column index of the changing weight 1
	w2r: the row index of the changing weight 2
	w2c: the column index of the changing weight 2
	w1i: the re-initialized value for weight 1
	w2i: the re-initialized value for weight 2
	alpha1: the learning rate for training without momentum
	alpha2: the leanring rate for training with momentum
	'''
	# contrust the contour variables
	w1_ = np.arange(-2, 2, 0.5)
	w2_ = np.arange(-2, 2, 0.5)
	W1_, W2_ = np.meshgrid(w1_, w2_)
	CostFunc = np.zeros([w1_.size, w2_.size])
	if os.path.isfile(dirpath+"/CostFunc_%i_%i_%i_%i.npy"%(w1r,w1c,w2r,w2c)):
		CostFunc = np.load(dirpath+"/CostFunc_%i_%i_%i_%i.npy"%(w1r,w1c,w2r,w2c))
	else:
		for i, w1 in enumerate(w1_):
			for j, w2 in enumerate(w2_):
				CostFunc[j,i] = CostFunction_part6(X_train, Y_train, \
					W0b0_part5,w1, w2, w1r, w1c, w2r, w2c)
				np.save(dirpath+"/CostFunc_%i_%i_%i_%i"%(w1r,w1c,w2r,w2c), CostFunc)

	# # part 6a) Plot the cost function contour plot
	# fig = plt.figure(60)
	# CS = plt.contour(W1_, W2_, CostFunc, cmap = cm.coolwarm)
	# plt.clabel(CS)
	# plt.title('Contour plot for cost function by changing w1 and w2')
	# plt.xlabel('w1: W[%i,%i]' %(w1r,w1c))
	# plt.ylabel('w2: W[%i,%i]' %(w2r,w2c))
	# fig.savefig(dirpath + '/part6_a.jpg')
	# plt.show()
	
	# perform gradient descents with and without momentum
	W0b0_temp = W0b0_part5.copy()
	W0b0_temp[w1r,w1c] = w1i
	W0b0_temp[w2r,w2c] = w2i
	gd_traj = grad_descent_part6(CostFunction, Gradient_part6, X_train, Y_train, W0b0_temp, alpha1 , w1r, w1c, w2r, w2c, max_iter=10)
	W0b0_temp = W0b0_part5.copy()
	W0b0_temp[w1r,w1c] = w1i
	W0b0_temp[w2r,w2c] = w2i
	mo_traj = grad_descent_part6(CostFunction, Gradient_part6, X_train, Y_train, W0b0_temp, alpha2 , w1r, w1c, w2r, w2c, max_iter=10, ifmomentum =True)

	# plot
	fig = plt.figure(61)
	CS = plt.contour(W1_, W2_, CostFunc, cmap = cm.coolwarm)
	plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
	plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
	plt.legend(loc='top left')
	plt.clabel(CS)
	plt.title('alpha without mom.:%07.7f, alpha with mom.:%07.7f'%(alpha1,alpha2))
	plt.xlabel('w1: W[%i,%i]' %(w1r,w1c))
	plt.ylabel('w2: W[%i,%i]' %(w2r,w2c))
	plt.show()
	fig.savefig(dirpath + '/part6_%i_%i_%i_%i.jpg'%(w1r,w1c,w2r,w2c))
	return

##############################################################################
#################################### Main ####################################

if __name__ == "__main__":
	os.chdir(os.path.dirname(__file__))
	dirpath = os.getcwd()
	X_train, Y_train, X_test, Y_test = importData()
	# part1()
	# part3b()
	
	# part4_alpha()
	# W0b0_part4, cost_func_part4, performanceTrain_part4, performanceTest_part4 = part4()
	# np.save(dirpath+'/W0b0_part4',W0b0_part4)
	# np.save(dirpath+'/performanceTrain_part4',W0b0_part4)
	# np.save(dirpath+'/performanceTest_part4',W0b0_part4)
	# W0b0_part4 = np.load(dirpath+"/W0b0_part4.npy")
	
	# W0b0_part5, cost_func_part5, performanceTrain_part5, performanceTest_part5 = part5()
	# np.save(dirpath+'/W0b0_part5',W0b0_part5)
	# np.save(dirpath+'/performanceTrain_part5',W0b0_part4)
	# np.save(dirpath+'/performanceTest_part5',W0b0_part4)
	W0b0_part5 = np.load(dirpath+"/W0b0_part5.npy")
	
	# part5_compare(performanceTrain_part4, performanceTest_part4, \
	# 				performanceTrain_part5, performanceTest_part5)
	
	# part6(300,2, 407,2,-2, 0, 4.8e-3, 4e-4) #part 6b) and c)
	# part6(100,2, 600,2,-0.5,-2, 2e-3, 2e-4) #overshoot, part6 e)
