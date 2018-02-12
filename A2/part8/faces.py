from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part1
def rgb2gray(rgb):
	'''Return the grayscale version of the RGB image rgb as a 2D numpy array
	whose range is 0..1
	Arguments:
	rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
	range of the values is 0..255
	'''

	r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

	return gray / 255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
	'''From:
	http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
	import threading
	class InterruptableThread(threading.Thread):
		def __init__(self):
			threading.Thread.__init__(self)
			self.result = None

		def run(self):
			try:
				self.result = func(*args, **kwargs)
			except:
				self.result = default

	it = InterruptableThread()
	it.start()
	it.join(timeout_duration)
	if it.isAlive():
		return False
	else:
		return it.result



def part1():
	for act_type in ['actors', 'actresses']:
		faces_subset = dirpath + "/facescrub_" + act_type + ".txt"
		uncropped_path = dirpath + "/facescrub_" + act_type + "_uncropped/"
		cropped_path = dirpath + "/facescrub_" + act_type + "_cropped/"
		if os.path.isdir(uncropped_path):
			os.unlink(uncropped_path)
		else:
			os.makedirs(uncropped_path)

		if os.path.isdir(cropped_path):
			os.unlink(cropped_path)
		else:
			os.makedirs(cropped_path)

		testfile = urllib.URLopener()

		act = list(set([a.split("\t")[0] for a in open(faces_subset).readlines()]))

		# Note: you need to create the uncropped folder first in order
		# for this to work

		for a in act:
			# name = a.split()[1].lower()
			name = a
			i = 0
			for line in open(faces_subset):
				if a in line:
					filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
					# A version without timeout (uncomment in case you need to
					# unsupress exceptions, which timeout() does)
					# testfile.retrieve(line.split()[4], "uncropped/"+filename)
					# timeout is used to stop downloading images which take too long to download
					timeout(testfile.retrieve, (line.split()[4], uncropped_path + filename), {}, 5)
					if not os.path.isfile(uncropped_path + filename):
						continue

					try:
						uncropped = imread(uncropped_path + filename)
						crop_coor = line.split("\t")[4].split(",")
						if (uncropped.ndim) == 3:
							cropped = uncropped[int(crop_coor[1]):int(crop_coor[3]),
									  int(crop_coor[0]):int(crop_coor[2]), :]
							resized = imresize(cropped, (32, 32))
							grayed = rgb2gray(resized)
						elif (uncropped.ndim) == 2:
							cropped = uncropped[int(crop_coor[1]):int(crop_coor[3]),
									  int(crop_coor[0]):int(crop_coor[2])]
							resized = imresize(cropped, (32, 32))
							grayed = resized / 255.
					except:
						continue



					imsave(cropped_path + filename, grayed, cmap = cm.gray)

					print filename
					i += 1
		return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part2
def get_names():
	NameList = []

	for i in ['actors', 'actresses']:
		act_type = i
		dataset_path = dirpath + "/facescrub_" + act_type + "_cropped/"
		for root, dirs, files in os.walk(dataset_path):
			for filename in files:
				temp = filename.rfind('.') - 1
				while filename[temp].isdigit():
					temp -= 1
				if filename[:temp + 1] not in NameList:
					NameList.append(filename[:temp + 1])

	return NameList


def part2(s = 0):
	act = ['Alec Baldwin', 'Bill Hader', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', \
		   'Steve Carell', 'America Ferrera', 'Angie Harmon', 'Fran Drescher', 'Kristin Chenoweth', \
		   'Lorraine Bracco', 'Peri Gilpin']
	act_rawdata = {}
	act_data = {}
	for i in act:
		act_rawdata[i] = np.empty((1024,0))
		act_data[i] = [0, 0, 0]

	for j in ['actors', 'actresses']:
		act_type = j
		dataset_path = dirpath + "/facescrub_" + act_type + "_cropped/"
		for root, dirs, files in os.walk(dataset_path):
			dirs.sort()
			files.sort()
			for filename in files:
				im = imread(dataset_path + filename)
				im = im/255.
				im = np.array([im.flatten()]).T
				if np.amax(im) > 1.0 or np.amin(im) < 0. or np.isnan(im).any() or np.isinf(im).any():
					continue
				else:
					for i in act:
						if i in filename:
							act_rawdata[i] = np.hstack((act_rawdata[i], im))

	# randomly shuffle act_rawdata
	np.random.seed(s)
	for i in act:
		act_rawdata[i] = act_rawdata[i][:,np.random.permutation(act_rawdata[i].shape[1])]

	# act_data is a dictionary whose key is the name of actor/actress, value is the 3-element long array that the first
	# element is a array that contains the training set, second one contains validation set, third one
	# contains test set.
	for i in act:
		act_data[i][0] = act_rawdata[i][:, :min(70, act_rawdata[i].shape[1] - 20)]
		act_data[i][1] = act_rawdata[i][:, -20:-10]
		act_data[i][2] = act_rawdata[i][:, -10:]
	return act_data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part3
def f(x, y, theta):
	x = vstack((ones((1, x.shape[1])), x))
	return sum((y - dot(theta.T, x)) ** 2)


def df(x, y, theta):
	x = vstack((ones((1, x.shape[1])), x))
	return -2 * sum((y - dot(theta.T, x)) * x, 1)


def performance(X, Y, theta):
	X = vstack((np.ones((1, X.shape[1])), X))
	h = dot(theta.T, X)
	cor = 0
	for i in range(len(Y)):
		if Y[i] == 1 and h[i] > 0:
			cor += 1
		elif Y[i] == -1 and h[i] < 0:
			cor += 1
	return float(cor) / len(Y)


def grad_descent(f, df, x, y, init_t, alpha, EPS=1e-5, max_iter=80000):
	prev_t = init_t - 10 * EPS
	t = init_t.copy()
	iter = 0
	cost_func_ = []
	while norm(t - prev_t) > EPS and iter < max_iter:
		cost_func_.append(f(x,y,t))
		prev_t = t.copy()
		t -= alpha * df(x, y, t)
		if iter % 500 == 0:
			print "Iter", iter
			print "Cost", f(x, y, t)
			print "Gradient: ", df(x, y, t), "\n"
		iter += 1
		
	return t, cost_func_

def part3(alpha= 0.000010, st_devi = 0, max_iteration = 80000):
	X_train = np.hstack((act_data['Alec Baldwin'][0], act_data['Steve Carell'][0]))
	Y_train = np.append(np.ones(act_data['Alec Baldwin'][0].shape[1]),
						np.full(act_data['Steve Carell'][0].shape[1], -1))
	X_vali = np.hstack((act_data['Alec Baldwin'][1], act_data['Steve Carell'][1]))
	Y_vali = np.append(np.ones(10), np.full(10, -1))
	X_test = np.hstack((act_data['Alec Baldwin'][2], act_data['Steve Carell'][2]))
	Y_test = np.append(np.ones(10), np.full(10, -1))

	np.random.seed(0)
	theta0 = np.random.normal(scale=st_devi, size=1025)
	theta, cost_func_ = grad_descent(f, df, X_train, Y_train, theta0, alpha, max_iter = max_iteration)

	print("This is the result for part3")
	print("Cost function on training set:", f(X_train, Y_train, theta), "Normalized:", f(X_train, Y_train, theta)/(Y_train.shape[0]))
	print("Cost function on validation set:", f(X_vali, Y_vali, theta), "Normalized:", f(X_vali, Y_vali, theta)/(Y_vali.shape[0]))
	print("Performance on training set", performance(X_train, Y_train, theta))
	print("Performance on validation set", performance(X_vali, Y_vali, theta))

	return theta

def part3_alpha():
	X_train = np.hstack((act_data['Alec Baldwin'][0], act_data['Steve Carell'][0]))
	Y_train = np.append(np.ones(act_data['Alec Baldwin'][0].shape[1]),
						np.full(act_data['Steve Carell'][0].shape[1], -1))
	X_vali = np.hstack((act_data['Alec Baldwin'][1], act_data['Steve Carell'][1]))
	Y_vali = np.append(np.ones(10), np.full(10, -1))
	X_test = np.hstack((act_data['Alec Baldwin'][2], act_data['Steve Carell'][2]))
	Y_test = np.append(np.ones(10), np.full(10, -1))

	fig = plt.figure(31)
	alpha_ = [1e-7, 1e-6, 1e-5, 2.1e-5]
	for j, i in zip(range(len(alpha_)), alpha_):
		np.random.seed(0)
		theta0 = np.random.normal(scale=0, size=1025)
		theta, cost_func_ = grad_descent(f, df, X_train, Y_train, theta0, i, max_iter = 30)
		plt.plot(range(0,30), cost_func_, "-%i"%j, label = "alpha = %010.10f"%i)
	plt.xlabel("number of iterations")
	plt.ylabel('cost function')
	plt.legend(loc = "best")

	fig.savefig(dirpath + '/part3_1.jpg')
	plt.show()
	return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part4

def part4a():
	# obtain theta by using only two images of each actor
	theta0 = np.random.normal(scale=0.00, size=1025)
	X_train_2 = np.hstack((act_data['Alec Baldwin'][0][:, :2], act_data['Steve Carell'][0][:, :2]))
	Y_train_2 = np.append(np.ones(2), np.full(2, -1))
	theta_2, cost_func_ = grad_descent(f, df, X_train_2, Y_train_2, theta0, 0.00000002, EPS=1e-7)

	fig = plt.figure(40)
	st = fig.suptitle('Result of part4a', fontsize="x-large")

	theta_reshaped_full = np.reshape(theta_part3[1:], (32, 32))
	plt.subplot(121)
	plt.imshow(theta_reshaped_full, cmap='RdBu')
	plt.xlabel('Full training set', fontsize="large")

	theta_reshaped_2 = np.reshape(theta_2[1:], (32, 32))
	plt.subplot(122)
	plt.imshow(theta_reshaped_2, cmap='RdBu')
	plt.xlabel('Two images of each actor', fontsize="large")
	st.set_y(0.85)

	fig.savefig(dirpath + '/result_of_part4a.jpg')
	plt.show()
	return

def part4b_helper(st_devi, max_iteration):
	theta_part4b = part3(st_devi = st_devi, max_iteration = max_iteration)
	fig = plt.figure(41)
	tit = 'std_deviation =' + str(st_devi) + ', max_itera =' + str(max_iteration)
	st = fig.suptitle(tit, fontsize="x-large")
	theta_reshaped_full = np.reshape(theta_part4b[1:], (32, 32))
	plt.imshow(theta_reshaped_full, cmap='RdBu')
	plt.xlabel('Full training set', fontsize="large")

	fig.savefig(dirpath + '/part4b_std_dev_' + str(st_devi) + '_max_itera_' \
				+ str(max_iteration) +'_.jpg')
	plt.show()
	return

def part4b():
	part4b_helper(1e-5,80)
	part4b_helper(0.01, 80)
	part4b_helper(0.001, 80000)
	part4b_helper(0.0001, 80000)
	part4b_helper(0, 80000)

	return
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part5

def part5_in_act():
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	gender1 = [-1, -1, -1, 1, 1, 1]  # -1 means female, 1 means male
	np.random.seed(1)
	X_vali1 = np.empty((1024, 0))
	Y_vali1 = np.array([])
	X_test1 = np.empty((1024, 0))
	Y_test1 = np.array([])
	theta_ = np.empty((1025, 0))
	perform_test1 = []
	perform_vali1 = []
	perform_train = []
	size_ = []

	for i in range(len(act1)):
		X_vali1 = np.hstack((X_vali1, act_data[act1[i]][1]))
		Y_vali1 = np.append(Y_vali1, np.full(10, gender1[i]))
		X_test1 = np.hstack((X_test1, act_data[act1[i]][2]))
		Y_test1 = np.append(Y_test1, np.full(10, gender1[i]))

	for percentage in np.arange(0.1, 1, 0.1):
		print percentage
		X_train = np.empty((1024, 0))
		size = 0
		Y_train = np.array([])
		for i in range(len(act1)):
			num = int(percentage * act_data[act1[i]][0].shape[1])
			size += num
			X_train = np.hstack((X_train, act_data[act1[i]][0][:, :num]))
			Y_train = np.append(Y_train, np.full(num, gender1[i]))

		size_.append(size)
		theta0 = np.random.normal(loc=0, scale=0, size=1025)
		theta, cost_func_ = grad_descent(f, df, X_train, Y_train, theta0, 0.000001, max_iter=50000)
		theta_ = np.hstack((theta_, np.array([theta]).T))
		perform_test1.append(performance(X_test1, Y_test1, theta))
		perform_vali1.append(performance(X_vali1, Y_vali1, theta))
		perform_train.append(performance(X_train, Y_train, theta))

	return theta_, perform_test1, perform_vali1, perform_train, size_


def part5_not_in_act(theta_):
	act = get_names()
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	act2 = []
	for i in act:
		if i not in act1:
			act2.append(i)

	gender2 = np.full(len(act2), -1)
	faces_subset = dirpath + "/facescrub_" + 'actors' + ".txt"
	temp = open(faces_subset, 'r').read()
	for i in range(len(act2)):
		if act2[i] in temp:
			gender2[i] = 1

	performance_vali2 = []
	performance_test2 = []
	X_vali2 = np.empty((1024, 0), float)
	Y_vali2 = np.array([])
	X_test2 = np.empty((1024, 0), float)
	Y_test2 = np.array([])

	for j in range(len(act2)):
		X_vali2 = np.hstack((X_vali2, act_data[act2[j]][1]))
		Y_vali2 = np.append(Y_vali2, np.full(10, gender2[j]))

		X_test2 = np.hstack((X_test2, act_data[act2[j]][2]))
		Y_test2 = np.append(Y_test2, np.full(10, gender2[j]))

	for i in range(theta_.shape[1]):
		performance_vali2.append(performance(X_vali2, Y_vali2, theta_[:, i]))
		performance_test2.append(performance(X_test2, Y_test2, theta_[:, i]))

	return performance_vali2, performance_test2

def part5():
	theta_, perform_test1, perform_vali1, perform_train, size_ = part5_in_act()
	perform_vali2, performa_test2 = part5_not_in_act(theta_)
	fig =plt.figure(50)
	plt.plot(size_, perform_train, '-1', label='training set')
	plt.plot(size_, perform_vali1, '-2', label='validation set for actors in act')
	plt.plot(size_, perform_vali2, '-3', label='validation set for actors not in act')
	plt.legend(loc='lower right')
	plt.title('Part5: Performance on different datasets')

	fig.savefig(dirpath + '/result_of_part5.jpg')
	plt.show()
	return theta_

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part6

# part6c
def f_multi(x, y, theta):
	x = np.vstack( (np.ones((1, x.shape[1])), x))
	return np.sum( (y - np.dot(theta.T,x)) ** 2)

def df_multi(x, y, theta):
	x = np.vstack( (np.ones((1, x.shape[1])), x))
	return 2 * np.dot(x, (np.dot(theta.T,x) - y).T)

def performance_multi(x, y, theta):
	x = np.vstack( (np.ones((1, x.shape[1])), x))
	h = np.dot(theta.T,x)
	cor = 0.
	for i in range(y.shape[1]):
		if y[np.argmax(h[:,i]),i] == 1:
			cor += 1.
	return cor/y.shape[1]


#part6d
def finite_diff(f, x, y, theta, row, col, h):
	#function for calculating one component of gradient 
	#using finite-difference approximation
	theta_h = np.copy(theta)
	theta_h[row,col] = theta_h[row,col] + h
	return (f(x, y ,theta_h) - f(x, y, theta))/h
	
def part6d():
	#construct training set
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

	X_train = np.empty((1024, 0))
	Y_train = np.empty((len(act1), 0))
	for i in range(len(act1)):
		temp = np.zeros((len(act1), act_data[act1[i]][0].shape[1]))
		temp[i, :] = 1
		X_train = np.hstack((X_train, act_data[act1[i]][0]))
		Y_train = np.hstack((Y_train, temp))

	#run 50 iterations of gradient descent
	np.random.seed(1)
	theta0 = np.random.normal(scale=0, size=(1025, 6))
	theta, cost_func_ = grad_descent(f_multi, df_multi, X_train, Y_train, theta0, 0.000001, max_iter=50)
	gradient = df_multi(X_train, Y_train, theta)
	
	#pick 5 random coordinates
	np.random.seed(1)
	row_ = np.random.randint(0,1026,5)
	col_ = np.random.randint(0,7,5)
	h_ = (10**exp for exp in range(-11,-5))
	error_ = []
	hh =  []
	for h in h_:
		hh.append(h)
		temp = 0
		for i in range(5):
			temp += abs(finite_diff(f_multi, X_train, Y_train, theta, row_[i],\
									col_[i],h) - gradient[row_[i], col_[i]])
		error_.append(temp/5.)
	
	#save figure
	fig = plt.figure(6)
	plt.title('Average difference over 5 coordinates vs. h')
	plt.xlabel('h')
	plt.ylabel("Average difference over 5 coordinates")
	plt.loglog(hh,error_,'-1')

	fig.savefig(dirpath + '/part6d_1.jpg')
	plt.show()

	#print out result
	for i in range(5):
		print("Difference in gradient[%i, %i] is %010.10f" %(row_[i], col_[i] \
			  ,abs(finite_diff(f_multi, X_train, Y_train, theta, row_[i],\
								col_[i],10**(-8)) - gradient[row_[i], col_[i]])))
		 
	return
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# part7


def part7():
	# part7
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	act_data = part2(2)
	X_train = np.empty((1024, 0))
	Y_train = np.empty((len(act1), 0))
	X_vali = np.empty((1024, 0))
	Y_vali = np.empty((len(act1), 0))
	X_test = np.empty((1024, 0))
	Y_test = np.empty((len(act1), 0))
	for i in range(len(act1)):
		temp = np.zeros((len(act1), act_data[act1[i]][0].shape[1]))
		temp[i, :] = 1
		X_train = np.hstack((X_train, act_data[act1[i]][0]))
		Y_train = np.hstack((Y_train, temp))

		temp = np.zeros((len(act1), act_data[act1[i]][1].shape[1]))
		temp[i, :] = 1
		X_vali = np.hstack((X_vali, act_data[act1[i]][1]))
		Y_vali = np.hstack((Y_vali, temp))
		
		temp = np.zeros((len(act1), act_data[act1[i]][2].shape[1]))
		temp[i, :] = 1
		X_test = np.hstack((X_test, act_data[act1[i]][2]))
		Y_test = np.hstack((Y_test, temp))
		
	np.random.seed(1)
	theta0 = np.random.normal(scale=0, size=(1025, 6))
	theta, cost_func_ = grad_descent(f_multi, df_multi, X_train, Y_train, theta0, 0.000004, max_iter=10000)

	print("this is the result for part7")
	print("Cost function on training set:", f_multi(X_train, Y_train, theta),"Normalized:", f_multi(X_train, Y_train, theta)/(Y_train.shape[0]))
	print("Cost function on validation set:", f_multi(X_vali, Y_vali, theta),"Normalized:", f_multi(X_vali, Y_vali, theta)/(Y_vali.shape[0]))
	print("Cost function on validation set:", f_multi(X_test, Y_test, theta),"Normalized:", f_multi(X_test, Y_test, theta)/(Y_test.shape[0]))
	print("Performance on training set", performance_multi(X_train, Y_train, theta))
	print("Performance on validation set", performance_multi(X_vali, Y_vali, theta))
	print("Performance on test set", performance_multi(X_test, Y_test, theta))

	return theta

def part7_alpha():
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

	X_train = np.empty((1024, 0))
	Y_train = np.empty((len(act1), 0))
	for i in range(len(act1)):
		temp = np.zeros((len(act1), act_data[act1[i]][0].shape[1]))
		temp[i, :] = 1
		X_train = np.hstack((X_train, act_data[act1[i]][0]))
		Y_train = np.hstack((Y_train, temp))

	fig = plt.figure(71)
	alpha_ = [5e-7, 1e-6, 4e-6, 7.2e-6]
	for j, i in zip(range(len(alpha_)), alpha_):
		np.random.seed(0)
		theta0 = np.random.normal(scale=0, size=(1025, 6))
		theta, cost_func_ = grad_descent(f_multi, df_multi, X_train, Y_train, theta0, i, max_iter=30)
		plt.plot(range(0,30), cost_func_, "-%i"%j, label = "alpha = %010.10f"%i)
		plt.xlabel("number of iterations")
		plt.ylabel('cost function')

	plt.legend(loc = "upper right")
	fig.savefig(dirpath + '/part7_1.jpg')
	plt.show()
	return

def part7_itera():
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	act_data = part2(2)
	X_train = np.empty((1024, 0))
	Y_train = np.empty((len(act1), 0))
	X_vali = np.empty((1024, 0))
	Y_vali = np.empty((len(act1), 0))
	for i in range(len(act1)):
		temp = np.zeros((len(act1), act_data[act1[i]][0].shape[1]))
		temp[i, :] = 1
		X_train = np.hstack((X_train, act_data[act1[i]][0]))
		Y_train = np.hstack((Y_train, temp))

		temp = np.zeros((len(act1), act_data[act1[i]][2].shape[1]))
		temp[i, :] = 1
		X_vali = np.hstack((X_vali, act_data[act1[i]][2]))
		Y_vali = np.hstack((Y_vali, temp))
	
	itera_ = [10,100,1000,5000,7000,10000,12000,15000,20000,30000]
	perf_ = []
	fig = plt.figure(72)
	np.random.seed(0)
	theta0 = np.random.normal(scale=0, size=(1025, 6))
	temp = 0
	for itera in itera_:
		temp = itera - temp
		theta0, cost_func_ = grad_descent(f_multi, df_multi, X_train, Y_train, theta0, 0.000004, max_iter=temp)
		temp = itera
		perf_.append(performance_multi(X_vali, Y_vali, theta0))
		
	plt.plot(itera_, perf_)
	plt.xlabel("Maximum number of iterations")
	plt.ylabel('Performance on Validation Set')

	fig.savefig(dirpath + '/part7_2.jpg')
	plt.show()
	return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# part8
def part8():
	act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	for i in range(theta_part7.shape[1]):
		theta_reshaped = np.reshape(theta_part7[1:, i], (32, 32))
		fig = plt.figure(i)
		plt.title(act1[i])
		plt.imshow(theta_reshaped, cmap='RdBu')
		fig.savefig(dirpath + '/theta_for_' + act1[i] + '.jpg')
		plt.show()
	
	return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main Block
if __name__ == "__main__":
	#os.chdir(os.path.dirname(__file__))
	dirpath = os.getcwd()
	#part1()
	act_data = part2()
	part3_alpha()
	theta_part3 = part3()
	part4a()
	part4b()
	theta_part5 = part5()
	part6d()
	part7_alpha()
	part7_itera()
	theta_part7 = part7()
	part8()