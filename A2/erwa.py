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

from scipy.io import loadmat

import os

# Constants
# -----------------------------------------------------------------------------
input_dim = 28
input_size = 28*28
num_digits = 10

# Functions used in multiple parts
# -----------------------------------------------------------------------------
def load_mnist():
    """
    load mnist data base and divide by 255
    """
    M = loadmat("mnist_all.mat")
    for i in range(num_digits):
        M['train'+str(i)]=M['train'+str(i)]/255.
        M['test'+str(i)]=M['test'+str(i)]/255.
    return M

def create_sets_without_validation():
    """
    generate input and desired output arrays
    """
    M = load_mnist()
    training_x = np.empty((0,input_size))
    test_x = np.empty((0,input_size))
    training_y = np.empty((0,num_digits))
    test_y = np.empty((0,num_digits))
    for i in range(num_digits):
        training_x = np.vstack((training_x,M['train'+str(i)]))
        test_x = np.vstack((test_x,M['test'+str(i)]))
        temp_train_y = np.zeros((M['train'+str(i)].shape[0],num_digits))
        temp_train_y[:,i]=1
        training_y = np.vstack((training_y,temp_train_y))
        temp_test_y = np.zeros((M['test'+str(i)].shape[0],num_digits))
        temp_test_y[:,i]=1
        test_y = np.vstack((test_y,temp_test_y))
    return training_x, training_y, test_x, test_y

def create_sets(v_size_per_digit=0):
    """
    generate input and desired output arrays
    """
    if not v_size_per_digit==0:
        M = load_mnist()
        
        training_x = np.empty((0,input_size))
        validation_x = np.empty((0,input_size))
        test_x = np.empty((0,input_size))
        
        training_y = np.empty((0,num_digits))
        validation_y = np.empty((0,num_digits))
        test_y = np.empty((0,num_digits))
    
        for i in range(num_digits):
            
            full_x = M['train'+str(i)]
            full_y = np.zeros((M['train'+str(i)].shape[0],num_digits))
            full_y[:,i]=1
            combined_xy = np.hstack((full_x,full_y))
            np.random.shuffle(combined_xy)
            
            training_x = np.vstack((training_x,combined_xy[:-v_size_per_digit,:-num_digits]))
            training_y = np.vstack((training_y,combined_xy[:-v_size_per_digit,-num_digits:]))
            
            validation_x = np.vstack((validation_x,combined_xy[-v_size_per_digit:,:-num_digits]))
            validation_y = np.vstack((validation_y,combined_xy[-v_size_per_digit:,-num_digits:]))        
            
            test_x = np.vstack((test_x,M['test'+str(i)]))
            temp_test_y = np.zeros((M['test'+str(i)].shape[0],num_digits))
            temp_test_y[:,i]=1
            test_y = np.vstack((test_y,temp_test_y))
    else:
        validation_x = np.empty((0,input_size))
        validation_y = np.empty((0,num_digits))
        training_x, training_y, test_x, test_y = create_sets_without_validation()
        
    return training_x, training_y, validation_x, validation_y, test_x, test_y

def softmax_forward(y):
    '''
    Return the output of the softmax function for the matrix of output y. 
    Input:
        y: NxM matrix where N is the number of outputs (10) for a single case, and M
    is the number of cases
    '''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def softmax_backward(y,output):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return output-y
    
def tanh_layer(y, W, b):    
    '''
    Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases
    '''
    return tanh(dot(W.T, y)+b)

def fully_connected_layer_forward(x,W,b):
    """
    Return output of fully connected layer 
    Input:
        x: N*M where M = 784
        W: 10*784
        b: 10*1
    Output:
        fully connected: 10*N    
    """
    return np.matmul(W,x.T)+b

def fully_connected_layer_backward(dy,x):
    """
    Return fully connected layer 
    Input:
        dy: 10*N
        x: N*M where M = 784    N*784
        W: 10*784
        b: 10*1
    Output:
        dW: 10*784
        db: 10*1
    """
    dW = np.matmul(dy,x)
    db = np.matmul(dy, np.ones([x.shape[0],1]))
    return dW,db

def NLL(y, output):
    """
    Negative log loss function
    """
    return -sum(y*log(output)) 
    
def forward(x, Wb):
    W1 = Wb[:,:-1]
    b1 = Wb[:,-1:]
    L1 = fully_connected_layer_forward(x,W1,b1)
    output = softmax_forward(L1)
    return output

def backward(y,output,x):
    dy = softmax_backward(y,output)
    dW,db = fully_connected_layer_backward(dy,x)
    return np.hstack((dW, db))

def performance(x, y,Wb, print_output = True,report = "test set"):
    """
    report performance on the test set
    """
    output = forward(x,Wb)
    result = 0.
    for i in range(x.shape[0]):
        if argmax(output.T[i]) == argmax(y[i]):
            result+=1
    
    if print_output:
        print "\nPerformance on "+report+ ": " +str(result)+"/"+str(x.shape[0])+"\n"
    
    return result/x.shape[0]
    
def grad_descent(f, df, loss, x, y, init_t,ada_learning_rate = True, learning_curve=False, \
                 tx=[], ty=[], figure = 'default',momentum = True, damping = 0.9, alpha=0.0001, max_iter=500, EPS = 1e-5):
    """
    Input:
        ada_learning_rate: adaptive learning rate flag
        momentum: momentum flag
        damping: momentum danping ratio
    Output:
    """
    v = np.zeros((init_t.shape[0],init_t.shape[1])) #used for momentum
    
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
    currloss = loss(y, f(x,t)) # used for adaptive alpha
    curr_alpha = alpha # used for adaptive alpha
    
    iter  = 0
    num_iter = []
    performance_training = []
    performance_test = []
    
    while norm(t - prev_t) >  EPS and iter < max_iter:
        
        #plot learning curve
        if learning_curve and iter % 5 == 0:
            num_iter.append(iter)
            performance_training.append(performance(x, y.T, t,print_output = False))
            performance_test.append(performance(tx, ty.T, t,print_output = False))
        
        #normal gradient descent
        prev_t = t.copy()
        if momentum:
            v = damping*v+curr_alpha*df(y, f(x,t), x)
            t -= v
        else:
            t -= curr_alpha*df(y, f(x,t), x)
        
        if iter % 5 == 0:
            print "Iter", iter
            if currloss < loss(y, f(x,t)) and ada_learning_rate:
                curr_alpha = curr_alpha/2
            currloss = loss(y, f(x,t))
            print currloss
            #print "Gradient: ", df(x, y, t), "\n"
        
        iter += 1
    
    if learning_curve:
        plt.plot(num_iter, performance_training,'-')
        plt.plot(num_iter, performance_test,'-')
        plt.legend(['training', 'test'])
        plt.xlabel('Number of Iterations')
        plt.ylabel('Performance')
        plt.savefig('figures/'+figure+'.jpg')
        plt.show()
    
    return t

# Part 1
# -----------------------------------------------------------------------------
def part1():
    """
    Done!
    Print required figures
    add axies and sub-titles if needed 
    """
    try:
        os.makedirs("figures")
    except OSError:
        pass
    
    np.random.seed(0)
    
    #Load the MNIST digit data
    M = load_mnist()
    
    f, axarr = plt.subplots(10, 10)
    for i in range(10):
        np.random.shuffle(M["train"+str(i)])
        for j in range(10):
            axarr[i, j].imshow(M["train"+str(i)][j].reshape((28,28)), cmap=cm.gray)
            axarr[i, j].axes.get_xaxis().set_visible(False)
            axarr[i, j].axes.get_yaxis().set_visible(False)
    plt.savefig('figures/part1f1.jpg')
    plt.show()
    
# Part 2
# -----------------------------------------------------------------------------    
def part2():
    """
    Done!
    Nothing much to do for this function.
    required funcitons are defined above.
    """
    np.random.seed(0)
    
    #create weights
    Wb = np.random.normal(0.,0.5,[num_digits,input_size+1])
    
    #Load the MNIST digit data, and provide an example input
    M = load_mnist()
    x = M["train5"][148:149]
    
    # Calculate the output, L1 is the node values in the first layer
    output = forward(x, Wb)
    
    print "The output example is:"
    print output
    
    return 0

# Part 3
# -----------------------------------------------------------------------------
def part3_grad_check(training_x, training_y):
    """
    To do: improve this function to perform gradient check and include in the report
    """
    h = 0.0000001
    #create weights
    Wb = np.random.normal(0.,0.5,[num_digits,input_size+1])
    
    output = forward(training_x, Wb)
    dWb = backward(training_y.T,output,training_x)
    
    error = []
    sample = 10
    while sample > 0:
        row = int(round((num_digits-1)*np.random.rand()))
        col = int(round((input_size)*np.random.rand()))
        if not dWb[row,col] == 0:
            Wbp = Wb.copy()
            Wbp[row,col] += h
            outputp = forward(training_x, Wbp)
            finite_diff = (NLL(training_y.T, outputp)-NLL(training_y.T, output))/h
            error.append(abs(finite_diff-dWb[row,col])/(abs(finite_diff)+abs(dWb[row,col])))
            sample -= 1
    
    error_mean = np.mean(error)
    error_std = np.std(error)
    print "Relative error is:"+ str(error_mean)+" +/- "+str(error_std)
    
    return 0

def part3():
    """
    Perform gradient check
    """
    np.random.seed(1)
    
    #Load the MNIST digit data
    training_x, training_y,validation_x,validation_y, test_x, test_y = create_sets(v_size_per_digit=0)
    
    part3_grad_check(training_x, training_y)
    
    return 0
 
# Part 4
# -----------------------------------------------------------------------------
def part4():
    try:
        os.makedirs("figures")
    except OSError:
        pass
    
    #Load the MNIST digit data
    training_x, training_y,validation_x,validation_y, test_x, test_y = create_sets(v_size_per_digit=10)     
    #create weights    
    np.random.seed(0)
    Wb = np.random.normal(0.,0.01,[num_digits,input_size+1])
    trained_Wb = grad_descent(forward, backward, NLL, training_x, training_y.T, Wb,momentum = False,learning_curve=True, max_iter=1000, alpha=0.0001,figure = "part4f1", tx=test_x, ty=test_y.T)
    trained_W = trained_Wb[:,:-1]

    for i in range(trained_W.shape[0]):
        plt.figure(i+1)
        plt.imshow((trained_W[i,:]).reshape(28,28),interpolation='none', cmap=cm.coolwarm)
        print "Digit "+str(i)+ " is: \n"
        plt.show()
        plt.imsave("figures/part4f"+str(i+2)+".jpg",(trained_W[i,:]).reshape(28,28), cmap=cm.coolwarm)

    """
    f, axarr = plt.subplots(5, 2)
    for i in range(5):
        for j in range(2):
            axarr[i, j].imshow((trained_W[i*2+j,:]).reshape(28,28),interpolation='none', cmap=cm.coolwarm)
            axarr[i, j].axes.get_xaxis().set_visible(False)
            axarr[i, j].axes.get_yaxis().set_visible(False)
    plt.savefig('figures/part1f2.jpg')
    plt.show()
    """
    return 0

# Part 5
# -----------------------------------------------------------------------------
def part5():
    try:
        os.makedirs("figures")
    except OSError:
        pass
    
    #Load the MNIST digit data
    training_x, training_y,validation_x,validation_y, test_x, test_y = create_sets(v_size_per_digit=10)     

    np.random.seed(0)
    #create weights
    Wb = np.random.normal(0.,0.5,[num_digits,input_size+1])
    trained_Wb = grad_descent(forward, backward, NLL, training_x, training_y.T, Wb,learning_curve=True,figure = "part4f2", tx=test_x, ty=test_y.T)
    
    return 0

# Part 6
# -----------------------------------------------------------------------------
def part6_get_loss_2var(w1,w2,w1_r,w1_c,w2_r,w2_c,Wb,x,y):
    var_Wb = Wb.copy()
    var_Wb[w1_r,w1_c] = w1
    var_Wb[w2_r,w2_c] = w2
    output = forward(x,var_Wb)
    return NLL(y.T, output)

def part6_grad_descent_2var(f, df, loss, x, y, init_t, w1,w2,w1_r,w1_c,w2_r,w2_c,ada_learning_rate = True,momentum = True, damping = 0.9, alpha=0.0001, max_iter=50, EPS = 1e-5):
    """
    Input:
        ada_learning_rate: adaptive learning rate flag
        momentum: momentum flag
        damping: momentum danping ratio
    Output:
    """
    v = np.zeros((init_t.shape[0],init_t.shape[1])) #used for momentum
    
    var_t = init_t.copy()
    var_t[w1_r,w1_c] = w1
    var_t[w2_r,w2_c] = w2    

    prev_t = var_t-10*EPS
    t = var_t.copy()
    
    traj = []
    
    currloss = loss(y, f(x,t)) # used for adaptive alpha
    curr_alpha = alpha # used for adaptive alpha
    
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        if iter % 1 == 0:
            traj.append((t[w1_r,w1_c],t[w2_r,w2_c]))
        #normal gradient descent
        prev_t = t.copy()
        if momentum:
            v = damping*v+curr_alpha*df(y, f(x,t), x)
            t[w1_r,w1_c] -= v[w1_r,w1_c]
            t[w2_r,w2_c] -= v[w2_r,w2_c]
        else:
            t[w1_r,w1_c] -= curr_alpha*(df(y, f(x,t), x)[w1_r,w1_c])
            t[w2_r,w2_c] -= curr_alpha*(df(y, f(x,t), x)[w2_r,w2_c])
        
        if iter % 5 == 0:
            print "Iter", iter
            if currloss < loss(y, f(x,t)) and ada_learning_rate:
                curr_alpha = curr_alpha/2
            currloss = loss(y, f(x,t))
            print currloss
            #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    
    return traj

def part6_train(training_x, training_y,validation_x,validation_y, test_x, test_y):
    #create weights
    Wb = np.random.normal(0.,0.001,[num_digits,input_size+1])
    trained_Wb = grad_descent(forward, backward, NLL, training_x, training_y.T, Wb,alpha=0.000001)
    
    np.save("data/trained_Wb",trained_Wb)

def part6():
    try:
        os.makedirs("figures")
    except OSError:
        pass
    try:
        os.makedirs("data")
    except OSError:
        pass
    
    np.random.seed(0)
    
    #Load the MNIST digit data
    training_x, training_y,validation_x,validation_y, test_x, test_y = create_sets(v_size_per_digit=0)    
    
    #train the nn, only need to run once
    #part6_train(training_x, training_y,validation_x,validation_y, test_x, test_y)
    
    np.random.seed(2)
    
    trained_Wb = np.load("data/trained_Wb.npy")
    
    #ramdomly choose W1 and W2 at the center of the image
    w1_c = 28*(input_dim/2  + int(round(6*np.random.rand())) - 3) + input_dim/2  + int(round(6*np.random.rand())) - 3
    w2_c = 28*(input_dim/2  + int(round(6*np.random.rand())) - 3) + input_dim/2  + int(round(6*np.random.rand())) - 3
    w1_r = int(round((num_digits-1)*np.random.rand()))
    w2_r = int(round((num_digits-1)*np.random.rand()))
    
    #contour plot
    w1s = np.arange(-2, 2, 0.1)
    w2s = np.arange(-2, 2, 0.1)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            C[j,i] = part6_get_loss_2var(w1,w2,w1_r,w1_c,w2_r,w2_c,trained_Wb,training_x,training_y)
    
    init_w1 = -1.6
    init_w2 = -1.6        
    gd_traj = part6_grad_descent_2var(forward, backward, NLL, training_x, training_y.T, trained_Wb, init_w1,init_w2,w1_r,w1_c,w2_r,w2_c, alpha=0.0025,ada_learning_rate = False, momentum = False)
    mo_traj = part6_grad_descent_2var(forward, backward, NLL, training_x, training_y.T, trained_Wb, init_w1,init_w2,w1_r,w1_c,w2_r,w2_c, alpha=0.0025,ada_learning_rate = False, damping = 0.4)
    
    CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='top left')
    plt.title('Contour plot')
    plt.savefig('figures/part6f1.jpg')
    plt.show()
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    #part1()
    #part2()
    #part3()
    #part4()
    #part5()
    #part6()
    
    