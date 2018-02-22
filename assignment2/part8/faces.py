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
import hashlib
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils

##############################################################################
#################### Part 8: Download and Process Data #######################

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


def get_sha256(filename):
    '''
    input: str filename
    output: correspoding SHA-256 value
    '''
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        sha256.update(f.read())
    return sha256.hexdigest()

def Download():
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


                    #SHA-256 check
                    sha256 = get_sha256(uncropped_path + filename)
                    if sha256 != line.split()[6]:      
                        print "SHA-256 Checking fails for %s, now trying to delete it" %filename
                        try:
                            os.remove(uncropped_path + filename)
                            print "Delete successfully!"
                        except:
                            print "Cannot remove the file!!"
                        continue

                    #Crop images
                    try:
                        uncropped = imread(uncropped_path + filename)
                        crop_coor = line.split("\t")[4].split(",")
                        if (uncropped.ndim) == 3:
                            cropped = uncropped[int(crop_coor[1]):int(crop_coor[3]),
                                      int(crop_coor[0]):int(crop_coor[2]), :]
                        elif (uncropped.ndim) == 2:
                            cropped = uncropped[int(crop_coor[1]):int(crop_coor[3]),
                                      int(crop_coor[0]):int(crop_coor[2])]
                    except:
                        continue

                    plt.imsave(cropped_path + filename, cropped)

                    print filename
                    i += 1

    return

def Resize(S, grayed = True):
    '''
    input: a turple of size. e.g. (32,32)
    Resize the cropped images, and store them into XXX_size(_grayed)
    hardcode_list is a list that contains filenames 
    '''
    hardcode_list = ['Kristin Chenoweth71.jpg']
    for act_type in ['actors', 'actresses']:
        if grayed:
            store_path = dirpath + "/facescrub_%s_%i_grayed/" %(act_type, S[0])
        else:
            store_path = dirpath + "/facescrub_%s_%i/" %(act_type, S[0])
        os.makedirs(store_path)
        dataset_path = dirpath + "/facescrub_" + act_type + "_cropped/"
        for root, dirs, files in os.walk(dataset_path):
            dirs.sort()
            files.sort()
            for filename in files:
                if filename not in hardcode_list:
                    im = imread(dataset_path + filename, mode='RGB')
                    resized = imresize(im, S)
                    if grayed:
                        if im.ndim == 3:
                            resized = rgb2gray(resized)
                        elif im.ndim == 2:
                            resized = resized/255.
                        else:
                            continue
                        plt.imsave(store_path + filename, resized, cmap = cm.gray)
                    else:
                        resized = resized/255.
                        plt.imsave(store_path + filename, resized)

##############################################################################
############################### Part 8: Get Data #############################

def get_data(S, act, grayed = True, s = 0):
    '''
    input: S is the size of imgaes, grayed is whether the images are grayed, s is the random seed

    Getting validation sets and training sets.

    return act_data which is a dictionary whose key is the name of actor/actress, value is the 3-element long array that the first element is a array that contains the training set, second one contains validation set, third one contains test set.
    '''
    
    '''
    act_rawdata = {}
    act_data = {}
    for i in act:
        act_rawdata[i] = np.empty((S[0]*S[0],0))
        act_data[i] = [0, 0, 0]

    for act_type in ['actors', 'actresses']:
        if grayed:
            dataset_path = dirpath + "/facescrub_%s_%i_grayed/" %(act_type, S[0])
        else:
            dataset_path = dirpath + "/facescrub_%s_%i/" %(act_type, S[0])
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

    # construct output act_data
    for i in act:
        act_data[i][0] = act_rawdata[i][:, :min(70, act_rawdata[i].shape[1] - 30)]
        act_data[i][1] = act_rawdata[i][:, -20:-10]
        act_data[i][2] = act_rawdata[i][:, -20:]

    return act_data

    '''



    act_rawdata = {}
    act_data = {}
    for i in act:
        act_rawdata[i] = np.empty((S[0]*S[0],0))
        act_data[i] = [0, 0, 0]

    for act_type in ['actors', 'actresses']:
        if grayed:
            dataset_path = dirpath + "/facescrub_%s_%i_grayed/" %(act_type, S[0])
        else:
            dataset_path = dirpath + "/facescrub_%s_%i/" %(act_type, S[0])
        for root, dirs, files in os.walk(dataset_path):
            dirs.sort()
            files.sort()
            for filename in files:
                im = imread(dataset_path + filename)
                im = im/255.
                im = np.array([im[:,:,0].flatten()]).T
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
        act_data[i][0] = act_rawdata[i][:, :min(70, act_rawdata[i].shape[1] - 30)]
        act_data[i][1] = act_rawdata[i][:, -30:-20]
        act_data[i][2] = act_rawdata[i][:, -20:]
    return act_data

##############################################################################
############################### Part 8: Pytorch ##############################

def Run(S, batch_size, nEpoch, alpha = 1e-2, grayed = True, s = 0):
    '''
    input: S is a tuple of size
            batch_size is the size of minbatch used in gradient descent
            nEpoch is the number of epoch it will run
            alpha is the learning rate
            grayed indicates whether the images are grayed
            s is the random seed used
    
    This function:
        1. initialize torch seed
        2. get training, validation, and training sets by calling function get_data
        3. format training, validation, and training sets
        4. define torch variablaes
        5. initilize dataloader which will be used in minibatch gradient descent
        6. define model and loss function
        7. initialize weight
        8. define optimizer
        9. minibatch gradient descent and calculate performance in every 5 epoches
        10. plot learning rate

    return pytorch model

    '''
    act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # set torch seed
    torch.manual_seed(s)

    # get data
    act_data = get_data(S, act1, grayed = grayed, s = s)

    # construct training set, validation set, test set
    X_train = np.empty((S[0]*S[0], 0))
    Y_train = np.empty((len(act1), 0))
    X_vali = np.empty((S[0]*S[0], 0))
    Y_vali = np.empty((len(act1), 0))
    X_test = np.empty((S[0]*S[0], 0))
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

    X_train = X_train.T
    Y_train = Y_train.T
    X_vali = X_vali.T
    Y_vali = Y_vali.T
    X_test = X_test.T
    Y_test = Y_test.T

    # pack X_train and Y_train into one variables
    Y_classes = np.argmax(Y_train, 1).reshape((X_train.shape[0],1))
    XY_train = np.hstack((X_train, Y_classes))

    # define torch variablaes
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    X_TRAIN = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
    X_VALI = Variable(torch.from_numpy(X_vali), requires_grad=False).type(dtype_float)
    X_TEST = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)

    # initilize dataloader which will be used in minibatch gradient descent
    dataloader = data_utils.DataLoader(XY_train, batch_size=batch_size, shuffle=True)


    # define model
    model = torch.nn.Sequential(
        torch.nn.Linear(S[0]*S[0], 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, len(act1)),
        torch.nn.Softmax()
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # initialize weight
    model[0].weight.data.normal_(0.0,0.01)
    model[2].weight.data.normal_(0.0,0.01)
    model[0].bias.data.normal_(0.0,0.01)
    model[2].bias.data.normal_(0.0,0.01)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # gradient descent
    num_epoch = []
    perf_train = []
    perf_vali = []
    perf_test = []

    for Epoch in range(nEpoch):
        for iMinibatch, Minibatch in enumerate(dataloader):
            x = Variable(Minibatch[:,:-1], requires_grad = False).type(dtype_float)
            y = Variable(Minibatch[:,-1], requires_grad = False).type(dtype_long)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to make a step

        # calculate the performance
        if Epoch % 10 == 0:
            print "The current epoch is %i"%Epoch
            num_epoch.append(Epoch)
            perf_train.append(np.mean(np.argmax(model(X_TRAIN).data.numpy(), 1) == np.argmax(Y_train, 1)))
            perf_vali.append(np.mean(np.argmax(model(X_VALI).data.numpy(), 1) == np.argmax(Y_vali, 1)))
            perf_test.append(np.mean(np.argmax(model(X_TEST).data.numpy(), 1) == np.argmax(Y_test, 1)))


    # plot learning curve
    fig = plt.figure(80)
    plt.plot(num_epoch, perf_train,'-1', label = 'training set')
    plt.plot(num_epoch, perf_vali,'-2', label = 'validation set')
    plt.plot(num_epoch, perf_test,'-3', label = 'test set')
    plt.title('Part 8: Learning Rate')
    plt.xlabel('Number of epoches')
    plt.ylabel('Performance')
    plt.legend(loc = 'upper right')
    fig.savefig(dirpath + '/part8_%i_%i_%i_%07.07f.jpg'%(S[0], batch_size, nEpoch, alpha))
    plt.show()

    return model

##############################################################################
#################################### Part 9 ##################################

def part9(S):
    '''
    input: S is a tuple of size

    plot the image of weights generated by part 8
    '''

    for i in range(12):
        fig = plt.figure(90+i)
        plt.imshow(model[0].weight.data.numpy()[i, :].reshape(S), cmap=plt.cm.coolwarm)
        fig.savefig(dirpath + '/part9_%i.jpg'%i)
        plt.show()


##############################################################################
################################## Main Block ################################

if __name__ == "__main__":
    #os.chdir(os.path.dirname(__file__))
    dirpath = os.getcwd()
    #Download()
    #Resize((32,32))
    #model = Run((32,32), 10, 1500, alpha = 3e-4, grayed = True, s = 1)
    part9((32,32))

