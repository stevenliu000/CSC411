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
import torchvision.models as models
import torchvision

# a list of class names
from caffe_classes import class_names

import torch.nn as nn


def Resize(S, grayed = False):
    '''
    input: a turple of size. e.g. (227,227)
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


def get_data(S, act, model, grayed = False, s = 0):
    act_rawdata = {}
    act_data = {}
    for i in act:
        act_rawdata[i] = np.empty((0,9216))
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
                
                # extracting activation data of AlexNet
                im = imread(dataset_path + filename)[:,:,:3]
                im = im - np.mean(im.flatten())
                im = im/np.max(np.abs(im.flatten()))
                im = np.rollaxis(im, -1).astype(np.float32)
                im = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
                im = model.process_X(im)
                for i in act:
                    if i in filename:
                        act_rawdata[i] = np.vstack((act_rawdata[i], im))

    # randomly shuffle act_rawdata
    np.random.seed(s)
    for i in act:
        act_rawdata[i] = act_rawdata[i][np.random.permutation(act_rawdata[i].shape[0]),:]

    for i in act:
        act_data[i][0] = act_rawdata[i][:min(70, act_rawdata[i].shape[1] - 30),:]
        act_data[i][1] = act_rawdata[i][-30:-20,:]
        act_data[i][2] = act_rawdata[i][-20:,:]
    return act_data


# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
        
        # Initilize weights
        classifier_weight_i = [0]
        for i in classifier_weight_i:
            self.classifier[i].weight.data.normal_(0.0,0.01)
            self.classifier[i].bias.data.normal_(0.0,0.01)

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Addition of the extra layer, the linear function
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 6),
            nn.Softmax()
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    # Exrtact the activation on the last layer
    def process_X(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x.data.numpy()

##############################################################################
################################### Part 10 ##################################

def part10(S, batch_size, nEpoch, alpha = 1e-2, grayed = True, s = 0):
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
        (but now we are using alexNet activations as inputs)
        3. format training, validation, and training sets
        4. define torch variablaes
        5. initilize dataloader which will be used in minibatch gradient descent
        6. define loss function
        7. initialize weight
        8. define optimizer
        9. minibatch gradient descent and calculate performance in every 5 epoches
        10. plot learning rate

    return pytorch model

    '''

    # set torch seed
    torch.manual_seed(s)

    # define model
    model = MyAlexNet()

    act1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # get data
    act_data = get_data(S, act1, model, grayed = grayed, s = s)

    # construct training set, validation set, test set
    X_train = np.empty([0,9216])
    Y_train = np.empty((0, len(act1)))
    X_vali = np.empty([0,9216])
    Y_vali = np.empty((0, len(act1)))
    X_test = np.empty([0,9216])
    Y_test = np.empty((0, len(act1)))
    for i in range(len(act1)):
        temp = np.zeros((act_data[act1[i]][0].shape[0],len(act1)))
        temp[:, i] = 1
        X_train = np.vstack((X_train, act_data[act1[i]][0]))
        Y_train = np.vstack((Y_train, temp))

        temp = np.zeros((act_data[act1[i]][1].shape[0],len(act1)))
        temp[:, i] = 1
        X_vali = np.vstack((X_vali, act_data[act1[i]][1]))
        Y_vali = np.vstack((Y_vali, temp))

        temp = np.zeros((act_data[act1[i]][2].shape[0],len(act1)))
        temp[:, i] = 1
        X_test = np.vstack((X_test, act_data[act1[i]][2]))
        Y_test = np.vstack((Y_test, temp))

    # define torch variablaes
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    X_TRAIN = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
    X_VALI = Variable(torch.from_numpy(X_vali), requires_grad=False).type(dtype_float)
    X_TEST = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)


    # pack X_train and Y_train into one variables
    Y_classes = np.argmax(Y_train, 1).reshape((X_train.shape[0],1))
    XY_train = np.hstack((X_train, Y_classes))

    # initilize dataloader which will be used in minibatch gradient descent
    dataloader = data_utils.DataLoader(XY_train, batch_size=batch_size, shuffle=True)

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=alpha)

    # gradient descent
    num_epoch = []
    perf_train = []
    perf_vali = []
    perf_test = []

    for Epoch in range(nEpoch):
        for iMinibatch, Minibatch in enumerate(dataloader):
            x = Variable(Minibatch[:,:-1], requires_grad = False).type(dtype_float)
            y = Variable(Minibatch[:,-1], requires_grad = False).type(dtype_long)
            y_pred = model.classifier(x)
            loss = loss_fn(y_pred, y)
            model.classifier.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to make a step

        # calculate the performance
        if Epoch % 10 == 0:
            print "The current epoch is %i"%Epoch
            num_epoch.append(Epoch)
            perf_train.append(np.mean(np.argmax(model.classifier(X_TRAIN).data.numpy(), 1) \
                                      == np.argmax(Y_train, 1)))
            perf_vali.append(np.mean(np.argmax(model.classifier(X_VALI).data.numpy(), 1) \
                                     == np.argmax(Y_vali, 1)))
            perf_test.append(np.mean(np.argmax(model.classifier(X_TEST).data.numpy(), 1) \
                                     == np.argmax(Y_test, 1)))


    # plot learning curve
    fig = plt.figure(100)
    plt.plot(num_epoch, perf_train,'-1', label = 'training set')
    plt.plot(num_epoch, perf_vali,'-2', label = 'validation set')
    plt.plot(num_epoch, perf_test,'-3', label = 'test set')
    plt.title('Part 10: Learning Rate')
    plt.xlabel('Number of epoches')
    plt.ylabel('Performance')
    plt.legend(loc = 'lower right')
    fig.savefig(dirpath + '/part10_%i_%i_%i_%07.07f.jpg'%(S[0], batch_size, nEpoch, alpha))
    plt.show()

    return model


##############################################################################
################################## Main Block ################################

if __name__ == "__main__":
    #os.chdir(os.path.dirname(__file__))
    dirpath = os.getcwd()
    #Download()
    #Resize((227,227), grayed = False)
    model = part10((227,227), 10, 600, alpha = 3e-4, grayed = False, s = 1)



