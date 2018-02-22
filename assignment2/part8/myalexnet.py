
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# a list of class names
from caffe_classes import class_names

import torch.nn as nn


def Resize(S, grayed = False):
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


def get_data(S, act, grayed = False, s = 0):
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
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

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
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, )
            nn.Softmax()
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# model_orig = torchvision.models.alexnet(pretrained=True)
model = MyAlexNet()
model.eval()

# Read an image
im = imread('kiwi227.png')[:,:,:3]
im = im - np.mean(im.flatten())
im = im/np.max(np.abs(im.flatten()))
im = np.rollaxis(im, -1).astype(float32)

# turn the image into a numpy variable
im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    

# run the forward pass AlexNet prediction
softmax = torch.nn.Softmax()
all_probs = softmax(model.forward(im_v)).data.numpy()[0]
sorted_ans = np.argsort(all_probs)

for i in range(-1, -6, -1):
    print("Answer:", class_names[sorted_ans[i]], ", Prob:", all_probs[sorted_ans[i]])

ans = np.argmax(model.forward(im_v).data.numpy())
prob_ans = softmax(model.forward(im_v)).data.numpy()[0][ans]
print("Top Answer:", class_names[ans], "P(ans) = ", prob_ans)




def part10():
    



