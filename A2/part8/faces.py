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


def get_sha256(filename):
    '''
    input: filename
    output: correspoding SHA-256 value
    '''
    sha256 = hashlib.sha256()
    sha256.update(filename)
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

                    sha256 = get_sha256(filename)
                    if sha256 == line.split()[6]:      #SHA-256 check

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



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main Block
if __name__ == "__main__":
    #os.chdir(os.path.dirname(__file__))
    dirpath = os.getcwd()
    Download()
