
from pylab import *
import numpy as np
import random
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

os.chdir('c:/Users/Guerzhoy/Desktop/CSC320/CSC320Report (1)/')

#%matplotlib



def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X



    

def get_digit_matrix(img_dir):
    im_files = sorted([img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".jpg"])
    im_shape = array(imread(im_files[0])).shape[:2] # open one image to get the size 
    im_matrix = array([imread(im_file).flatten() for im_file in im_files])
    im_matrix = array([im_matrix[i,:]/(norm(im_matrix[i,:])+0.0001) for i in range(im_matrix.shape[0])])
    return (im_matrix, im_shape)
    

def get_reconstruction(V, im, mean_im):
    coefs = [np.dot(V[i,:], (im-mean_im)) for i in range(V.shape[0])]
    new_im = mean_im.copy()
    for i in range(len(coefs)):
        new_im = new_im + coefs[i]*V[i, :]
    return new_im

def display_25_rand_images(im_matrix,im_shape):
    '''Display 25 components in V'''
    #gray()
    fig = figure()
    for i in range(25):
        num = random.randint(1, 2359)
        im = array(im_matrix[num,:]).reshape(im_shape)
        subplot(5, 5, i+1)
        imshow(im)
        axis('off')
    savefig('randim.jpg')  
    show()
    
 

def display_save_25_comps(V, im_shape):
    '''Display 25 components in V'''
    figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
    savefig('display_save_25_comps.jpg')  
    show()         

def salt_and_pepper_noise(flattened_im, noise_prop):
    im = flattened_im.copy()
    pix_inds = range(len(im))
    perm_inds = np.random.permutation(pix_inds)
    
    im[perm_inds[:int(0.5*noise_prop*len(im))]] = max(im)
    im[perm_inds[int(0.5*noise_prop*len(im)):int(noise_prop*len(im))]] = min(im)
    
    return array(im)


        
def occlusion_noise(flattened_im, num_blocks, size_blocks, im_shape):
    im = flattened_im.reshape(im_shape).copy()
    max_im = max(flattened_im)
    min_im = min(flattened_im)
    for i in range(num_blocks/2):
        r_x = np.random.randint(0, im_shape[0])
        r_y = np.random.randint(0, im_shape[1])
        im[r_x:r_x+size_blocks, r_y:r_y+size_blocks] = max_im

    for i in range(num_blocks/2):
        r_x = np.random.randint(0, im_shape[0])
        r_y = np.random.randint(0, im_shape[1])
        im[r_x:r_x+size_blocks, r_y:r_y+size_blocks] = min_im

    return im.flatten()


def auto_thresh(flattened_im):
    im = flattened_im.copy()
    thr = 0.07; sorted(flattened_im)[int(len(flattened_im)*.65)]
    print(thr)
    im[where(flattened_im>thr)] = 1
    im[where(flattened_im<=thr)] = 0
    return im

#a = auto_thresh(r)
#imshow(a.reshape(im_shape))

#Download and unpack digits from :
#http://programmingcomputervision.com/downloads/pcv_data.zip

#Change this:
letters_dir = 'c:/Users/Guerzhoy/Desktop/CSC320/CSC320Report (1)/a_thumbs/'

im_matrix, im_shape = get_digit_matrix(letters_dir)
for i in range(im_matrix.shape[0]):
    im_matrix[i,:] = im_matrix[i,:]/255.0

V,S,mean_im = pca(im_matrix)

#Problem 1
display_25_rand_images(im_matrix, im_shape)

#Problem 2
imsave('immean.jpg',mean_im.reshape(im_shape))
display_save_25_comps(V, im_shape)

#Problem 3
imsave('5pc.jpg',get_reconstruction(V[:5], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('25pc.jpg',get_reconstruction(V[:25], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('100pc.jpg',get_reconstruction(V[:100], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('150pc.jpg',get_reconstruction(V[:150], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('200pc.jpg',get_reconstruction(V[:200], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('400pc.jpg',get_reconstruction(V[:400], im_matrix[0,:],  mean_im).reshape(im_shape))
imsave('all_pc.jpg',get_reconstruction(V, im_matrix[0,:],  mean_im).reshape(im_shape))

#Problem 4
imsave('a0reconstruction.jpg',get_reconstruction(V[:,], im_matrix[0,:],  mean_im).reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .05)
imsave('a0noise0pt05.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt05reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .1)
imsave('a0noise0pt1.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt1reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .2)
imsave('a0noise0pt2.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt2reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .3)
imsave('a0noise0pt3.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt3reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .45)
imsave('a0noise0pt45.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt45reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[0,:], .60)
imsave('a0noise0pt60.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a0noise0pt60reconstruction200PCs.jpg',r.reshape(im_shape))
#############
imsave('a2reconstruction.jpg',get_reconstruction(V[:,], im_matrix[2,:],  mean_im).reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .05)
imsave('a2noise0pt05.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt05reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .1)
imsave('a2noise0pt1.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt1reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .2)
imsave('a2noise0pt2.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt2reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .3)
imsave('a2noise0pt3.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt3reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .45)
imsave('a2noise0pt45.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt45reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[2,:], .60)
imsave('a2noise0pt60.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a2noise0pt60reconstruction200PCs.jpg',r.reshape(im_shape))
#############
imsave('a10reconstruction.jpg',get_reconstruction(V[:,], im_matrix[10,:],  mean_im).reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .05)
imsave('a10noise0pt05.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt05reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .1)
imsave('a10noise0pt1.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt1reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .2)
imsave('a10noise0pt2.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt2reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .3)
imsave('a10noise0pt3.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt3reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .45)
imsave('a10noise0pt45.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt45reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[10,:], .60)
imsave('a10noise0pt60.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a10noise0pt60reconstruction200PCs.jpg',r.reshape(im_shape))
#############
imsave('a15reconstruction.jpg',get_reconstruction(V[:,], im_matrix[15,:],  mean_im).reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .05)
imsave('a15noise0pt05.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt05reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .1)
imsave('a15noise0pt1.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt1reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .2)
imsave('a15noise0pt2.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt2reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .3)
imsave('a15noise0pt3.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt3reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .45)
imsave('a15noise0pt45.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt45reconstruction200PCs.jpg',r.reshape(im_shape))

n = salt_and_pepper_noise(im_matrix[15,:], .60)
imsave('a15noise0pt60.jpg',n.reshape(im_shape))
r = get_reconstruction(V[:100,], n,  mean_im)
r[where(r<0)] = 0
imsave('a15noise0pt60reconstruction200PCs.jpg',r.reshape(im_shape))
#############



rand_letter = np.random.permutation(range(im_matrix.shape[0]))[:8]




