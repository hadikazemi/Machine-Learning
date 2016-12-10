####
#
# This file read a list of image addresses and their corresponding labels from a pickle file.
# Then, read all images and save on a hdf5 file. It does not matter how big is the size of the data. 
#
####

import cv2
import numpy as np
import tables
import pickle
from random import shuffle

# Parameters to set 
data_order = 'th'               # 'th' for Theano, 'tf' for Tensorflow
img_dtype = np.dtype(np.uint8)  # dtype in which the images will be saved
shuffle_data = True             # shuffle the addresses before saving 
hdf5_path = 'dataset.hdf5'      # address to save the hdf5 file
addrs_file = 'addresses.pkl'    # path to a pickle file which contains image addresses and labels

# read addresses and labels from a pickle file
f = open(addrs_file, 'r')
addrs = pickle.load(f)
labels = pickle.load(f)
f.close()

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = tables.open_file(hdf5_path, mode='w')
image_storage = hdf5_file.create_earray(hdf5_file.root, 'images', tables.Atom.from_dtype(img_dtype), shape=data_shape)
mean_storage = hdf5_file.create_earray(hdf5_file.root, 'mean', tables.Atom.from_dtype(np.dtype(float)), shape=data_shape)

# create labels array and copy the labels data in it
hdf5_file.create_array(hdf5_file.root, 'labels', labels)
  
# a numpy array to save the mean of the images
mean = np.zeros(data_shape[1:], np.float32)

# loop over addresses
for i in range(len(addrs)):
    # print how many images are saved every 1000 images
    if i%1000 ==0 and i > 1:
        print '{}/{}'.format(i, len(addrs)) 
        
    # read an image and resize to (224, 224) 
    # cv2 load images as BGR, convert it to RGB (does not affect performance)
    addr = addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img,(224, 224), interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # add any image pre-processing here
    
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
        
    # save the image and calculate the mean so far
    image_storage.append(img[None])
    mean += img / float(len(labels))  

# save the mean and close the hdf5 file
mean_storage.append(mean[None])
hdf5_file.close()
