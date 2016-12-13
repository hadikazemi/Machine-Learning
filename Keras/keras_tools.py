#
# Contains helpful functions and tools 
#
# =============================================================================

import tables
from random import shuffle
from math import ceil
import numpy as np
from keras.utils import np_utils

def ImageLoaderShuffled(batch_size, hdf5_path, nb_class):
    """
    load data from a hdf5 file in an asynchronous fashion. The file should contain an array namely "images"
    which contains training or testing images (created with "hdf5_builder.py") as well as the training mean
    and labels.
    
    Parameters
    ----------
    batch_size: int
        Number of images to load in each batch
    hdf5_path: string
        path to hdf5 file
    nb_class: int
        number of classes (to create one hot encoding from labels)
        
    Returns
    -------
    images : numpy array. shape = (batch_size, channels, image_height, image_width)
        Batch images
    labels: numpy array. shape =(batch_size, nb_class)
        Batch labels
    """
    # open the hdf5 file and read the training mean
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    mm = hdf5_file.root.mean[0]
    mm = mm[np.newaxis, ...]
    
    # Total number of samples
    data_num = hdf5_file.root.images.shape[0]
    
    # create list of batches to shuffle the data
    batches_list = [i for i in range(int(ceil(float(data_num) / batch_size)))]
    shuffle(batches_list)
    n = 0
    
    # loop over batches
    for i in batches_list:
        n += 1
        i_s = i * batch_size                        # index of the first image in this batch
        i_e = min([(i+1) * batch_size, data_num])   # index of the last image in this batch
        
        # read batch images and remove training mean
        images = hdf5_file.root.images[i_s:i_e] - mm
        
        # read labels and convert to one hot encoding
        labels = hdf5_file.root.labels[i_s:i_e]
        labels = np_utils.to_categorical(np.array(labels), nb_classes=nb_class)
        
        print n,'/',len(batches_list)
            
        yield images, labels
    hdf5_file.close()
    
    
def build_vgg_model(weight_path=None, nb_class=1000, nb_fc=[4096, 4096], img_width=224, img_height=224, 
                    include_top=True, include_drop=True, include_soft=True):
    """
    Create a VGG16 model and load weights.
    
    Parameters
    ----------
    weight_path: string
        A path to the model's weights to load.
    nb_class: int
        number of classes on softmax layer
    nb_fc: [int int]
        number of nodes of the two fully connected layers before the softmax layer 
    img_width, img_height: int
        width and heights of the input image respectively
    include_top: bool
        whether you want the fully connected layers or just the convolutional layers
    include_drop: bool
        whether you want dropout or not
    include_soft: bool
        whether you want softmax layer or not  
    
    Returns
    -------
    model: Keras Sequential model
        VGG model
    """    
    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Dense
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='block1_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='block2_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    if include_top:
        model.add(Dense(nb_fc[0], activation='relu', name='fc1'))
        if include_drop:
            model.add(Dropout(0.5))
        model.add(Dense(nb_fc[1], activation='relu', name='fc2'))
        if include_drop:
            model.add(Dropout(0.5))
        if include_soft:
            model.add(Dense(nb_class, activation='softmax', name='predictions'))
    
    # if weight_path==None load 'imagenet' weights otherwise load the weights
    if weight_path==None:
        from keras.applications.vgg16 import VGG16
        model_tmp = VGG16(weights='imagenet', include_top=True)
        for layer in model_tmp.layers:
            if 'conv' in layer.name:
                weights = layer.get_weights()
                model.get_layer(layer.name).set_weights(weights)
            if 'fc1' in layer.name and nb_fc[0] == 4096:
                weights = layer.get_weights()
                model.get_layer(layer.name).set_weights(weights)
            if 'fc2' in layer.name and nb_fc[1] == 4096:
                weights = layer.get_weights()
                model.get_layer(layer.name).set_weights(weights) 
            if 'predictions' in layer.name and nb_class == 1000:
                weights = layer.get_weights()
                model.get_layer(layer.name).set_weights(weights)                 
    else:
        model.load_weights(weight_path)
        
    return model
