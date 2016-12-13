#
# Contains helpful functions and tools 
#
# =============================================================================

import tables
from random import shuffle
from math import ceil
import numpy as np
from keras.utils import np_utils
from keras import backend as K


def euclidean_distance(vects):
    """
    calculate the euclidean distance between two vectors.
    
    Parameters
    ----------
    vects: list
        List of two vectors to calculate the euclidean distance
        
    Returns
    -------
    euclidean_distance : float. 
        Euclidean distance between two vectors
    """    
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)


def coupled_loss(y_true, y_pred):
    """
    A custom Keras objective function which calculates the coupled loss. y_true should be an array of ones then minimizing 
    the coupled_loss will minimize y_pred which is the Euclidean distance
    """    
    return K.mean(y_true * K.square(y_pred))


def image_loader(batch_size, hdf5_path, nb_class):
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
    
   
def coupled_image_loader(batch_size, hdf5_path, nb_class):
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
    [images_l, images_r]: list of numpy arrays. shape = 2 * (batch_size, channels, image_height, image_width)
        Batch images - left and right
    [labels, labels, np.ones((labels.shape[0],1))]: list of numpy arrays. 2 * shape = (batch_size, nb_class) and a shape = (batch_size)
        Batch labels - left and right branches and the coupled layers
    """
    # open the hdf5 file and read the training means
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    ml = hdf5_file.root.mean_l[0]
    ml = ml[np.newaxis, ...]
    
    mr = hdf5_file.root.mean_r[0]
    mr = mr[np.newaxis, ...]    
    
    # Total number of samples
    data_num = hdf5_file.root.images_l.shape[0]
    
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
        images_l = hdf5_file.root.images_l[i_s:i_e] - ml
        images_r = hdf5_file.root.images_r[i_s:i_e] - mr
        
        # read labels and convert to one hot encoding
        labels = hdf5_file.root.labels[i_s:i_e]
        labels = np_utils.to_categorical(np.array(labels), nb_classes=nb_class)
        
        print n,'/',len(batches_list)
            
        yield [images_l, images_r], [labels, labels, np.ones((labels.shape[0],1))]
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
    model.add(Flatten(name='flatten'))
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


def build_coupled_vgg_model(weight_path=None, nb_class=1000, nb_fc=[4096, 4096], img_width=224, img_height=224, 
                    include_top=True, include_drop=True, include_soft=True, coupled_layer='flatten'):
    """
    Create two coupled VGG16 models and load weights.
    
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
    coupled_layer: string
        the name of coupled layer
    
    Returns
    -------
    model: Keras Sequential model
        VGG model
    """    
    from keras.layers import Lambda
    from keras.models import Model
    
    # The left branch VGG16
    model_left = build_vgg_model(weight_path=None, nb_class=nb_class, nb_fc=nb_fc, img_width=img_width, img_height=img_height, 
                    include_top=include_top, include_drop=include_drop, include_soft=include_soft)
    
    # Rename the name of layers in left branch VGG16 to prevent name conflict with the right branch
    for i in range(1, len(model_left.layers)):
        if model_left.layers[i].name == coupled_layer:
            coupled_layer_index = i
        model_left.layers[i].name = model_left.layers[i].name + '_l'
     
    # The right branch VGG16   
    model_right = build_vgg_model(weight_path=None, nb_class=nb_class, nb_fc=nb_fc, img_width=img_width, img_height=img_height, 
                include_top=include_top, include_drop=include_drop, include_soft=include_soft)
    
    # Rename the name of layers in right branch VGG16 to prevent name conflict with the left branch
    for i in range(1, len(model_right.layers)):
        model_right.layers[i].name = model_right.layers[i].name + '_r'
    
    # A custom layer which its output is the Euclidean distance
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name="coupled_loss")(
    [model_right.layers[coupled_layer_index].output, model_left.layers[coupled_layer_index].output])
    
    # The main model which contains both left and right branches   
    model = Model(input=[model_left.input, model_right.input], output=[model_left.output, model_right.output, distance])    
    
    # if weight_path!=None load the provided weights
    if not weight_path is None:
        model.load_weights(weight_path)
        
    return model
