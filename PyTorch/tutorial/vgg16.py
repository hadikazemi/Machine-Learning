from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import imread
import numpy as np
from imagenet_classes import class_names
import tensorflow as tf
from skimage import transform
import os
import tarfile
import urllib

batch_size = 500   # Number of samples in each batch
epoch_num = 4      # Number of epochs to train the network
lr = 0.0005        # Learning rate


class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(7*7*512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5)
        x = self.fc8(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x))
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data[0]

# define the CNN and move the network into GPU
vgg16 = VGG16(10)
vgg16.cuda()

# Download weights
if not os.path.isdir('weights'):
    os.makedirs('weights')
if not os.path.isfile('weights/vgg_16.ckpt'):
    print('Downloading the checkpoint ...')
    urllib.urlretrieve("http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz", "weights/vgg_16_2016_08_28.tar.gz")
    with tarfile.open('weights/vgg_16_2016_08_28.tar.gz', "r:gz") as tar:
        tar.extractall('weights/')
    os.remove('weights/vgg_16_2016_08_28.tar.gz')
    print('Download is complete !')

reader = tf.train.NewCheckpointReader('weights/vgg_16.ckpt')
debug_string = reader.debug_string()

# load the weights from the ckpt file (TensorFlow format)
load_dic = {}
for l in list(vgg16.state_dict()):
    if 'conv' in l:
        tensor_to_load = 'vgg_16/conv{}/{}/{}{}'.format(l[4], l[:7], l[8:], 's' if 'weight' in l else 'es')
        v_tensor = reader.get_tensor(tensor_to_load)
        if 'weight' in l:
            v_tensor = np.transpose(v_tensor, (3, 2, 1, 0))
        else:
            v_tensor = np.transpose(v_tensor)
        load_dic[l] = torch.from_numpy(v_tensor).float()
    if 'fc' in l:
        tensor_to_load = 'vgg_16/fc{}/{}{}'.format(l[2], l[4:], 's' if 'weight' in l else 'es')
        v_tensor = reader.get_tensor(tensor_to_load)
        if 'weight' in l:
            v_tensor = np.transpose(v_tensor, (3, 2, 1, 0))
        else:
            v_tensor = np.transpose(v_tensor)
        load_dic[l] = torch.from_numpy(v_tensor).float()

vgg16.load_state_dict(load_dic)

image = imread('../images/apple.jpg')
image = transform.resize(image, (224, 224, 3), preserve_range=True)
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

image -= np.array([_R_MEAN, _G_MEAN, _B_MEAN])
image = np.moveaxis(image, 2, 0)
image = image[None]

image = torch.from_numpy(image).float()  # convert the numpy array into torch tensor
image = Variable(image).cuda()           # create a torch variable and transfer it into GPU

m, ind = torch.max(vgg16.predict(image), 1)
print(m.data[0][0], '\n', ind.data[0][0])
print(class_names[ind.data[0][0]])

