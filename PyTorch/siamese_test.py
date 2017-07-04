from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
batch_size = 500   # Number of samples in each batch
epoch_num = 8      # Number of epochs to train the network
lr = 0.0005        # Learning rate
g_size = 10        # number of sample images per each class in the gallery

x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels


def create_gallery(sample_per_class=1):
    # A function to create a gallery of MNIST images,
    # Args:
    #   sample_per_class: number of sample images per each class in the gallery.
    # Returns:
    #   gallery: a 1D list of size 10, each element is a variable with size [sample_per_class, 1, 28, 28].
    gallery = []
    for i in range(10):
        ind_c = np.squeeze(np.argwhere(y_train == i))
        gallery_img = x_train[random.sample(ind_c, sample_per_class)]
        gallery_img = gallery_img.reshape((-1, 1, 28, 28))

        gallery_img = torch.from_numpy(gallery_img).float()  # convert the numpy array into torch tensor
        gallery_img = Variable(gallery_img).cuda()           # create a torch variable and transfer it into GPU
        gallery.append(gallery_img)
    return gallery


def next_batch(batch_number, is_train=True):
    # A function to read the next batch of MNIST images and labels
    # Args:
    #   batch_number: indicates which batch to load.
    #   is_train: indicates whether load the batch from train or test dataset
    # Returns:
    #   batch_x: a pytorch Variable of size [batch_size, 1, 28, 28].
    #   batch_y: a pytorch Variable of size [batch_size, ].
    if is_train:
        dataset_x = x_train
        dataset_y = y_train
    else:
        dataset_x = x_test
        dataset_y = y_test

    batch_x = dataset_x[batch_number*batch_size:min([(batch_number+1)*batch_size, dataset_x.shape[0]]), :]
    batch_y = dataset_y[batch_number*batch_size:min([(batch_number+1)*batch_size, dataset_y.shape[0]])]

    # reshape the sample to a batch of images in pytorch order (batch, channels, height, width)
    batch_x = batch_x.reshape((-1, 1, 28, 28))

    batch_y = torch.from_numpy(batch_y).long()  # convert the numpy array into torch tensor
    batch_y = Variable(batch_y).cuda()          # create a torch variable and transfer it into GPU

    batch_x = torch.from_numpy(batch_x).float()     # convert the numpy array into torch tensor
    batch_x = Variable(batch_x).cuda()              # create a torch variable and transfer it into GPU

    return batch_x, batch_y


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, 7, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64)
        return x


# define the CNN and move the network into GPU
cnn = CNN(10)
cnn.cuda()

cnn.load_state_dict(torch.load('./siamese.pth'))

# extract features for samples in the gallery
gallery = create_gallery(g_size)
gallery_features = [cnn(g) for g in gallery]

# extract features for the test samples
test_img, test_label = next_batch(0, is_train=False)
test_features = cnn(test_img)

# calculate the avergae euclidean distance between each test sample and all the ten classes in the gallery
# the test sample belongs to the class whose distance is minimum to the test sample
true_num = 0
for i in range(batch_size):
    euclidean_distance = [0] * 10
    for j in range(10):
        print(gallery_features[j].size(), test_features.size(), gallery_features[j][0:1].repeat(500, 1).size())
        for g in range(g_size):
            ed = F.pairwise_distance(gallery_features[j][g:g+1], test_features[i:i+1])
            ed = ed.data[0].cpu().numpy()[0]
            euclidean_distance[j] += ed
        euclidean_distance[j] /= float(g_size)
    # check if the prediction is correct
    if np.argmin(np.array(euclidean_distance)) == test_label[i].data[0]:
        true_num += 1

print('Accuracy: ', true_num/float(batch_size))
